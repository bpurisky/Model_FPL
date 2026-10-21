"""`ownership.json` -- Elite Ownership: how the top-N managers' squads differ
from the field, and effective ownership (EO).

Source: `data/distilled/gw{n}/top_picks.parquet` (collector `top-picks`, the
picks of the overall league's top N for the latest deadlined gameweek). The
export reads the highest gameweek that has one.

Measures, per player, over the `n_managers` managers actually fetched:

- `top_pct`        share holding him in their 15
- `top_start_pct`  share starting him (multiplier > 0)
- `captain_pct`    share captaining him
- `eo_pct`         `sum(multiplier) / n_managers * 100` (captain 2, triple
                   captain 3, bench 0), the standard effective ownership
- `overall_pct`    field-wide `selected_by_percent`, the latest distilled
                   snapshot at or before the picks' `fetched_at`
- `delta_pct`      `top_pct - overall_pct`

This is descriptive (§5.0.2): it reports what a sample of strong managers
did and makes no claim that copying them is good. "Top N" is a sample of
the *current* top of the table, not a fixed cohort, and an elite manager
with the triple-captain chip is counted at multiplier 3.

With no picks file yet the export writes `status: "no_data"` and empty
lists, so the surface renders its empty state instead of failing.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime
from pathlib import Path

import polars as pl

from analytics.deltas import state_as_of
from web.export.contract import OwnershipChip, OwnershipFile, OwnershipPlayer, build_header
from web.export.timeseries import POSITION_BY_ELEMENT_TYPE

logger = logging.getLogger("web.export.ownership")

DISTILLED_DIR = Path("data/distilled")
REFERENCE_DIR = Path("data/reference")
TOP_PICKS_FILENAME = "top_picks.parquet"
NORMALIZATION_BASIS = "top_n_sample_percent"
# A player nobody in the top N holds and the field barely holds is noise.
MIN_OVERALL_PCT = 0.5

_GW_DIR = re.compile(r"^gw(\d+)$")


def latest_top_picks(distilled_dir: Path) -> tuple[int, Path] | None:
    found = []
    if distilled_dir.exists():
        for d in distilled_dir.iterdir():
            m = _GW_DIR.match(d.name)
            if m and (d / TOP_PICKS_FILENAME).exists():
                found.append((int(m.group(1)), d / TOP_PICKS_FILENAME))
    return max(found) if found else None


def _identity(reference_dir: Path) -> pl.DataFrame:
    players = pl.read_parquet(reference_dir / "players.parquet")
    teams = pl.read_parquet(reference_dir / "teams.parquet").select(
        pl.col("id").alias("team"), pl.col("name").alias("team_name")
    )
    return players.join(teams, on="team", how="left").select(
        pl.col("id").alias("element_id"),
        pl.col("web_name").alias("name"),
        pl.col("team_name").alias("team"),
        pl.col("element_type").replace_strict(POSITION_BY_ELEMENT_TYPE, default=None).alias("position"),
    )


def _empty() -> OwnershipFile:
    return OwnershipFile(
        header=build_header(rows=0, source_gameweek=None, normalization_basis=NORMALIZATION_BASIS),
        status="no_data", gw=None, n_managers=0, fetched_at=None, chips=[], players=[],
    )


def build_ownership(
    *,
    distilled_dir: Path = DISTILLED_DIR,
    reference_dir: Path = REFERENCE_DIR,
) -> OwnershipFile:
    latest = latest_top_picks(distilled_dir)
    if latest is None:
        logger.info("ownership: no top_picks file yet -> no_data")
        return _empty()
    gw, path = latest
    picks = pl.read_parquet(path)
    n = picks["entry_id"].n_unique()
    if n == 0:
        return _empty()
    fetched_at: datetime = picks["fetched_at"].max()

    per_player = picks.group_by("element").agg(
        (pl.col("entry_id").n_unique() / n * 100).alias("top_pct"),
        ((pl.col("multiplier") > 0).sum() / n * 100).alias("top_start_pct"),
        (pl.col("is_captain").sum() / n * 100).alias("captain_pct"),
        (pl.col("multiplier").sum() / n * 100).alias("eo_pct"),
    ).rename({"element": "element_id"})

    state = state_as_of(distilled_dir, fetched_at).select("element_id", "now_cost", "selected_by_percent")
    identity = _identity(reference_dir)
    everyone = identity.join(state, on="element_id", how="left").join(per_player, on="element_id", how="full", coalesce=True)
    everyone = everyone.with_columns(
        pl.col("top_pct").fill_null(0.0), pl.col("top_start_pct").fill_null(0.0),
        pl.col("captain_pct").fill_null(0.0), pl.col("eo_pct").fill_null(0.0),
    ).filter((pl.col("top_pct") > 0) | (pl.col("selected_by_percent").fill_null(0.0) >= MIN_OVERALL_PCT))

    players = [
        OwnershipPlayer(
            element_id=r["element_id"],
            name=r["name"] or f"Player {r['element_id']}",
            team=r["team"] or "",
            position=r["position"] or "",
            now_cost=None if r["now_cost"] is None else int(r["now_cost"]),
            overall_pct=r["selected_by_percent"],
            top_pct=r["top_pct"], top_start_pct=r["top_start_pct"],
            captain_pct=r["captain_pct"], eo_pct=r["eo_pct"],
            delta_pct=None if r["selected_by_percent"] is None else r["top_pct"] - r["selected_by_percent"],
        )
        for r in everyone.to_dicts()
    ]
    players.sort(key=lambda p: (-p.eo_pct, p.element_id))

    chip_counts = (
        picks.filter(pl.col("active_chip").is_not_null())
        .group_by("entry_id").agg(pl.col("active_chip").first())
        .group_by("active_chip").len().sort("len", descending=True)
    )
    chips = [OwnershipChip(chip=r["active_chip"], count=r["len"]) for r in chip_counts.to_dicts()]

    logger.info("ownership: gw%d, %d managers, %d players, chips %s", gw, n, len(players), chips)
    return OwnershipFile(
        header=build_header(rows=len(players), source_gameweek=gw, normalization_basis=NORMALIZATION_BASIS),
        status="ok", gw=gw, n_managers=n, fetched_at=fetched_at, chips=chips, players=players,
    )
