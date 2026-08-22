"""Permanent, append-only record of each finished gameweek's actual
per-player results this season (§8: every reported metric must regenerate
from committed data — Phase 0's raw tier is 14-day insurance, not an
archive, and this is `squad/live.py`'s own documented follow-on: "switch to
per-gameweek splits ... before running this again after gw2").

Sourced from `/event/{gw}/live/`, which reports that gameweek's own stats
directly — no cumulative-total delta needed, no 600-request per-player
batch, one call. Schema is a subset of `data/historical/{season}.parquet`'s,
so `backtest/baselines.py` and `backtest/report.py`'s already-tested
functions work against it unmodified once enough gameweeks accumulate.
"""

from __future__ import annotations

import logging
from pathlib import Path

import polars as pl

from collector.client import FPLClient
from collector.config import CollectorConfig
from collector.schemas import BootstrapStatic, parse_bootstrap_static, parse_event_live
from squad.live import POSITION_BY_ELEMENT_TYPE, PROMOTED_CLUB_SHORT_NAMES

logger = logging.getLogger("papertrade.actuals")

ACTUALS_PATH = Path("data/current_season/2026-27.parquet")

_STAT_COLUMNS = [
    "minutes", "goals_scored", "assists", "clean_sheets", "goals_conceded", "saves", "bonus",
    "yellow_cards", "red_cards", "own_goals", "penalties_missed", "penalties_saved",
    "defensive_contribution", "total_points",
]

_SCHEMA = {
    "gw": pl.Int64, "element_id": pl.Int64, "position": pl.Utf8, "team": pl.Utf8, "is_promoted_club": pl.Boolean,
    **{c: pl.Int64 for c in _STAT_COLUMNS},
}


def gw_is_finished(bootstrap: BootstrapStatic, gw: int) -> bool:
    event = next((e for e in bootstrap.events if e.id == gw), None)
    return bool(event and event.finished)


def finished_gws_missing_from(bootstrap: BootstrapStatic, actuals: pl.DataFrame) -> list[int]:
    """Every finished gameweek not already recorded — lets the weekly
    automation catch up in one run if it was skipped or missed a week,
    without needing to know in advance which gw to ask for."""
    recorded = set(actuals["gw"].to_list()) if actuals.height else set()
    return sorted(e.id for e in bootstrap.events if e.finished and e.id not in recorded)


def _build_actuals_frame(bootstrap: BootstrapStatic, live_raw: dict, gw: int) -> pl.DataFrame:
    team_name = {t.id: t.name for t in bootstrap.teams}
    team_short = {t.id: t.short_name for t in bootstrap.teams}
    position_team = {
        e.id: (POSITION_BY_ELEMENT_TYPE[e.element_type], team_name[e.team], team_short[e.team] in PROMOTED_CLUB_SHORT_NAMES)
        for e in bootstrap.elements
    }

    rows = []
    for el in live_raw["elements"]:
        eid = el["id"]
        if eid not in position_team:
            continue
        position, team, is_promoted = position_team[eid]
        stats = el["stats"]
        rows.append({
            "gw": gw, "element_id": eid, "position": position, "team": team, "is_promoted_club": is_promoted,
            **{col: stats[col] for col in _STAT_COLUMNS},
        })
    return pl.DataFrame(rows, schema=_SCHEMA)


async def fetch_gw_actuals(cfg: CollectorConfig, gw: int) -> pl.DataFrame:
    """Refuses to run against a gameweek that hasn't finished — the same
    "no fabricated signal" discipline `squad/live.py:build_train_df` applies
    to an in-progress gameweek's cumulative stats, here applied to whether
    a gw's actuals are final at all rather than which teams have played."""
    async with FPLClient(**cfg.api.client_kwargs()) as client:
        bootstrap_raw = await client.get_json("/bootstrap-static/")
        live_raw = await client.get_json(f"/event/{gw}/live/")

    bootstrap = parse_bootstrap_static(bootstrap_raw, logger)
    parse_event_live(live_raw, logger)  # validated for drift; raw dicts read in _build_actuals_frame for the extra stat fields
    if not gw_is_finished(bootstrap, gw):
        raise RuntimeError(f"gw{gw} is not finished yet — refusing to record partial actuals as if final")

    return _build_actuals_frame(bootstrap, live_raw, gw)


async def fetch_missing_gw_actuals(cfg: CollectorConfig, actuals: pl.DataFrame | None = None) -> list[pl.DataFrame]:
    """One DataFrame per finished-but-unrecorded gameweek, oldest first —
    the auto-detect path the weekly automation uses instead of naming a gw."""
    actuals = load_actuals() if actuals is None else actuals
    async with FPLClient(**cfg.api.client_kwargs()) as client:
        bootstrap_raw = await client.get_json("/bootstrap-static/")
        bootstrap = parse_bootstrap_static(bootstrap_raw, logger)
        missing = finished_gws_missing_from(bootstrap, actuals)

        frames = []
        for gw in missing:
            live_raw = await client.get_json(f"/event/{gw}/live/")
            parse_event_live(live_raw, logger)
            frames.append(_build_actuals_frame(bootstrap, live_raw, gw))
    return frames


def append_actuals(df: pl.DataFrame, path: Path = ACTUALS_PATH) -> Path:
    """Append-only (§2.3's distilled-tier philosophy): refuses to write a
    gameweek that's already recorded, so an evaluation already run against
    this data can't be silently invalidated by a later re-run."""
    if df.height == 0:
        raise ValueError("refusing to append an empty actuals frame")
    gw = df["gw"][0]
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        existing = pl.read_parquet(path)
        if gw in existing["gw"].to_list():
            raise FileExistsError(f"gw{gw} actuals are already recorded in {path} — never overwritten")
        combined = pl.concat([existing, df])
    else:
        combined = df
    combined.write_parquet(path)
    return path


def load_actuals(path: Path = ACTUALS_PATH) -> pl.DataFrame:
    if not path.exists():
        return pl.DataFrame(schema=_SCHEMA)
    return pl.read_parquet(path)
