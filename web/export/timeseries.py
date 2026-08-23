"""`timeseries.parquet` — per-player market history (§5.3.2, §5.4.8).

The Trend Explorer's source: price, ownership and projection over time,
with deadline markers on the x-axis. Built from the distilled tier, which
§2.3 defines as delta-only — a row exists only where a value changed since
the previous snapshot — and which §0.1 calls the sole basis of every
trending metric, because the FPL API has no history endpoint and data not
collected is permanently lost.

Delta-only is the property a consumer most needs to be told about, and it
is invisible in the data itself. A line chart drawn straight from these
rows is a step function sampled at irregular intervals, not a regular
time series: a player whose price never moved has two rows in a month, and
that is a *fact about his price*, not a gap to interpolate across. The
column list stays raw for exactly that reason — the last observed value
carries forward, and forward-filling here would manufacture observations
nobody made.

`gw` comes from the shard's own directory (`data/distilled/gw{n}/`) rather
than from comparing timestamps against deadlines. That is the collector's
own partitioning and therefore the collector's own answer to which
gameweek a snapshot belongs to; re-deriving it here would be a second
opinion that can disagree with the first.

Two projections, and they are not interchangeable:

`ep_next` is **FPL's** expected points, published in the payload. It is
free, present from the first snapshot, and not ours.

`model_projection` is **this repo's**, and exists only where a freeze
does. Freezes are written once per gameweek inside the six hours before
its deadline (§6.1), so this column is a step function at gameweek grain
against `ep_next`'s snapshot grain, and it is null before the first
freeze. That asymmetry is real and the UI should render it as two series
rather than blending them — a null here means "the model had not spoken
yet", which is not a low projection.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

import polars as pl

logger = logging.getLogger("web.export.timeseries")

DISTILLED_DIR = Path("data/distilled")
REFERENCE_DIR = Path("data/reference")
FREEZES_DIR = Path("papertrade/freezes")

# Straight from §2.3's distilled schema. Named rather than globbed so a new
# collector column arrives here deliberately, with a registry entry, rather
# than appearing in the export because it appeared upstream.
DISTILLED_COLUMNS = [
    "snapshot_ts", "element_id", "now_cost", "selected_by_percent",
    "transfers_in_event", "transfers_out_event", "form", "status",
    "chance_of_playing_next_round", "news_added", "ep_next",
]

_GW_DIR = re.compile(r"^gw(\d+)$")

POSITION_BY_ELEMENT_TYPE = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}


def shard_gameweek(path: Path) -> int | None:
    """The gameweek a shard belongs to, from its directory name."""
    match = _GW_DIR.match(path.parent.name)
    return int(match.group(1)) if match else None


def load_distilled(distilled_dir: Path = DISTILLED_DIR) -> pl.DataFrame:
    """Every snapshot the collector has kept, stamped with its gameweek."""
    if not distilled_dir.exists():
        return pl.DataFrame(schema={c: pl.Null for c in DISTILLED_COLUMNS})

    frames = []
    for path in sorted(distilled_dir.glob("**/*.parquet")):
        gw = shard_gameweek(path)
        if gw is None:
            logger.warning("skipping shard outside a gw directory: %s", path)
            continue
        shard = pl.read_parquet(path)
        present = [c for c in DISTILLED_COLUMNS if c in shard.columns]
        frames.append(
            shard.select(present).with_columns(pl.lit(gw, dtype=pl.Int64).alias("gw"))
        )
    if not frames:
        return pl.DataFrame(schema={c: pl.Null for c in DISTILLED_COLUMNS})
    return pl.concat(frames, how="diagonal_relaxed")


def player_identity(reference_dir: Path = REFERENCE_DIR) -> pl.DataFrame:
    """name, team and position per element, so the Explorer can label a
    series without loading the panel — which §5.3.4 does not commit."""
    players = pl.read_parquet(reference_dir / "players.parquet")
    teams = pl.read_parquet(reference_dir / "teams.parquet").select(
        pl.col("id").alias("team"), pl.col("name").alias("team_name")
    )
    return (
        players.select(
            pl.col("id").alias("element_id"),
            pl.col("web_name").alias("name"),
            pl.col("team"),
            pl.col("element_type"),
        )
        .join(teams, on="team", how="left")
        .select(
            "element_id",
            "name",
            pl.col("team_name").alias("team"),
            pl.col("element_type")
            .replace_strict(POSITION_BY_ELEMENT_TYPE, default=None)
            .alias("position"),
        )
    )


def model_projections(freezes_dir: Path = FREEZES_DIR) -> pl.DataFrame:
    """This repo's own projection per (gw, element), from the freezes.

    A freeze covers a horizon — it projects its own gameweek and the two
    after it — and only the projection *for the gameweek it was frozen
    against* is used here. The later two are forecasts made before the
    intervening football happened, and putting them on the same series as
    a deadline-day projection would present a three-week-old guess as
    current.
    """
    empty = pl.DataFrame(
        schema={"gw": pl.Int64, "element_id": pl.Int64, "model_projection": pl.Float64}
    )
    if not freezes_dir.exists():
        return empty

    rows = []
    for path in sorted(freezes_dir.glob("gw*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        gw = payload.get("gameweek")
        own = (payload.get("projections") or {}).get(str(gw), {})
        for element_id, projection in own.items():
            rows.append(
                {"gw": int(gw), "element_id": int(element_id), "model_projection": float(projection)}
            )
    return pl.DataFrame(rows, schema=empty.schema) if rows else empty


def build_timeseries(
    distilled_dir: Path = DISTILLED_DIR,
    reference_dir: Path = REFERENCE_DIR,
    freezes_dir: Path = FREEZES_DIR,
) -> pl.DataFrame:
    """One row per element per snapshot at which something changed."""
    snapshots = load_distilled(distilled_dir)
    if not snapshots.height:
        raise ValueError(
            f"no distilled shards under {distilled_dir} — the collector has not run, "
            "and §0.1 means this history cannot be reconstructed after the fact"
        )

    out = snapshots.join(player_identity(reference_dir), on="element_id", how="left")
    out = out.join(model_projections(freezes_dir), on=["gw", "element_id"], how="left")
    if "model_projection" not in out.columns:  # pragma: no cover - join always adds it
        out = out.with_columns(pl.lit(None, dtype=pl.Float64).alias("model_projection"))

    ordered = [
        "snapshot_ts", "gw", "element_id", "name", "team", "position",
        "now_cost", "selected_by_percent", "transfers_in_event", "transfers_out_event",
        "form", "status", "chance_of_playing_next_round", "news_added",
        "ep_next", "model_projection",
    ]
    return out.select([c for c in ordered if c in out.columns]).sort(
        ["element_id", "snapshot_ts"]
    )


def write_timeseries(df: pl.DataFrame, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "timeseries.parquet"
    df.write_parquet(path, compression="zstd")
    return path
