"""§1.2/§7's deltas module: 1h/24h/72h/since-GW-deadline deltas over the
distilled time series (§2.3: delta-only Parquet shards, one row per player
only when a tracked value changed since the previous snapshot). Feeds
`analytics/trending.py`'s market signals.

A player's value "as of" any past timestamp is the most recent snapshot at
or before that timestamp — the same point-in-time query
`collector/snapshot.py:latest_distilled_state` already uses for "now",
generalized here to an arbitrary cutoff. Queried across *every* gameweek's
shard directory at once (`gw*/*.parquet`), not just the current one — the
per-gameweek directory is a storage-organization detail (§2.3), not a real
partition boundary, and a 72h or since-GW window can easily reach back
across a gameweek's own boundary.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import duckdb
import polars as pl

# The distilled columns worth differencing numerically (§2.3's full list).
# `status` (categorical) and `news_added` (a timestamp, not a quantity)
# are carried as current-value-only context instead — see compute_deltas.
NUMERIC_DELTA_COLUMNS = [
    "now_cost", "selected_by_percent", "transfers_in_event", "transfers_out_event",
    "form", "chance_of_playing_next_round", "ep_next",
]
CONTEXT_COLUMNS = ["status", "news_added"]

_STATE_SCHEMA = {
    "element_id": pl.Int64, "now_cost": pl.Int64, "selected_by_percent": pl.Float64,
    "transfers_in_event": pl.Int64, "transfers_out_event": pl.Int64, "form": pl.Float64,
    "status": pl.Utf8, "chance_of_playing_next_round": pl.Int64,
    "news_added": pl.Datetime("us", "UTC"), "ep_next": pl.Float64,
    "snapshot_ts": pl.Datetime("us", "UTC"),
}


def state_as_of(distilled_dir: Path, as_of: datetime) -> pl.DataFrame:
    """Every player's most recent distilled row at-or-before `as_of`,
    across all gameweeks' shards. Empty (typed) frame if nothing qualifies
    — e.g. `as_of` predates the collector's first snapshot."""
    pattern = str(distilled_dir / "gw*" / "*.parquet").replace("\\", "/")
    if not distilled_dir.exists() or not any(distilled_dir.glob("gw*/*.parquet")):
        return pl.DataFrame(schema=_STATE_SCHEMA)
    con = duckdb.connect()
    try:
        # DuckDB otherwise returns TIMESTAMPTZ in the host's local time
        # zone, which then fails to compare against our UTC `as_of`.
        con.execute("SET TimeZone='UTC'")
        query = f"""
            SELECT * EXCLUDE (rn) FROM (
                SELECT *, row_number() OVER (PARTITION BY element_id ORDER BY snapshot_ts DESC) AS rn
                FROM read_parquet('{pattern}')
                WHERE snapshot_ts <= ?
            ) WHERE rn = 1
        """
        return con.execute(query, [as_of]).pl()
    finally:
        con.close()


def reference_timestamps(as_of: datetime, since_gw_deadline: datetime | None = None) -> dict[str, datetime]:
    """The standard window set (§1.2: "1h/24h/72h/since-GW"). `since_gw_deadline`
    is the current gameweek's own deadline — omit it (e.g. no event has a
    known deadline yet) and only the three fixed windows are returned."""
    windows = {
        "1h": as_of - timedelta(hours=1),
        "24h": as_of - timedelta(hours=24),
        "72h": as_of - timedelta(hours=72),
    }
    if since_gw_deadline is not None:
        windows["since_gw"] = since_gw_deadline
    return windows


def compute_deltas(distilled_dir: Path, as_of: datetime, reference_times: dict[str, datetime]) -> pl.DataFrame:
    """element_id, current values of every tracked column, plus
    `{col}_delta_{label}` for each numeric column x each entry in
    `reference_times`. A player with no snapshot at-or-before a given
    reference time (new to the dataset since then) gets a null delta for
    that window, not a fabricated zero.
    """
    current = state_as_of(distilled_dir, as_of)
    if current.height == 0:
        return current

    result = current.select("element_id", *NUMERIC_DELTA_COLUMNS, *CONTEXT_COLUMNS)
    for label, ref_time in reference_times.items():
        past = state_as_of(distilled_dir, ref_time)
        if past.height == 0:
            for col in NUMERIC_DELTA_COLUMNS:
                result = result.with_columns(pl.lit(None, dtype=pl.Float64).alias(f"{col}_delta_{label}"))
            continue
        past_renamed = past.select("element_id", *[pl.col(c).alias(f"{c}__past") for c in NUMERIC_DELTA_COLUMNS])
        result = result.join(past_renamed, on="element_id", how="left")
        result = result.with_columns(
            [(pl.col(col) - pl.col(f"{col}__past")).alias(f"{col}_delta_{label}") for col in NUMERIC_DELTA_COLUMNS]
        )
        result = result.drop([f"{col}__past" for col in NUMERIC_DELTA_COLUMNS])
    return result
