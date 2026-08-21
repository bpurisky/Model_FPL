"""Write raw + distilled snapshots of the FPL API (§2.3).

Three tiers:
  raw        — full gzipped JSON per fetch, 14-day rolling retention. Insurance.
  distilled  — delta-only Parquet shards, one per gameweek directory, permanent.
               The sole basis of every trending metric.
  reference  — teams/fixtures/player metadata, rewritten wholesale each run.
"""

from __future__ import annotations

import gzip
import json
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import duckdb
import polars as pl
from pydantic import ValidationError

from collector.client import FPLClient, FPLNotFoundError
from collector.config import StorageConfig
from collector.schemas import (
    BootstrapStatic,
    Fixture,
    SchemaValidationError,
    parse_bootstrap_static,
    parse_element_summary,
    parse_event_live,
    parse_fixtures,
    resolve_current_event,
    resolve_next_event,
)

logger = logging.getLogger("collector.snapshot")

DISTILLED_COLUMNS = [
    "now_cost",
    "selected_by_percent",
    "transfers_in_event",
    "transfers_out_event",
    "form",
    "status",
    "chance_of_playing_next_round",
    "news_added",
    "ep_next",
]

_TS_FORMAT = "%Y%m%dT%H%M%SZ"


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def iso8601_compact(ts: datetime) -> str:
    return ts.astimezone(timezone.utc).strftime(_TS_FORMAT)


# --------------------------------------------------------------------------
# raw tier
# --------------------------------------------------------------------------


def write_raw(raw_dir: Path, endpoint: str, payload: Any, ts: datetime) -> Path:
    raw_dir.mkdir(parents=True, exist_ok=True)
    path = raw_dir / f"{endpoint}_{iso8601_compact(ts)}.json.gz"
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        json.dump(payload, fh)
    return path


def _parse_raw_timestamp(filename: str) -> datetime | None:
    stem = filename.removesuffix(".json.gz")
    _, _, ts_part = stem.rpartition("_")
    try:
        return datetime.strptime(ts_part, _TS_FORMAT).replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def prune_raw(raw_dir: Path, retention_days: int, now: datetime | None = None) -> list[Path]:
    """Delete raw files older than `retention_days`. Returns the removed paths."""
    now = now or utc_now()
    cutoff = now - timedelta(days=retention_days)
    removed: list[Path] = []
    if not raw_dir.exists():
        return removed
    for path in raw_dir.glob("*.json.gz"):
        ts = _parse_raw_timestamp(path.name)
        if ts is not None and ts < cutoff:
            path.unlink()
            removed.append(path)
    return removed


# --------------------------------------------------------------------------
# distilled tier
# --------------------------------------------------------------------------


def build_distilled_rows(bootstrap: BootstrapStatic, ts: datetime) -> pl.DataFrame:
    def to_float(value: str | None) -> float | None:
        if value in (None, ""):
            return None
        return float(value)

    records = [
        {
            "snapshot_ts": ts,
            "element_id": el.id,
            "now_cost": el.now_cost,
            "selected_by_percent": to_float(el.selected_by_percent),
            "transfers_in_event": el.transfers_in_event,
            "transfers_out_event": el.transfers_out_event,
            "form": to_float(el.form),
            "status": el.status,
            "chance_of_playing_next_round": el.chance_of_playing_next_round,
            "news_added": el.news_added,
            "ep_next": to_float(el.ep_next),
        }
        for el in bootstrap.elements
    ]
    schema = {
        "snapshot_ts": pl.Datetime(time_unit="us", time_zone="UTC"),
        "element_id": pl.Int64,
        "now_cost": pl.Int64,
        "selected_by_percent": pl.Float64,
        "transfers_in_event": pl.Int64,
        "transfers_out_event": pl.Int64,
        "form": pl.Float64,
        "status": pl.Utf8,
        "chance_of_playing_next_round": pl.Int64,
        "news_added": pl.Datetime(time_unit="us", time_zone="UTC"),
        "ep_next": pl.Float64,
    }
    return pl.DataFrame(records, schema=schema)


def latest_distilled_state(gw_dir: Path) -> pl.DataFrame | None:
    """The most recent row per element_id across every shard in `gw_dir`."""
    if not gw_dir.exists() or not any(gw_dir.glob("*.parquet")):
        return None
    pattern = str(gw_dir / "*.parquet").replace("\\", "/")
    con = duckdb.connect()
    try:
        # DuckDB otherwise returns TIMESTAMPTZ columns in the host's local
        # time zone, which then fails to compare against our UTC series.
        con.execute("SET TimeZone='UTC'")
        query = f"""
            SELECT * EXCLUDE (rn) FROM (
                SELECT *, row_number() OVER (PARTITION BY element_id ORDER BY snapshot_ts DESC) AS rn
                FROM read_parquet('{pattern}')
            ) WHERE rn = 1
        """
        return con.execute(query).pl()
    finally:
        con.close()


def _column_changed(new: pl.Series, prev: pl.Series) -> pl.Series:
    """True where the two series disagree, treating null-vs-null as unchanged."""
    null_mismatch = new.is_null() != prev.is_null()
    both_present_diff = (~new.is_null()) & (~prev.is_null()) & (new != prev)
    return null_mismatch | both_present_diff


def compute_changed_rows(new_df: pl.DataFrame, previous_df: pl.DataFrame | None) -> pl.DataFrame:
    """Rows in `new_df` whose tracked columns differ from the last known state.

    A player absent from `previous_df` (new to the dataset, or first snapshot
    of the gameweek) always counts as changed — that's how the baseline gets
    established.
    """
    if previous_df is None or previous_df.height == 0:
        return new_df

    prev = previous_df.select(["element_id", *DISTILLED_COLUMNS])
    merged = new_df.join(prev, on="element_id", how="left", suffix="_prev")

    change_mask = pl.Series([False] * merged.height)
    for col in DISTILLED_COLUMNS:
        prev_col = f"{col}_prev"
        change_mask = change_mask | _column_changed(merged[col], merged[prev_col])

    return merged.filter(change_mask).select(new_df.columns)


def write_distilled_shard(distilled_dir: Path, event_id: int, changed_df: pl.DataFrame, ts: datetime) -> Path | None:
    if changed_df.height == 0:
        return None
    gw_dir = distilled_dir / f"gw{event_id}"
    gw_dir.mkdir(parents=True, exist_ok=True)
    path = gw_dir / f"{iso8601_compact(ts)}.parquet"
    changed_df.write_parquet(path)
    return path


# --------------------------------------------------------------------------
# reference tier
# --------------------------------------------------------------------------


def write_reference(reference_dir: Path, bootstrap: BootstrapStatic, fixtures: list[Fixture]) -> None:
    reference_dir.mkdir(parents=True, exist_ok=True)

    pl.DataFrame([t.model_dump(include={"id", "name", "short_name", "strength"}) for t in bootstrap.teams]).write_parquet(
        reference_dir / "teams.parquet"
    )
    pl.DataFrame(
        [el.model_dump(include={"id", "web_name", "team", "element_type"}) for el in bootstrap.elements]
    ).write_parquet(reference_dir / "players.parquet")
    pl.DataFrame(
        [e.model_dump(include={"id", "name", "deadline_time", "finished", "is_current", "is_next"}) for e in bootstrap.events]
    ).write_parquet(reference_dir / "events.parquet")
    pl.DataFrame(
        [f.model_dump(include={"id", "event", "team_h", "team_a", "kickoff_time", "finished"}) for f in fixtures]
    ).write_parquet(reference_dir / "fixtures.parquet")


# --------------------------------------------------------------------------
# orchestration
# --------------------------------------------------------------------------


@dataclass
class BootstrapSnapshotResult:
    timestamp: datetime
    event_id: int
    raw_paths: list[Path]
    distilled_shard: Path | None
    rows_total: int
    rows_changed: int


async def run_bootstrap_snapshot(client: FPLClient, storage: StorageConfig, ts: datetime | None = None) -> BootstrapSnapshotResult:
    ts = ts or utc_now()
    raw_dir = Path(storage.raw_dir)
    distilled_dir = Path(storage.distilled_dir)
    reference_dir = Path(storage.reference_dir)

    raw_payload = await client.get_json("/bootstrap-static/")
    raw_path = write_raw(raw_dir, "bootstrap-static", raw_payload, ts)
    try:
        bootstrap = parse_bootstrap_static(raw_payload, logger)
    except ValidationError as exc:
        raise SchemaValidationError("bootstrap-static", raw_path, exc) from exc

    fixtures_raw = await client.get_json("/fixtures/")
    fixtures_raw_path = write_raw(raw_dir, "fixtures", fixtures_raw, ts)
    try:
        fixtures = parse_fixtures(fixtures_raw, logger)
    except ValidationError as exc:
        raise SchemaValidationError("fixtures", fixtures_raw_path, exc) from exc

    write_reference(reference_dir, bootstrap, fixtures)

    event_id = resolve_current_event(bootstrap, raw_payload)
    if event_id is None:
        event_id = resolve_next_event(bootstrap, raw_payload)
    if event_id is None:
        raise SchemaValidationError(
            "bootstrap-static", raw_path, RuntimeError("could not resolve current or next event id")
        )

    new_df = build_distilled_rows(bootstrap, ts)
    gw_dir = distilled_dir / f"gw{event_id}"
    previous_df = latest_distilled_state(gw_dir)
    changed_df = compute_changed_rows(new_df, previous_df)
    shard_path = write_distilled_shard(distilled_dir, event_id, changed_df, ts)

    return BootstrapSnapshotResult(
        timestamp=ts,
        event_id=event_id,
        raw_paths=[raw_path, fixtures_raw_path],
        distilled_shard=shard_path,
        rows_total=new_df.height,
        rows_changed=changed_df.height,
    )


@dataclass
class ElementSummarySnapshotResult:
    timestamp: datetime
    raw_path: Path
    fetched: int
    missing: int


async def run_element_summary_snapshot(
    client: FPLClient, storage: StorageConfig, element_ids: list[int], ts: datetime | None = None
) -> ElementSummarySnapshotResult:
    ts = ts or utc_now()
    payloads: dict[str, Any] = {}
    missing = 0
    for element_id in element_ids:
        try:
            payloads[str(element_id)] = await client.get_json(f"/element-summary/{element_id}/")
        except FPLNotFoundError:
            logger.warning("element-summary 404 for element_id=%s", element_id)
            missing += 1

    raw_path = write_raw(Path(storage.raw_dir), "element-summary-batch", payloads, ts)

    for element_id, payload in payloads.items():
        try:
            parse_element_summary(payload, logger, context=f"element-summary[{element_id}]")
        except ValidationError as exc:
            raise SchemaValidationError(f"element-summary[{element_id}]", raw_path, exc) from exc

    return ElementSummarySnapshotResult(timestamp=ts, raw_path=raw_path, fetched=len(payloads), missing=missing)


@dataclass
class EventLiveSnapshotResult:
    timestamp: datetime
    event_id: int
    raw_path: Path


async def run_event_live_snapshot(client: FPLClient, storage: StorageConfig, event_id: int, ts: datetime | None = None) -> EventLiveSnapshotResult:
    ts = ts or utc_now()
    raw_payload = await client.get_json(f"/event/{event_id}/live/")
    raw_path = write_raw(Path(storage.raw_dir), f"event-{event_id}-live", raw_payload, ts)
    try:
        parse_event_live(raw_payload, logger, context=f"event-{event_id}-live")
    except ValidationError as exc:
        raise SchemaValidationError(f"event-{event_id}-live", raw_path, exc) from exc
    return EventLiveSnapshotResult(timestamp=ts, event_id=event_id, raw_path=raw_path)
