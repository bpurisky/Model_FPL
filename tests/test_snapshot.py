"""§2.3 / §2.6 acceptance criteria for the raw/distilled/reference tiers."""

from __future__ import annotations

import asyncio
import gzip
import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path

import duckdb
import httpx
import pytest
from pydantic import ValidationError

from collector.client import FPLClient
from collector.config import StorageConfig
from collector.schemas import SchemaValidationError, parse_bootstrap_static
from collector.snapshot import (
    build_distilled_rows,
    compute_changed_rows,
    latest_distilled_state,
    prune_raw,
    run_bootstrap_snapshot,
    write_distilled_shard,
    write_raw,
)


# --------------------------------------------------------------------------
# raw tier
# --------------------------------------------------------------------------


def test_prune_raw_removes_only_stale_files(tmp_path: Path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    now = datetime(2026, 8, 21, 12, 0, tzinfo=timezone.utc)

    old_path = write_raw(raw_dir, "bootstrap-static", {"a": 1}, now - timedelta(days=20))
    recent_path = write_raw(raw_dir, "bootstrap-static", {"a": 2}, now - timedelta(days=1))

    removed = prune_raw(raw_dir, retention_days=14, now=now)

    assert old_path in removed
    assert not old_path.exists()
    assert recent_path.exists()


def test_write_raw_produces_readable_gzip(tmp_path: Path):
    raw_dir = tmp_path / "raw"
    ts = datetime(2026, 8, 21, 12, 0, tzinfo=timezone.utc)
    path = write_raw(raw_dir, "bootstrap-static", {"hello": "world"}, ts)
    assert path.name == "bootstrap-static_20260821T120000Z.json.gz"
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        assert json.load(fh) == {"hello": "world"}


# --------------------------------------------------------------------------
# distilled tier — delta-only
# --------------------------------------------------------------------------


def test_second_identical_snapshot_produces_no_new_rows(bootstrap_payload):
    bootstrap = parse_bootstrap_static(bootstrap_payload, logging.getLogger("test"))
    ts1 = datetime(2026, 8, 21, 10, 0, tzinfo=timezone.utc)
    ts2 = datetime(2026, 8, 21, 11, 0, tzinfo=timezone.utc)

    first = build_distilled_rows(bootstrap, ts1)
    second = build_distilled_rows(bootstrap, ts2)

    changed_first = compute_changed_rows(first, previous_df=None)
    assert changed_first.height == len(bootstrap.elements)  # baseline established

    changed_second = compute_changed_rows(second, previous_df=changed_first)
    assert changed_second.height == 0  # nothing changed


def test_changed_field_produces_exactly_one_row(bootstrap_payload):
    bootstrap = parse_bootstrap_static(bootstrap_payload, logging.getLogger("test"))
    ts1 = datetime(2026, 8, 21, 10, 0, tzinfo=timezone.utc)
    ts2 = datetime(2026, 8, 21, 11, 0, tzinfo=timezone.utc)

    baseline = compute_changed_rows(build_distilled_rows(bootstrap, ts1), previous_df=None)

    bootstrap_payload["elements"][0]["selected_by_percent"] = "50.9"
    changed_bootstrap = parse_bootstrap_static(bootstrap_payload, logging.getLogger("test"))
    second = build_distilled_rows(changed_bootstrap, ts2)

    changed = compute_changed_rows(second, previous_df=baseline)
    assert changed.height == 1
    assert changed["element_id"][0] == 101


def test_distilled_row_count_far_below_players_times_snapshots(tmp_path: Path, bootstrap_payload):
    bootstrap = parse_bootstrap_static(bootstrap_payload, logging.getLogger("test"))
    n_players = len(bootstrap.elements)
    n_snapshots = 20
    distilled_dir = tmp_path / "distilled"
    gw_dir = distilled_dir / "gw2"

    base_ts = datetime(2026, 8, 21, 8, 0, tzinfo=timezone.utc)
    for i in range(n_snapshots):
        ts = base_ts + timedelta(minutes=i)
        new_df = build_distilled_rows(bootstrap, ts)
        previous_df = latest_distilled_state(gw_dir)
        changed_df = compute_changed_rows(new_df, previous_df)
        write_distilled_shard(distilled_dir, 2, changed_df, ts)
        # Only the first snapshot introduces any change; the rest are identical.

    total_rows = 0
    for shard in gw_dir.glob("*.parquet"):
        con = duckdb.connect()
        total_rows += con.execute(f"SELECT count(*) FROM read_parquet('{shard.as_posix()}')").fetchone()[0]
        con.close()

    assert total_rows == n_players  # only the baseline snapshot wrote rows
    assert total_rows < n_players * n_snapshots


def test_duckdb_point_in_time_query_returns_historical_value(tmp_path: Path, bootstrap_payload):
    distilled_dir = tmp_path / "distilled"
    gw_dir = distilled_dir / "gw2"

    ts1 = datetime(2026, 8, 21, 8, 0, tzinfo=timezone.utc)
    bootstrap1 = parse_bootstrap_static(bootstrap_payload, logging.getLogger("test"))
    shard1 = write_distilled_shard(distilled_dir, 2, build_distilled_rows(bootstrap1, ts1), ts1)

    bootstrap_payload["elements"][0]["selected_by_percent"] = "60.0"
    ts2 = datetime(2026, 8, 21, 12, 0, tzinfo=timezone.utc)
    bootstrap2 = parse_bootstrap_static(bootstrap_payload, logging.getLogger("test"))
    changed = compute_changed_rows(build_distilled_rows(bootstrap2, ts2), previous_df=latest_distilled_state(gw_dir))
    write_distilled_shard(distilled_dir, 2, changed, ts2)

    con = duckdb.connect()
    pattern = str(gw_dir / "*.parquet").replace("\\", "/")

    as_of_between = datetime(2026, 8, 21, 10, 0, tzinfo=timezone.utc)
    row = con.execute(
        f"""
        SELECT selected_by_percent FROM read_parquet('{pattern}')
        WHERE element_id = 101 AND snapshot_ts <= ?
        ORDER BY snapshot_ts DESC LIMIT 1
        """,
        [as_of_between],
    ).fetchone()
    assert row[0] == pytest.approx(45.2)

    as_of_after = datetime(2026, 8, 21, 13, 0, tzinfo=timezone.utc)
    row = con.execute(
        f"""
        SELECT selected_by_percent FROM read_parquet('{pattern}')
        WHERE element_id = 101 AND snapshot_ts <= ?
        ORDER BY snapshot_ts DESC LIMIT 1
        """,
        [as_of_after],
    ).fetchone()
    assert row[0] == pytest.approx(60.0)
    con.close()


# --------------------------------------------------------------------------
# end-to-end: malformed payload halts the run and preserves raw (§2.6)
# --------------------------------------------------------------------------


def test_malformed_payload_halts_and_preserves_raw(tmp_path: Path, bootstrap_payload):
    del bootstrap_payload["elements"][0]["now_cost"]  # deliberately malformed

    def handler(request: httpx.Request) -> httpx.Response:
        if "bootstrap-static" in str(request.url):
            return httpx.Response(200, json=bootstrap_payload)
        return httpx.Response(200, json=[])

    storage = StorageConfig(
        raw_dir=str(tmp_path / "raw"),
        distilled_dir=str(tmp_path / "distilled"),
        reference_dir=str(tmp_path / "reference"),
        raw_retention_days=14,
    )

    async def scenario():
        transport = httpx.MockTransport(handler)
        async with FPLClient(
            base_url="https://example.invalid/api",
            user_agent="test-agent",
            rate_limit_per_second=50.0,
            max_retries=1,
            backoff_base=0.01,
            backoff_jitter=0.0,
            transport=transport,
        ) as client:
            with pytest.raises(SchemaValidationError) as excinfo:
                await run_bootstrap_snapshot(client, storage)
        return excinfo.value

    error = asyncio.run(scenario())

    assert isinstance(error.original, ValidationError)
    assert error.raw_path is not None
    assert error.raw_path.exists()
    with gzip.open(error.raw_path, "rt", encoding="utf-8") as fh:
        preserved = json.load(fh)
    assert "now_cost" not in preserved["elements"][0]  # the raw payload, malformed as-is

    # And the malformed run must not have written a distilled shard.
    assert not (tmp_path / "distilled").exists() or not any((tmp_path / "distilled").rglob("*.parquet"))


def test_reference_fixtures_record_match_state(tmp_path: Path, bootstrap_payload, fixtures_payload):
    """The reference tier used to keep only {id, event, team_h, team_a,
    kickoff_time, finished} — six fields, none of which changes once FPL
    publishes the calendar. data/reference/fixtures.parquet was therefore
    byte-identical from its first commit through twenty-odd hourly runs,
    and nothing downstream could tell a played match from an unplayed one.
    """
    import polars as pl

    from collector.schemas import parse_fixtures
    from collector.snapshot import write_reference

    fixtures_payload[0].update(
        {"finished": False, "finished_provisional": True, "started": True, "team_h_score": 3, "team_a_score": 0}
    )
    bootstrap = parse_bootstrap_static(bootstrap_payload, logging.getLogger("test"))
    fixtures = parse_fixtures(fixtures_payload, logging.getLogger("test"))

    write_reference(tmp_path, bootstrap, fixtures)

    row = pl.read_parquet(tmp_path / "fixtures.parquet").row(0, named=True)
    assert row["started"] is True
    assert row["finished_provisional"] is True
    assert row["finished"] is False
    assert (row["team_h_score"], row["team_a_score"]) == (3, 0)


def test_reference_fixtures_carry_fpl_difficulty(tmp_path: Path, bootstrap_payload, fixtures_payload):
    """§5.3.2's fixtures.json shows FPL's static rating beside fdr.py's
    Elo. Both have to come off the same reference row, so persisting it
    here is what makes the comparison possible at all."""
    import polars as pl

    from collector.schemas import parse_fixtures
    from collector.snapshot import write_reference

    bootstrap = parse_bootstrap_static(bootstrap_payload, logging.getLogger("test"))
    fixtures = parse_fixtures(fixtures_payload, logging.getLogger("test"))

    write_reference(tmp_path, bootstrap, fixtures)

    row = pl.read_parquet(tmp_path / "fixtures.parquet").row(0, named=True)
    assert (row["team_h_difficulty"], row["team_a_difficulty"]) == (2, 4)
