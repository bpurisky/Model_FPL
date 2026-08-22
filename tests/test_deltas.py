"""§1.2/§7 deltas: point-in-time state and windowed deltas over the
distilled time series."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import polars as pl
import pytest

from analytics.deltas import _STATE_SCHEMA, compute_deltas, reference_timestamps, state_as_of

BASE = datetime(2026, 8, 21, 12, 0, tzinfo=timezone.utc)


def _row(ts: datetime, eid: int, now_cost: int, selected_by_percent: float, form: float = 5.0) -> dict:
    return {
        "snapshot_ts": ts, "element_id": eid, "now_cost": now_cost, "selected_by_percent": selected_by_percent,
        "transfers_in_event": 0, "transfers_out_event": 0, "form": form, "status": "a",
        "chance_of_playing_next_round": None, "news_added": None, "ep_next": 4.0,
    }


def _write_shard(gw_dir, ts: datetime, rows: list[dict]) -> None:
    gw_dir.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(rows, schema=_STATE_SCHEMA).write_parquet(gw_dir / f"{ts.strftime('%Y%m%dT%H%M%SZ')}.parquet")


def test_state_as_of_returns_empty_typed_frame_when_no_data(tmp_path):
    result = state_as_of(tmp_path, BASE)
    assert result.height == 0
    assert set(result.columns) == set(_STATE_SCHEMA)


def test_state_as_of_picks_the_latest_snapshot_at_or_before_cutoff(tmp_path):
    gw1 = tmp_path / "gw1"
    _write_shard(gw1, BASE - timedelta(hours=2), [_row(BASE - timedelta(hours=2), 101, now_cost=50, selected_by_percent=10.0)])
    _write_shard(gw1, BASE - timedelta(hours=1), [_row(BASE - timedelta(hours=1), 101, now_cost=51, selected_by_percent=12.0)])
    _write_shard(gw1, BASE, [_row(BASE, 101, now_cost=52, selected_by_percent=15.0)])

    at_now = state_as_of(tmp_path, BASE)
    assert at_now.row(0, named=True)["now_cost"] == 52

    at_90_min_ago = state_as_of(tmp_path, BASE - timedelta(minutes=90))
    assert at_90_min_ago.row(0, named=True)["now_cost"] == 50  # only the -2h snapshot qualifies

    before_any_data = state_as_of(tmp_path, BASE - timedelta(hours=10))
    assert before_any_data.height == 0


def test_state_as_of_spans_multiple_gameweek_directories(tmp_path):
    """A window can reach back across a gw boundary -- gw directories are a
    storage detail, not a real partition (module docstring)."""
    _write_shard(tmp_path / "gw1", BASE - timedelta(hours=100), [_row(BASE - timedelta(hours=100), 101, now_cost=50, selected_by_percent=5.0)])
    _write_shard(tmp_path / "gw2", BASE, [_row(BASE, 101, now_cost=55, selected_by_percent=20.0)])

    old = state_as_of(tmp_path, BASE - timedelta(hours=90))
    assert old.row(0, named=True)["now_cost"] == 50  # found in gw1's shard despite querying "now" near gw2


def test_reference_timestamps_includes_since_gw_only_when_given():
    refs = reference_timestamps(BASE)
    assert set(refs) == {"1h", "24h", "72h"}

    deadline = BASE - timedelta(days=3)
    refs_with_gw = reference_timestamps(BASE, since_gw_deadline=deadline)
    assert refs_with_gw["since_gw"] == deadline


def test_compute_deltas_computes_numeric_differences(tmp_path):
    gw1 = tmp_path / "gw1"
    _write_shard(gw1, BASE - timedelta(hours=24), [_row(BASE - timedelta(hours=24), 101, now_cost=50, selected_by_percent=10.0)])
    _write_shard(gw1, BASE, [_row(BASE, 101, now_cost=53, selected_by_percent=25.0)])

    result = compute_deltas(tmp_path, BASE, reference_timestamps(BASE))

    row = result.row(0, named=True)
    assert row["now_cost"] == 53
    assert row["now_cost_delta_24h"] == pytest.approx(3.0)
    assert row["selected_by_percent_delta_24h"] == pytest.approx(15.0)
    # the only snapshot at-or-before "1h ago" is the 24h-old one, so the 1h
    # delta correctly reuses it rather than reporting a false zero-change
    assert row["now_cost_delta_1h"] == pytest.approx(3.0)


def test_compute_deltas_all_null_for_a_window_with_no_baseline_data(tmp_path):
    gw1 = tmp_path / "gw1"
    _write_shard(gw1, BASE, [_row(BASE, 101, now_cost=50, selected_by_percent=10.0)])

    result = compute_deltas(tmp_path, BASE, {"72h": BASE - timedelta(hours=72)})

    row = result.row(0, named=True)
    assert row["now_cost_delta_72h"] is None


def test_compute_deltas_empty_when_no_current_state(tmp_path):
    result = compute_deltas(tmp_path, BASE, reference_timestamps(BASE))
    assert result.height == 0
