"""§6.3's data dependency: a permanent, append-only record of real per-gw
results. `fetch_gw_actuals` itself is live/async and exercised by actually
running it, not unit-tested here; `gw_is_finished` and `append_actuals`'s
append-only guard are pure and covered directly.
"""

from __future__ import annotations

from datetime import datetime, timezone

import polars as pl
import pytest

from collector.schemas import BootstrapStatic, Event, Team
from papertrade.actuals import _SCHEMA, append_actuals, finished_gws_missing_from, gw_is_finished, load_actuals


def _bootstrap(finished: bool, gw: int = 1) -> BootstrapStatic:
    return BootstrapStatic(
        events=[Event(id=gw, name=f"Gameweek {gw}", deadline_time=datetime(2026, 8, 21, tzinfo=timezone.utc), finished=finished)],
        teams=[Team(id=1, name="Team A", short_name="TMA")],
        elements=[],
    )


def _multi_event_bootstrap(finished_ids: list[int], unfinished_ids: list[int]) -> BootstrapStatic:
    events = [
        Event(id=gw, name=f"Gameweek {gw}", deadline_time=datetime(2026, 8, 21, tzinfo=timezone.utc), finished=True)
        for gw in finished_ids
    ] + [
        Event(id=gw, name=f"Gameweek {gw}", deadline_time=datetime(2026, 8, 21, tzinfo=timezone.utc), finished=False)
        for gw in unfinished_ids
    ]
    return BootstrapStatic(events=events, teams=[Team(id=1, name="Team A", short_name="TMA")], elements=[])


def test_gw_is_finished_true_when_event_finished_flag_set():
    assert gw_is_finished(_bootstrap(finished=True), 1) is True


def test_gw_is_finished_false_when_not_finished():
    assert gw_is_finished(_bootstrap(finished=False), 1) is False


def test_gw_is_finished_false_for_unknown_gw():
    assert gw_is_finished(_bootstrap(finished=True), 99) is False


def _row(gw: int, eid: int) -> dict:
    return {
        "gw": gw, "element_id": eid, "position": "MID", "team": "Team A", "is_promoted_club": False,
        "minutes": 90, "goals_scored": 0, "assists": 1, "clean_sheets": 0, "goals_conceded": 1, "saves": 0,
        "bonus": 1, "yellow_cards": 0, "red_cards": 0, "own_goals": 0, "penalties_missed": 0,
        "penalties_saved": 0, "defensive_contribution": 0, "total_points": 5,
    }


def test_append_actuals_creates_and_reads_back(tmp_path):
    path = tmp_path / "actuals.parquet"
    df = pl.DataFrame([_row(1, 101)], schema=_SCHEMA)

    append_actuals(df, path=path)

    loaded = load_actuals(path)
    assert loaded.height == 1
    assert loaded["gw"].to_list() == [1]


def test_append_actuals_refuses_to_duplicate_a_gw(tmp_path):
    path = tmp_path / "actuals.parquet"
    append_actuals(pl.DataFrame([_row(1, 101)], schema=_SCHEMA), path=path)

    with pytest.raises(FileExistsError, match="gw1"):
        append_actuals(pl.DataFrame([_row(1, 102)], schema=_SCHEMA), path=path)

    # original data untouched by the rejected attempt
    assert load_actuals(path)["element_id"].to_list() == [101]


def test_append_actuals_accumulates_across_gameweeks(tmp_path):
    path = tmp_path / "actuals.parquet"
    append_actuals(pl.DataFrame([_row(1, 101)], schema=_SCHEMA), path=path)
    append_actuals(pl.DataFrame([_row(2, 101)], schema=_SCHEMA), path=path)

    loaded = load_actuals(path)
    assert sorted(loaded["gw"].to_list()) == [1, 2]


def test_append_actuals_rejects_empty_frame(tmp_path):
    with pytest.raises(ValueError, match="empty"):
        append_actuals(pl.DataFrame(schema=_SCHEMA), path=tmp_path / "actuals.parquet")


def test_finished_gws_missing_from_excludes_unfinished_and_already_recorded():
    bootstrap = _multi_event_bootstrap(finished_ids=[1, 2, 3], unfinished_ids=[4])
    actuals = pl.DataFrame([_row(1, 101)], schema=_SCHEMA)  # gw1 already recorded

    missing = finished_gws_missing_from(bootstrap, actuals)

    assert missing == [2, 3]  # gw1 recorded, gw4 not finished yet


def test_finished_gws_missing_from_empty_when_nothing_recorded_yet():
    bootstrap = _multi_event_bootstrap(finished_ids=[1], unfinished_ids=[2])
    missing = finished_gws_missing_from(bootstrap, pl.DataFrame(schema=_SCHEMA))
    assert missing == [1]


def test_load_actuals_returns_empty_typed_frame_when_missing(tmp_path):
    loaded = load_actuals(tmp_path / "does-not-exist.parquet")
    assert loaded.height == 0
    assert set(loaded.columns) == set(_SCHEMA)
