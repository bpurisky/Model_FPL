"""§6.3's data dependency: a permanent, append-only record of real per-gw
results. `fetch_gw_actuals` itself is live/async and exercised by actually
running it, not unit-tested here; `gw_is_finished` and `append_actuals`'s
append-only guard are pure and covered directly.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import polars as pl
import pytest

from collector.schemas import BootstrapStatic, Element, Event, Team
from papertrade.actuals import (
    _SCHEMA,
    _build_actuals_frame,
    append_actuals,
    finished_gws_missing_from,
    gw_is_finished,
    load_actuals,
)
from squad.live import FLOAT_STAT_COLUMNS, INT_STAT_COLUMNS


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


def _row(gw: int, eid: int, **stats) -> dict:
    """A complete store row. Built from the column lists rather than
    written out longhand: `pl.DataFrame(rows, schema=_SCHEMA)` fills a key
    a row is missing with null instead of raising, so a hand-written
    literal here silently turns into null columns the moment the schema is
    widened -- which is exactly what happened to the previous version of
    this helper."""
    return {
        "gw": gw, "element_id": eid, "position": "MID", "team": "Team A", "is_promoted_club": False,
        **{col: 0 for col in INT_STAT_COLUMNS},
        **{col: 0.0 for col in FLOAT_STAT_COLUMNS},
        **{"minutes": 90, "assists": 1, "goals_conceded": 1, "bonus": 1, "total_points": 5},
        **stats,
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


def _live_stats(**overrides) -> dict:
    """One element's `stats` block from /event/{gw}/live/. The float
    columns are decimal *strings* because that is what FPL serves --
    verified against the live endpoint 2026-08-22."""
    return {
        **{col: 0 for col in INT_STAT_COLUMNS},
        **{col: "0.0" for col in FLOAT_STAT_COLUMNS},
        **overrides,
    }


def _live_bootstrap() -> BootstrapStatic:
    return BootstrapStatic(
        events=[Event(id=1, name="Gameweek 1", deadline_time=datetime(2026, 8, 21, tzinfo=timezone.utc), finished=True)],
        teams=[Team(id=1, name="Team A", short_name="TMA"), Team(id=7, name="Coventry City", short_name="COV")],
        elements=[
            Element(id=101, web_name="P101", team=1, element_type=3, now_cost=50,
                    selected_by_percent="1.0", transfers_in_event=0, transfers_out_event=0, form="0.0", status="a"),
            Element(id=102, web_name="P102", team=7, element_type=2, now_cost=45,
                    selected_by_percent="1.0", transfers_in_event=0, transfers_out_event=0, form="0.0", status="a"),
        ],
    )


def test_build_actuals_frame_parses_float_stats_out_of_their_strings():
    """FPL serves expected_* and the ICT four as decimal strings. Writing
    them through unparsed would put strings in a Float64 column and break
    squad/live.py's reconstruction subtraction."""
    live_raw = {"elements": [
        {"id": 101, "stats": _live_stats(minutes=90, total_points=6, expected_goals="0.28", expected_assists="0.13")},
        {"id": 102, "stats": _live_stats(minutes=45, total_points=2)},
    ]}

    df = _build_actuals_frame(_live_bootstrap(), live_raw, gw=1)

    row = df.filter(pl.col("element_id") == 101).row(0, named=True)
    assert row["expected_goals"] == pytest.approx(0.28)
    assert row["expected_assists"] == pytest.approx(0.13)
    assert row["minutes"] == 90


def test_build_actuals_frame_writes_no_nulls():
    """`pl.DataFrame(rows, schema=_SCHEMA)` fills a key the rows don't
    carry with null rather than raising, so a column added to the schema
    but not to the row builder would land silently in the append-only
    store as a column of nulls. This is the test that catches that."""
    live_raw = {"elements": [{"id": 101, "stats": _live_stats(minutes=90, total_points=6)}]}

    df = _build_actuals_frame(_live_bootstrap(), live_raw, gw=1)

    nulls = {col: n for col, n in zip(df.columns, df.null_count().row(0)) if n}
    assert nulls == {}


def test_build_actuals_frame_matches_the_store_schema_exactly():
    live_raw = {"elements": [{"id": 101, "stats": _live_stats()}]}

    df = _build_actuals_frame(_live_bootstrap(), live_raw, gw=1)

    assert df.schema == pl.Schema(_SCHEMA)


def test_build_actuals_frame_resolves_position_team_and_promotion():
    live_raw = {"elements": [
        {"id": 101, "stats": _live_stats()},
        {"id": 102, "stats": _live_stats()},
    ]}

    df = _build_actuals_frame(_live_bootstrap(), live_raw, gw=1)

    by_id = {r["element_id"]: r for r in df.to_dicts()}
    assert by_id[101]["position"] == "MID"
    assert by_id[101]["is_promoted_club"] is False
    assert by_id[102]["position"] == "DEF"
    assert by_id[102]["is_promoted_club"] is True  # Coventry, config/promoted_clubs.yaml


def test_build_actuals_frame_skips_elements_absent_from_bootstrap():
    """The live endpoint can carry an element bootstrap-static doesn't
    (mid-week registrations), and there is no position or team to file it
    under."""
    live_raw = {"elements": [
        {"id": 101, "stats": _live_stats()},
        {"id": 999, "stats": _live_stats()},
    ]}

    df = _build_actuals_frame(_live_bootstrap(), live_raw, gw=1)

    assert df["element_id"].to_list() == [101]


def test_store_schema_is_a_subset_of_the_historical_panel():
    """The store and data/historical/{season}.parquet have to line up for
    backtest/baselines.py and backtest/report.py to run against this
    season's results unmodified, and for a cross-season panel to be a
    concat rather than a merge. The expected-goals four were the last
    columns to break this -- they are recorded here from
    /event/{gw}/live/, and backtest/backfill.py had to start reading them
    out of merged_gw.csv before the relation held again.
    """
    panel_path = Path("data/historical/2025-26.parquet")
    if not panel_path.exists():  # pragma: no cover - backfill not run
        pytest.skip("data/historical/2025-26.parquet absent; run `python -m backtest backfill`")

    panel_columns = set(pl.read_parquet(panel_path).collect_schema().names())

    assert set(_SCHEMA) - panel_columns == set()
