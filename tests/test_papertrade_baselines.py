"""papertrade/baselines.py: the live actuals decorated into the panel shape
backtest/baselines.py expects, and the leakage check that guards it."""

from __future__ import annotations

from datetime import datetime, timezone

import polars as pl
import pytest

from backtest.leakage import LeakageError
from papertrade.actuals import _SCHEMA
from papertrade.baselines import LiveReference, baseline_predictions, live_panel
from squad.live import FLOAT_STAT_COLUMNS, INT_STAT_COLUMNS


def _utc(day: int, hour: int = 15) -> datetime:
    return datetime(2026, 9, day, hour, tzinfo=timezone.utc)


def _row(gw: int, eid: int, team: str, total_points: int) -> dict:
    return {
        "gw": gw, "element_id": eid, "position": "MID", "team": team, "is_promoted_club": False,
        **{col: 0 for col in INT_STAT_COLUMNS},
        **{col: 0.0 for col in FLOAT_STAT_COLUMNS},
        **{"minutes": 90, "total_points": total_points},
    }


def _reference(fixtures: list[dict], deadlines: dict[int, datetime]) -> LiveReference:
    return LiveReference(
        fixtures=pl.DataFrame(
            fixtures,
            schema={"id": pl.Int64, "event": pl.Int64, "team_h": pl.Int64, "team_a": pl.Int64,
                    "kickoff_time": pl.Datetime("us", "UTC"), "team_h_difficulty": pl.Int64, "team_a_difficulty": pl.Int64},
        ),
        teams=pl.DataFrame({"id": [1, 2, 3], "name": ["Alpha", "Beta", "Gamma"]}),
        events=pl.DataFrame(
            {"id": list(deadlines), "deadline_time": list(deadlines.values())},
            schema={"id": pl.Int64, "deadline_time": pl.Datetime("us", "UTC")},
        ),
    )


def _fixture(fid: int, gw: int, home: int, away: int, kickoff: datetime, h_diff: int, a_diff: int) -> dict:
    return {"id": fid, "event": gw, "team_h": home, "team_a": away, "kickoff_time": kickoff,
            "team_h_difficulty": h_diff, "team_a_difficulty": a_diff}


def test_live_panel_aggregates_a_double_gameweek_and_nulls_a_blank():
    reference = _reference(
        [
            # gw2: Alpha plays twice (difficulty 2 at home, 4 away) -- a double
            _fixture(1, 2, 1, 2, _utc(5), 2, 3),
            _fixture(2, 2, 3, 1, _utc(3), 3, 4),
        ],
        {2: _utc(1)},
    )
    # Gamma has a fixture; a fourth club with none would be blank -- use a
    # name that isn't in the fixture list at all.
    actuals = pl.DataFrame([_row(2, 10, "Alpha", 5), _row(2, 11, "Unscheduled", 1)], schema=_SCHEMA)

    panel = live_panel(actuals, reference).sort("element_id")

    alpha = panel.row(0, named=True)
    assert alpha["opponent_difficulty"] == pytest.approx(3.0)  # mean(2, 4)
    assert alpha["kickoff_time"] == _utc(3)  # earliest of the two
    blank = panel.row(1, named=True)
    assert blank["opponent_difficulty"] is None
    assert blank["kickoff_time"] is None


def test_baseline_predictions_returns_none_without_an_earlier_gameweek():
    reference = _reference([_fixture(1, 1, 1, 2, _utc(2), 3, 3)], {1: _utc(1)})
    actuals = pl.DataFrame([_row(1, 10, "Alpha", 5)], schema=_SCHEMA)
    assert baseline_predictions(live_panel(actuals, reference), 1, reference) is None


def test_baseline_predictions_run_every_backtest_baseline():
    reference = _reference(
        [_fixture(1, 1, 1, 2, _utc(2), 3, 3), _fixture(2, 2, 1, 2, _utc(9), 2, 4)],
        {1: _utc(1), 2: _utc(8)},
    )
    actuals = pl.DataFrame([_row(1, 10, "Alpha", 6), _row(2, 10, "Alpha", 2)], schema=_SCHEMA)

    predictions = baseline_predictions(live_panel(actuals, reference), 2, reference)

    assert set(predictions) == {"trailing_mean", "fpl_form_approx", "fixture_adjusted_trailing_mean"}
    assert predictions["trailing_mean"]["prediction"].to_list() == [6.0]
    # Alpha's gw2 difficulty is 2 (home), one step easier than 3 -> scaled up
    assert predictions["fixture_adjusted_trailing_mean"]["prediction"][0] > 6.0


def test_a_training_match_kicking_off_after_the_deadline_raises():
    """The case numbering can't see: a "gw1" fixture played after gw2's
    deadline. A violation raises; it does not warn."""
    reference = _reference(
        [_fixture(1, 1, 1, 2, _utc(10), 3, 3), _fixture(2, 2, 1, 2, _utc(12), 3, 3)],
        {1: _utc(1), 2: _utc(8)},
    )
    actuals = pl.DataFrame([_row(1, 10, "Alpha", 6), _row(2, 10, "Alpha", 2)], schema=_SCHEMA)

    with pytest.raises(LeakageError, match="gw2"):
        baseline_predictions(live_panel(actuals, reference), 2, reference)
