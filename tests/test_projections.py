"""§4.2: the event model's heads combine into a points projection, using
the same config-driven rules the ground truth is scored with (§4.1)."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import polars as pl
import pytest

from analytics.projections import (
    _adverse_scale,
    _favorable_scale,
    expected_points_from_projection,
    project_event_vectors,
    project_points,
)
from analytics.scoring import load_scoring_config

CONFIG_2025_26 = load_scoring_config(Path("config/scoring_2025_26.yaml"))
BASE_DAY = datetime(2025, 1, 1, tzinfo=timezone.utc)


def _row(element_id, gw, minutes=90, goals=0, assists=0, cs=0, gc=0, saves=0, bonus=0, dc=None, position="MID", promoted=False):
    return {
        "element_id": element_id, "gw": gw, "minutes": minutes, "goals_scored": goals, "assists": assists,
        "clean_sheets": cs, "goals_conceded": gc, "saves": saves, "bonus": bonus, "yellow_cards": 0, "red_cards": 0,
        "own_goals": 0, "penalties_missed": 0, "penalties_saved": 0, "defensive_contribution": dc,
        "position": position, "team": "Team A", "is_promoted_club": promoted, "kickoff_time": BASE_DAY + timedelta(days=gw),
    }


def _roster_row(element_id, position="MID", promoted=False, difficulty=3.0):
    return {"element_id": element_id, "position": position, "team": "Team A", "is_promoted_club": promoted, "custom_difficulty": difficulty}


def test_favorable_and_adverse_scales_are_inverse_around_neutral():
    assert _favorable_scale(3.0) == pytest.approx(1.0)
    assert _adverse_scale(3.0) == pytest.approx(1.0)
    assert _favorable_scale(1.0) > 1.0  # easy fixture helps the attacker
    assert _favorable_scale(5.0) < 1.0
    assert _adverse_scale(5.0) > 1.0  # hard fixture hurts the defense more
    assert _adverse_scale(1.0) < 1.0


def test_project_event_vectors_produces_expected_columns():
    train_df = pl.DataFrame([_row(1, gw, goals=1) for gw in range(1, 4)])
    roster = pl.DataFrame([_roster_row(1)])
    result = project_event_vectors(train_df, roster, target_gw=4, config=CONFIG_2025_26)
    for col in ["p_blank", "p_short", "p_full", "goals_scored_trailing", "assists_trailing", "clean_sheets_trailing", "p_dc_threshold_trailing"]:
        assert col in result.columns


def test_expected_points_matches_hand_computed_value_for_a_reliable_starter():
    """A FWD who always plays 90, always scores exactly 1, never anything
    else, at neutral difficulty: expectation should be minutes(2) + goal(4)."""
    row = {
        "position": "FWD", "custom_difficulty": 3.0, "p_blank": 0.0, "p_short": 0.0, "p_full": 1.0,
        "goals_scored_trailing": 1.0, "assists_trailing": 0.0, "clean_sheets_trailing": 0.0,
        "goals_conceded_trailing": 0.0, "saves_trailing": 0.0, "own_goals_trailing": 0.0,
        "penalties_missed_trailing": 0.0, "penalties_saved_trailing": 0.0, "yellow_cards_trailing": 0.0,
        "red_cards_trailing": 0.0, "bonus_trailing": 0.0, "p_dc_threshold_trailing": 0.0,
    }
    assert expected_points_from_projection(row, CONFIG_2025_26) == pytest.approx(2.0 + 4.0)


def test_easier_fixture_increases_attacking_projection():
    base_row = {
        "position": "FWD", "p_blank": 0.0, "p_short": 0.0, "p_full": 1.0,
        "goals_scored_trailing": 1.0, "assists_trailing": 0.0, "clean_sheets_trailing": 0.0,
        "goals_conceded_trailing": 0.0, "saves_trailing": 0.0, "own_goals_trailing": 0.0,
        "penalties_missed_trailing": 0.0, "penalties_saved_trailing": 0.0, "yellow_cards_trailing": 0.0,
        "red_cards_trailing": 0.0, "bonus_trailing": 0.0, "p_dc_threshold_trailing": 0.0,
    }
    easy = expected_points_from_projection({**base_row, "custom_difficulty": 1.0}, CONFIG_2025_26)
    hard = expected_points_from_projection({**base_row, "custom_difficulty": 5.0}, CONFIG_2025_26)
    assert easy > hard


def test_harder_fixture_increases_expected_goals_conceded_penalty():
    base_row = {
        "position": "DEF", "p_blank": 0.0, "p_short": 0.0, "p_full": 1.0,
        "goals_scored_trailing": 0.0, "assists_trailing": 0.0, "clean_sheets_trailing": 0.0,
        "goals_conceded_trailing": 2.0, "saves_trailing": 0.0, "own_goals_trailing": 0.0,
        "penalties_missed_trailing": 0.0, "penalties_saved_trailing": 0.0, "yellow_cards_trailing": 0.0,
        "red_cards_trailing": 0.0, "bonus_trailing": 0.0, "p_dc_threshold_trailing": 0.0,
    }
    easy = expected_points_from_projection({**base_row, "custom_difficulty": 1.0}, CONFIG_2025_26)
    hard = expected_points_from_projection({**base_row, "custom_difficulty": 5.0}, CONFIG_2025_26)
    assert hard < easy  # harder fixture -> more conceding -> lower points for a defender


def test_defensive_contribution_threshold_rate_feeds_points():
    row_common = {
        "position": "DEF", "custom_difficulty": 3.0, "p_blank": 0.0, "p_short": 0.0, "p_full": 1.0,
        "goals_scored_trailing": 0.0, "assists_trailing": 0.0, "clean_sheets_trailing": 0.0,
        "goals_conceded_trailing": 0.0, "saves_trailing": 0.0, "own_goals_trailing": 0.0,
        "penalties_missed_trailing": 0.0, "penalties_saved_trailing": 0.0, "yellow_cards_trailing": 0.0,
        "red_cards_trailing": 0.0, "bonus_trailing": 0.0,
    }
    never_hits_threshold = expected_points_from_projection({**row_common, "p_dc_threshold_trailing": 0.0}, CONFIG_2025_26)
    always_hits_threshold = expected_points_from_projection({**row_common, "p_dc_threshold_trailing": 1.0}, CONFIG_2025_26)
    assert always_hits_threshold - never_hits_threshold == pytest.approx(2.0)  # DC bonus in 2025-26 config


def test_project_points_returns_one_prediction_per_roster_player():
    train_df = pl.DataFrame([_row(1, gw, goals=1, position="FWD") for gw in range(1, 4)] + [_row(2, gw, cs=1, position="DEF") for gw in range(1, 4)])
    roster = pl.DataFrame([_roster_row(1, "FWD"), _roster_row(2, "DEF")])
    difficulty_table = pl.DataFrame({"team": ["Team A"], "gw": [4], "custom_difficulty": [3.0]})
    result = project_points(train_df, roster, target_gw=4, config=CONFIG_2025_26, difficulty_table=difficulty_table)
    assert set(result["element_id"].to_list()) == {1, 2}
    assert result["prediction"].null_count() == 0


def test_promoted_club_player_with_no_history_still_gets_a_nonzero_projection():
    train_df = pl.DataFrame([_row(10, gw, goals=1, position="FWD", promoted=True) for gw in range(1, 4)])
    roster = pl.DataFrame([_roster_row(99, "FWD", promoted=True)])  # no history of its own
    difficulty_table = pl.DataFrame({"team": ["Team A"], "gw": [4], "custom_difficulty": [3.0]})
    result = project_points(train_df, roster, target_gw=4, config=CONFIG_2025_26, difficulty_table=difficulty_table)
    assert result["prediction"][0] > 0  # pooled from the promoted-club FWD peer, not zeroed out
