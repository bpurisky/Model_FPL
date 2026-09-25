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


def _row(element_id, gw, minutes=90, goals=0, assists=0, cs=0, gc=0, saves=0, bonus=0, dc=None, position="MID", promoted=False,
         xg=None, xa=None, team="Team A"):
    """xG/xA default to the realized goals/assists, so a test that only
    cares about "this player scores" needn't say it twice."""
    return {
        "element_id": element_id, "gw": gw, "minutes": minutes, "goals_scored": goals, "assists": assists,
        "expected_goals": float(goals if xg is None else xg), "expected_assists": float(assists if xa is None else xa),
        "clean_sheets": cs, "goals_conceded": gc, "saves": saves, "bonus": bonus, "yellow_cards": 0, "red_cards": 0,
        "own_goals": 0, "penalties_missed": 0, "penalties_saved": 0, "defensive_contribution": dc,
        "position": position, "team": team, "is_promoted_club": promoted, "kickoff_time": BASE_DAY + timedelta(days=gw),
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
    for col in ["p_blank", "p_short", "p_full", "expected_goals_trailing", "expected_assists_trailing", "clean_sheets_trailing", "p_dc_threshold_trailing"]:
        assert col in result.columns


def test_expected_points_matches_hand_computed_value_for_a_reliable_starter():
    """A FWD who always plays 90, always makes exactly 1 xG, never anything
    else, at neutral difficulty: expectation should be minutes(2) + goal(4)."""
    row = {
        "position": "FWD", "custom_difficulty": 3.0, "p_blank": 0.0, "p_short": 0.0, "p_full": 1.0,
        "expected_goals_trailing": 1.0, "expected_assists_trailing": 0.0, "clean_sheets_trailing": 0.0,
        "goals_conceded_trailing": 0.0, "saves_trailing": 0.0, "own_goals_trailing": 0.0,
        "penalties_missed_trailing": 0.0, "penalties_saved_trailing": 0.0, "yellow_cards_trailing": 0.0,
        "red_cards_trailing": 0.0, "bonus_trailing": 0.0, "p_dc_threshold_trailing": 0.0,
    }
    assert expected_points_from_projection(row, CONFIG_2025_26) == pytest.approx(2.0 + 4.0)


def test_easier_fixture_increases_attacking_projection():
    base_row = {
        "position": "FWD", "p_blank": 0.0, "p_short": 0.0, "p_full": 1.0,
        "expected_goals_trailing": 1.0, "expected_assists_trailing": 0.0, "clean_sheets_trailing": 0.0,
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
        "expected_goals_trailing": 0.0, "expected_assists_trailing": 0.0, "clean_sheets_trailing": 0.0,
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
        "expected_goals_trailing": 0.0, "expected_assists_trailing": 0.0, "clean_sheets_trailing": 0.0,
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


def test_goals_head_reads_xg_not_realized_goals():
    """Two forwards with the same chances: one finished them, one didn't.
    The goals head projects them alike, because finishing over a handful of
    games is mostly variance."""
    train_df = pl.DataFrame(
        [_row(1, gw, goals=1, xg=0.5, position="FWD") for gw in range(1, 5)]
        + [_row(2, gw, goals=0, xg=0.5, position="FWD") for gw in range(1, 5)]
    )
    roster = pl.DataFrame([_roster_row(1, "FWD"), _roster_row(2, "FWD")])
    result = project_event_vectors(train_df, roster, target_gw=5, config=CONFIG_2025_26)
    xg = dict(zip(result["element_id"], result["expected_goals_trailing"]))
    assert xg[1] == pytest.approx(0.5) and xg[2] == pytest.approx(0.5)


def test_scoring_rates_are_per_appearance_scaled_by_chance_of_playing():
    """A player who missed two of his last three games keeps his scoring
    rate per appearance; only the minutes head says he may not play."""
    train_df = pl.DataFrame(
        [_row(1, gw, xg=0.6, position="FWD") for gw in range(1, 4)]
        + [_row(1, 4, minutes=0, xg=0.0, position="FWD"), _row(1, 5, minutes=0, xg=0.0, position="FWD")]
    )
    roster = pl.DataFrame([_roster_row(1, "FWD")])
    result = project_event_vectors(train_df, roster, target_gw=6, config=CONFIG_2025_26).row(0, named=True)
    assert result["p_blank"] == pytest.approx(2 / 3)
    assert result["expected_goals_trailing"] == pytest.approx(0.6 * (1 / 3))


def test_prior_history_extends_scoring_rates_but_not_minutes():
    """Last season's rows (gw <= 0) lengthen the xG window; this season's
    minutes alone decide P(plays)."""
    this_season = pl.DataFrame([_row(1, 1, xg=0.0, position="FWD")])
    last_season = pl.DataFrame([_row(1, gw, minutes=0 if gw == 0 else 90, xg=1.0, position="FWD") for gw in range(-3, 1)])
    roster = pl.DataFrame([_roster_row(1, "FWD")])
    without = project_event_vectors(this_season, roster, target_gw=2, config=CONFIG_2025_26).row(0, named=True)
    with_prior = project_event_vectors(this_season, roster, target_gw=2, config=CONFIG_2025_26, prior_history=last_season).row(0, named=True)
    assert without["expected_goals_trailing"] == pytest.approx(0.0)
    # three carried appearances at 1.0 xG plus this season's one at 0.0
    assert with_prior["expected_goals_trailing"] == pytest.approx(0.75)
    # last season's closing blank does not count against him
    assert with_prior["p_blank"] == pytest.approx(0.0)


def test_availability_flag_scales_chance_of_playing_and_with_it_every_rate():
    """A nailed starter flagged 50% keeps his per-appearance rates but plays
    half as often; flagged 0% he projects nothing but the blank; an
    unflagged team-mate is untouched."""
    train_df = pl.DataFrame(
        [_row(1, gw, xg=0.4, position="FWD") for gw in range(1, 4)]
        + [_row(2, gw, xg=0.4, position="FWD") for gw in range(1, 4)]
        + [_row(3, gw, xg=0.4, position="FWD") for gw in range(1, 4)]
    )
    roster = pl.DataFrame([_roster_row(1, "FWD"), _roster_row(2, "FWD"), _roster_row(3, "FWD")])
    flags = pl.DataFrame({"element_id": [1, 2, 3], "chance_of_playing_next_round": [50, 0, None]},
                         schema={"element_id": pl.Int64, "chance_of_playing_next_round": pl.Int64})
    out = {r["element_id"]: r for r in project_event_vectors(train_df, roster, target_gw=4, config=CONFIG_2025_26, availability=flags).to_dicts()}
    assert out[1]["p_full"] == pytest.approx(0.5) and out[1]["p_blank"] == pytest.approx(0.5)
    assert out[1]["expected_goals_trailing"] == pytest.approx(0.2)
    assert out[2]["p_blank"] == pytest.approx(1.0) and out[2]["expected_goals_trailing"] == pytest.approx(0.0)
    assert out[3]["p_full"] == pytest.approx(1.0) and out[3]["expected_goals_trailing"] == pytest.approx(0.4)
