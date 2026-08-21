"""§4.1: compute_points is a pure function, exhaustively tested, entirely
config-driven — every rule change here is a fixture pointing at a different
YAML file, never a different code path."""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from analytics.scoring import (
    EventVector,
    compute_bps,
    compute_points,
    event_vector_from_row,
    load_scoring_config,
    validate_against_actual,
)

CONFIG_2024_25 = load_scoring_config(Path("config/scoring_2024_25.yaml"))
CONFIG_2025_26 = load_scoring_config(Path("config/scoring_2025_26.yaml"))
CONFIG_2026_27 = load_scoring_config(Path("config/scoring_2026_27.yaml"))


def test_minutes_bands():
    assert compute_points(EventVector(position="MID", minutes=0), CONFIG_2024_25) == 0
    assert compute_points(EventVector(position="MID", minutes=45), CONFIG_2024_25) == 1
    assert compute_points(EventVector(position="MID", minutes=90), CONFIG_2024_25) == 2


def test_goals_by_position_matches_real_data():
    # Cross-checked directly against 2024-25 merged_gw.csv (Ollie Watkins et al.).
    assert compute_points(EventVector(position="FWD", minutes=90, goals_scored=1), CONFIG_2024_25) == 6  # 2 + 4
    assert compute_points(EventVector(position="MID", minutes=90, goals_scored=1), CONFIG_2024_25) == 7  # 2 + 5
    assert compute_points(EventVector(position="DEF", minutes=90, goals_scored=1), CONFIG_2024_25) == 8  # 2 + 6
    assert compute_points(EventVector(position="GK", minutes=90, goals_scored=1), CONFIG_2024_25) == 8  # 2 + 6


def test_clean_sheets_by_position_matches_real_data():
    # Cross-checked against Virgil van Dijk / Bruno Fernandes / Thomas Partey (2024-25).
    assert compute_points(EventVector(position="DEF", minutes=90, clean_sheets=1), CONFIG_2024_25) == 6  # 2 + 4
    assert compute_points(EventVector(position="MID", minutes=90, clean_sheets=1), CONFIG_2024_25) == 3  # 2 + 1
    assert compute_points(EventVector(position="FWD", minutes=90, clean_sheets=1), CONFIG_2024_25) == 2  # 2 + 0
    assert compute_points(EventVector(position="GK", minutes=90, clean_sheets=1), CONFIG_2024_25) == 6  # 2 + 4


def test_clean_sheet_requires_60_plus_minutes():
    subbed_off_early = EventVector(position="DEF", minutes=55, clean_sheets=1)
    assert compute_points(subbed_off_early, CONFIG_2024_25) == 1  # minutes only, no CS credit


def test_goals_conceded_penalty_requires_60_plus_minutes_and_only_gk_def():
    full_game = EventVector(position="DEF", minutes=90, goals_conceded=4)
    assert compute_points(full_game, CONFIG_2024_25) == 2 - 2  # 2 (min) - 2 (4 conceded / 2 * -1)

    short_appearance = EventVector(position="DEF", minutes=30, goals_conceded=4)
    assert compute_points(short_appearance, CONFIG_2024_25) == 1  # no penalty below 60 min

    midfielder = EventVector(position="MID", minutes=90, goals_conceded=4)
    assert compute_points(midfielder, CONFIG_2024_25) == 2  # MID never penalized for conceding


def test_saves_per_n_mode_matches_real_data():
    # Cross-checked against Caoimhin Kelleher et al. (2024-25): 3 saves = 1 point.
    assert compute_points(EventVector(position="GK", minutes=90, clean_sheets=1, saves=3), CONFIG_2024_25) == 7
    assert compute_points(EventVector(position="GK", minutes=90, saves=2), CONFIG_2024_25) == 2  # below the n=3 boundary
    assert compute_points(EventVector(position="GK", minutes=90, saves=6), CONFIG_2024_25) == 4  # 2 + 2


def test_saves_flat_plus_bonus_mode_2026_27():
    event = EventVector(position="GK", minutes=90, saves=3, close_range_saves=1, big_chance_saves=1)
    # 2 (minutes) + 3*1 (flat) + 1*1 (close-range) + 1*1 (big-chance)
    assert compute_points(event, CONFIG_2026_27) == 2 + 3 + 1 + 1


def test_own_goals_penalties_and_cards():
    assert compute_points(EventVector(position="DEF", minutes=90, own_goals=1), CONFIG_2024_25) == 0  # 2 - 2
    assert compute_points(EventVector(position="FWD", minutes=90, penalties_missed=1), CONFIG_2024_25) == 0  # 2 - 2
    assert compute_points(EventVector(position="GK", minutes=90, penalties_saved=1), CONFIG_2024_25) == 7  # 2 + 5
    assert compute_points(EventVector(position="MID", minutes=90, yellow_cards=1), CONFIG_2024_25) == 1  # 2 - 1
    assert compute_points(EventVector(position="MID", minutes=90, red_cards=1), CONFIG_2024_25) == -1  # 2 - 3


def test_bonus_is_a_direct_pass_through():
    assert compute_points(EventVector(position="MID", minutes=90, bonus=3), CONFIG_2024_25) == 5  # 2 + 3


def test_defensive_contribution_threshold_matches_real_data():
    # Cross-checked against 2025-26 merged_gw.csv directly (§4.1 recon).
    def_at_threshold = EventVector(position="DEF", minutes=90, defensive_contribution=10)
    def_below_threshold = EventVector(position="DEF", minutes=90, defensive_contribution=9)
    mid_at_threshold = EventVector(position="MID", minutes=90, defensive_contribution=12)
    mid_below_threshold = EventVector(position="MID", minutes=90, defensive_contribution=11)

    assert compute_points(def_at_threshold, CONFIG_2025_26) == 4  # 2 + 2
    assert compute_points(def_below_threshold, CONFIG_2025_26) == 2
    assert compute_points(mid_at_threshold, CONFIG_2025_26) == 4
    assert compute_points(mid_below_threshold, CONFIG_2025_26) == 2


def test_defensive_contribution_never_applies_to_goalkeepers():
    gk_huge_dc = EventVector(position="GK", minutes=90, defensive_contribution=50)
    assert compute_points(gk_huge_dc, CONFIG_2025_26) == 2  # minutes only


def test_defensive_contribution_absent_before_the_rule_existed():
    """Same event vector, same defensive_contribution value — only the
    config differs. This is the "no Python edit" property (§4.4) directly:
    the DC rule simply isn't in 2024-25's config, so it can't fire."""
    event = EventVector(position="DEF", minutes=90, defensive_contribution=99)
    assert compute_points(event, CONFIG_2024_25) == 2  # no DC section in this season's config
    assert compute_points(event, CONFIG_2025_26) == 4  # same vector, 2025-26 config -> DC fires


def test_bps_clearances_divisor_changes_with_config_not_code():
    event = EventVector(position="DEF", minutes=90, clearances_blocks_interceptions=6)
    bps_2025_26 = compute_bps(event, CONFIG_2025_26)  # per-2: 6 // 2 * 2 = 6
    bps_2026_27 = compute_bps(event, CONFIG_2026_27)  # per-3: 6 // 3 * 2 = 4
    assert bps_2025_26 > bps_2026_27
    assert bps_2025_26 - bps_2026_27 == 2


def test_bps_tackled_deduction_removed_in_2026_27():
    event = EventVector(position="MID", minutes=90, times_tackled=3)
    bps_2025_26 = compute_bps(event, CONFIG_2025_26)
    bps_2026_27 = compute_bps(event, CONFIG_2026_27)
    assert bps_2026_27 - bps_2025_26 == 3  # the -1*3 penalty is gone


def test_event_vector_from_row_handles_missing_dc_columns():
    row = {
        "position": "DEF", "minutes": 90, "goals_scored": 0, "assists": 0, "clean_sheets": 1,
        "goals_conceded": 0, "own_goals": 0, "penalties_saved": 0, "penalties_missed": 0,
        "yellow_cards": 0, "red_cards": 0, "saves": 0, "bonus": 0,
        "defensive_contribution": None, "clearances_blocks_interceptions": None, "tackles": None,
    }
    event = event_vector_from_row(row)
    assert event.defensive_contribution is None
    assert event.clearances_blocks_interceptions == 0  # None coalesced to 0, not left None
    assert compute_points(event, CONFIG_2024_25) == 6


def test_validate_against_actual_reports_match_rate_and_discrepancies():
    df = pl.DataFrame(
        [
            {
                "season": "2024-25", "gw": 1, "element_id": 1, "name": "Correct Player", "position": "DEF",
                "n_fixtures": 1, "minutes": 90, "goals_scored": 0, "assists": 0, "clean_sheets": 1,
                "goals_conceded": 0, "own_goals": 0, "penalties_saved": 0, "penalties_missed": 0,
                "yellow_cards": 0, "red_cards": 0, "saves": 0, "bonus": 0, "total_points": 6,
                "defensive_contribution": None, "clearances_blocks_interceptions": None, "tackles": None,
            },
            {
                "season": "2024-25", "gw": 1, "element_id": 2, "name": "Wrong Player", "position": "MID",
                "n_fixtures": 1, "minutes": 90, "goals_scored": 1, "assists": 0, "clean_sheets": 0,
                "goals_conceded": 0, "own_goals": 0, "penalties_saved": 0, "penalties_missed": 0,
                "yellow_cards": 0, "red_cards": 0, "saves": 0, "bonus": 0, "total_points": 999,  # deliberately wrong
                "defensive_contribution": None, "clearances_blocks_interceptions": None, "tackles": None,
            },
        ]
    )
    report = validate_against_actual(df, CONFIG_2024_25)
    assert report.n_rows == 2
    assert report.n_exact_match == 1
    assert report.match_rate == pytest.approx(0.5)
    assert report.discrepancies.height == 1
    assert report.discrepancies["name"][0] == "Wrong Player"
