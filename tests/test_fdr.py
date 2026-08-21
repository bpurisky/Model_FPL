"""§4.3: self-contained, point-in-time-correct Elo (see analytics/fdr.py's
module docstring for why this isn't sourced from FPL-Core-Insights), plus
the custom-vs-FPL difficulty comparison."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import polars as pl
import pytest

from analytics.fdr import build_fdr_comparison, compute_elo_ratings, custom_difficulty, team_gameweek_difficulty
from backtest.backfill import NORMALIZED_DIR, RAW_CACHE_DIR, load_match_results

BASE_DAY = datetime(2025, 8, 1, tzinfo=timezone.utc)


def _match(fixture, event, team_h, team_a, h_score, a_score, day_offset):
    return {
        "fixture": fixture, "event": event, "team_h": team_h, "team_a": team_a,
        "team_h_score": h_score, "team_a_score": a_score, "kickoff_time": BASE_DAY + timedelta(days=day_offset),
    }


def test_home_win_increases_home_elo_decreases_away_elo():
    matches = pl.DataFrame([_match(1, 1, 10, 20, 3, 0, 0)])
    result = compute_elo_ratings(matches)
    assert result.final[10] > 1500.0
    assert result.final[20] < 1500.0
    # zero-sum-ish: the two changes are equal and opposite (home advantage
    # only affects *expectation*, not the symmetry of the update itself)
    assert (result.final[10] - 1500.0) == pytest.approx(-(result.final[20] - 1500.0))


def test_draw_between_equal_teams_leaves_elo_unchanged():
    matches = pl.DataFrame([_match(1, 1, 10, 20, 1, 1, 0)])
    result = compute_elo_ratings(matches, home_advantage=0.0)
    assert result.final[10] == pytest.approx(1500.0)
    assert result.final[20] == pytest.approx(1500.0)


def test_pre_match_elo_never_reflects_that_matchs_own_result():
    """Point-in-time by construction (§0.3): fixture 2's recorded pre-match
    Elo must equal team 10's rating *before* fixture 1, updated only by
    fixture 1's result — never influenced by fixture 2 itself."""
    matches = pl.DataFrame(
        [
            _match(1, 1, 10, 20, 3, 0, 0),  # team 10 thrashes team 20
            _match(2, 2, 10, 30, 0, 0, 7),  # team 10's next match, a week later
        ]
    )
    result = compute_elo_ratings(matches)
    fixture1 = result.per_fixture.filter(pl.col("fixture") == 1).row(0, named=True)
    fixture2 = result.per_fixture.filter(pl.col("fixture") == 2).row(0, named=True)

    assert fixture1["elo_home_pre"] == pytest.approx(1500.0)  # nothing happened yet
    assert fixture2["elo_home_pre"] > 1500.0  # reflects fixture 1's win
    assert fixture2["elo_home_pre"] != fixture1["elo_home_pre"]


def test_processes_matches_in_kickoff_order_regardless_of_input_order():
    matches = pl.DataFrame(
        [
            _match(2, 2, 10, 30, 0, 0, 7),  # later match listed first
            _match(1, 1, 10, 20, 3, 0, 0),  # earlier match listed second
        ]
    )
    result = compute_elo_ratings(matches)
    fixture2 = result.per_fixture.filter(pl.col("fixture") == 2).row(0, named=True)
    assert fixture2["elo_home_pre"] > 1500.0  # still reflects fixture 1 despite input order


def test_custom_difficulty_bounded_and_monotonic_in_opponent_strength():
    weak_opponent = custom_difficulty(elo_team=1500, elo_opponent=1000, is_home=True)
    neutral_opponent = custom_difficulty(elo_team=1500, elo_opponent=1500 + 60, is_home=True)  # cancels home advantage
    strong_opponent = custom_difficulty(elo_team=1500, elo_opponent=2000, is_home=True)

    assert weak_opponent < neutral_opponent < strong_opponent
    assert neutral_opponent == pytest.approx(3.0)
    assert 1.0 <= weak_opponent <= 5.0
    assert 1.0 <= strong_opponent <= 5.0


def test_custom_difficulty_clamps_extreme_gaps():
    assert custom_difficulty(elo_team=1000, elo_opponent=3000, is_home=False) == 5.0
    assert custom_difficulty(elo_team=3000, elo_opponent=1000, is_home=False) == 1.0


def test_home_advantage_makes_the_same_opponent_easier_at_home():
    away_difficulty = custom_difficulty(elo_team=1500, elo_opponent=1700, is_home=False)
    home_difficulty = custom_difficulty(elo_team=1500, elo_opponent=1700, is_home=True)
    assert home_difficulty < away_difficulty


def test_build_fdr_comparison_joins_on_fixture():
    elo_result = compute_elo_ratings(pl.DataFrame([_match(1, 1, 10, 20, 2, 1, 0)]))
    fpl_difficulty = pl.DataFrame([{"fixture": 1, "team_h_difficulty": 3, "team_a_difficulty": 3}])
    comparison = build_fdr_comparison(elo_result, fpl_difficulty)
    row = comparison.row(0, named=True)
    assert row["team_h_difficulty"] == 3
    assert "custom_difficulty_home" in comparison.columns
    assert "custom_difficulty_away" in comparison.columns


def test_team_gameweek_difficulty_aggregates_double_gameweeks_like_backfill_does():
    teams = pl.DataFrame([{"id": 10, "name": "Team A"}, {"id": 20, "name": "Team B"}, {"id": 30, "name": "Team C"}])
    # Team A plays twice in gw2 — a double gameweek, same shape backfill.py
    # aggregates with a mean for FPL's own opponent_difficulty.
    matches = pl.DataFrame(
        [
            _match(1, 1, 10, 20, 1, 1, 0),
            _match(2, 2, 10, 30, 2, 0, 7),
            _match(3, 2, 20, 10, 0, 3, 8),
        ]
    )
    result = team_gameweek_difficulty(matches, teams)
    row = result.filter((pl.col("team") == "Team A") & (pl.col("gw") == 2))
    assert row.height == 1  # one row per (team, gw), even across two fixtures


@pytest.mark.skipif(
    not (RAW_CACHE_DIR / "2025-26" / "fixtures.csv").exists(), reason="requires the cached 2025-26 fixtures.csv"
)
def test_real_2025_26_elo_agrees_with_the_actual_league_table_at_the_extremes():
    """Sanity check against real results, not just synthetic data: derive
    the actual final standings from the same match results (not asserted
    from memory — this is a season that finished after this model's own
    knowledge cutoff) and confirm Elo ranks the real top team above the
    real bottom team.
    """
    matches = load_match_results(RAW_CACHE_DIR / "2025-26" / "fixtures.csv")
    result = compute_elo_ratings(matches)

    points = pl.concat(
        [
            matches.select(
                team=pl.col("team_h"),
                points=pl.when(pl.col("team_h_score") > pl.col("team_a_score"))
                .then(3)
                .when(pl.col("team_h_score") == pl.col("team_a_score"))
                .then(1)
                .otherwise(0),
            ),
            matches.select(
                team=pl.col("team_a"),
                points=pl.when(pl.col("team_a_score") > pl.col("team_h_score"))
                .then(3)
                .when(pl.col("team_a_score") == pl.col("team_h_score"))
                .then(1)
                .otherwise(0),
            ),
        ]
    ).group_by("team").agg(pl.col("points").sum()).sort("points")

    bottom_team = points.row(0, named=True)["team"]
    top_team = points.row(-1, named=True)["team"]
    assert result.final[top_team] > result.final[bottom_team]
