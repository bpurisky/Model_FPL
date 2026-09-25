"""analytics.scoreline: team strengths in xG, fixtures as Poisson."""

from __future__ import annotations

import math

import polars as pl
import pytest

from analytics.scoreline import (
    HOME_FACTOR,
    _expected_floor_half,
    scoreline_table,
    team_match_rates,
    team_priors,
    team_strengths,
)


def _strengths(**teams):
    return pl.DataFrame([{"team": t, "attack": a, "defence": d} for t, (a, d) in teams.items()])


def test_expected_floor_half_matches_a_direct_sum():
    lam = 1.7
    direct = sum((k // 2) * math.exp(-lam) * lam**k / math.factorial(k) for k in range(40))
    assert _expected_floor_half(lam) == pytest.approx(direct)
    assert _expected_floor_half(0.0) == 0.0


def test_a_fixture_is_attack_times_defence_over_the_league_with_home_advantage():
    strengths = _strengths(A=(2.0, 1.0), B=(1.0, 1.0))  # league attack 1.5
    fixtures = pl.DataFrame({"gw": [1], "team_h": ["A"], "team_a": ["B"]})
    table = {r["team"]: r for r in scoreline_table(fixtures, strengths, 1).to_dicts()}
    lam_a = 2.0 * 1.0 / 1.5 * HOME_FACTOR
    lam_b = 1.0 * 1.0 / 1.5 / HOME_FACTOR
    assert table["A"]["attack_mult"] == pytest.approx(lam_a / 2.0)
    assert table["B"]["expected_cs"] == pytest.approx(math.exp(-lam_a))
    assert table["A"]["expected_cs"] == pytest.approx(math.exp(-lam_b))
    assert table["A"]["n_fixtures"] == 1


def test_double_gameweeks_sum_and_blank_teams_are_absent():
    strengths = _strengths(A=(1.5, 1.5), B=(1.5, 1.5), C=(1.5, 1.5), D=(1.5, 1.5))
    fixtures = pl.DataFrame({"gw": [5, 5], "team_h": ["A", "B"], "team_a": ["C", "A"]})
    table = {r["team"]: r for r in scoreline_table(fixtures, strengths, 5).to_dicts()}
    assert table["A"]["n_fixtures"] == 2
    assert table["A"]["attack_mult"] == pytest.approx(HOME_FACTOR + 1 / HOME_FACTOR)
    assert "D" not in table


def test_matches_come_from_the_fixture_list_or_the_archive_column():
    rows = pl.DataFrame({
        "team": ["A", "A", "B"], "gw": [1, 1, 1], "n_fixtures": [1, 1, 1],
        "expected_goals": [0.5, 0.7, 0.3], "expected_goals_conceded": [0.3, 0.2, 1.2],
    })
    fixtures = pl.DataFrame({"gw": [1], "team_h": ["A"], "team_a": ["B"]})
    for rates in (team_match_rates(rows, fixtures), team_match_rates(rows)):
        a = rates.filter(pl.col("team") == "A").row(0, named=True)
        assert a["matches"] == 1
        assert a["xg_for"] == pytest.approx(1.2)
        assert a["xg_against"] == pytest.approx(0.3)


def test_a_promoted_club_starts_from_the_sides_it_replaced():
    prev = pl.DataFrame({
        "team": ["Stayed", "Relegated"], "gw": [1, 1], "matches": [1, 1],
        "xg_for": [2.0, 0.8], "xg_against": [1.0, 2.2],
    })
    priors = {r["team"]: r for r in team_priors(prev, ["Stayed", "Promoted"]).to_dicts()}
    assert priors["Stayed"]["prior_for"] == pytest.approx(2.0)
    assert priors["Promoted"]["prior_for"] == pytest.approx(0.8)
    assert priors["Promoted"]["prior_against"] == pytest.approx(2.2)


def test_strengths_shrink_this_season_toward_the_prior():
    priors = pl.DataFrame({"team": ["A"], "prior_for": [1.0], "prior_against": [1.0]})
    rates = pl.DataFrame({"team": ["A"], "gw": [1], "matches": [2], "xg_for": [6.0], "xg_against": [2.0]})
    s = team_strengths(rates, priors, before_gw=2, prior_matches=2).row(0, named=True)
    assert s["attack"] == pytest.approx((6.0 + 2.0) / 4)
    assert s["defence"] == pytest.approx((2.0 + 2.0) / 4)
    # gameweeks at or after before_gw are not yet played
    assert team_strengths(rates, priors, before_gw=1, prior_matches=2).row(0, named=True)["attack"] == pytest.approx(1.0)
