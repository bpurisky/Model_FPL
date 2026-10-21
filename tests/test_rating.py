"""squad/rating.py: team rating and chip comparison are plain expected-point
sums over the recommender's own projections, so they can be checked against
hand-built projections."""

from __future__ import annotations

import random

import pytest

from squad.optimize import optimize_squad
from squad.rating import chip_comparison, chips_available, squad_by_gw, team_rating
from tests.test_optimize import _flat_projections, _make_universe

HORIZON = [4, 5, 6]


def _setup(seed: int = 1, bank: int = 0):
    current, pool = _make_universe(seed, bank=bank)
    rng = random.Random(seed)
    projections = {gw: {p.element_id: rng.uniform(1, 8) for p in pool} for gw in HORIZON}
    result = optimize_squad(current, pool, projections, HORIZON, free_transfers=1)
    return current, pool, projections, result


def test_rating_ordering_and_bounds():
    current, pool, projections, result = _setup()
    r = team_rating(current, pool, projections, HORIZON, result, hit_cost=4)
    assert 0 <= r["team_rating"] <= 100
    assert r["current_xi_xp"] <= r["optimum_xp"] + 1e-6
    assert r["recommended_xp"] <= r["optimum_xp"] + 1e-6
    assert r["team_rating"] == pytest.approx(100 * r["current_xi_xp"] / r["optimum_xp"])


def test_rating_is_100_when_current_squad_is_already_best():
    current, pool, _, _ = _setup()
    owned = {sp.element_id for sp in current.players}
    projections = {gw: {p.element_id: (9.0 if p.element_id in owned else 0.5) for p in pool} for gw in HORIZON}
    result = optimize_squad(current, pool, projections, HORIZON, free_transfers=1)
    r = team_rating(current, pool, projections, HORIZON, result, hit_cost=4)
    assert r["team_rating"] == pytest.approx(100.0)


def test_flat_projections_rate_100_and_chips_follow_the_arithmetic():
    current, pool = _make_universe(2)
    projections = _flat_projections(pool, HORIZON, 4.0)
    result = optimize_squad(current, pool, projections, HORIZON, free_transfers=1)
    r = team_rating(current, pool, projections, HORIZON, result, hit_cost=4)
    assert r["team_rating"] == pytest.approx(100.0)
    # xi 11*4 + captain 4 per week
    assert r["current_xi_xp"] == pytest.approx(48.0 * 3)
    c = chip_comparison(current, pool, projections, HORIZON, result, r, chips_used=[("bboost", 3)])
    by = {(row["chip"], row["gw"]): row for row in c["chips"]}
    assert by[("bboost", 4)]["gain"] == pytest.approx(16.0)
    assert by[("3xc", 5)]["gain"] == pytest.approx(4.0)
    assert by[("freehit", 6)]["gain"] == pytest.approx(0.0, abs=1e-6)
    assert by[("bboost", 4)]["available"] is False  # used in gw3, same half
    assert by[("3xc", 4)]["available"] is True
    assert [row["gw"] for row in c["chips"] if row["chip"] == "wildcard"] == [4]
    assert set(c["best_gw"]) == {"bboost", "3xc", "freehit", "wildcard"}


def test_triple_captain_best_gw_is_the_week_with_the_biggest_captain():
    current, pool, projections, result = _setup(3)
    r = team_rating(current, pool, projections, HORIZON, result, hit_cost=4)
    c = chip_comparison(current, pool, projections, HORIZON, result, r)
    caps = {gw: projections[gw][result.captain[gw]] for gw in HORIZON}
    assert c["best_gw"]["3xc"] == max(caps, key=caps.get)
    assert all(row["gain"] >= -1e-6 for row in c["chips"] if row["chip"] in ("bboost", "3xc"))


def test_chips_available_resets_in_second_half():
    assert chips_available([("wildcard", 8)], "wildcard", 25) is True
    assert chips_available([("wildcard", 8)], "wildcard", 12) is False
    assert chips_available([("wildcard", 22)], "wildcard", 30) is False
    assert chips_available([("wildcard", 22)], "wildcard", 19) is True


def test_squad_by_gw_applies_planned_transfers():
    current, pool, projections, result = _setup(4)
    squads = squad_by_gw(result, HORIZON)
    assert squads[HORIZON[0]] == result.squad
    assert all(len(s) == 15 for s in squads.values())
