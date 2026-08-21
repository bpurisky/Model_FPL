"""§5.4/§5.5: the ILP solver never proposes an illegal squad, marginal
value works the way the spec describes, and the horizon control actually
changes the recommendation."""

from __future__ import annotations

import random
from datetime import datetime, timezone

import pytest

from squad.optimize import Player, optimize_squad, template_risk_flags
from squad.reconstruct import SquadPlayer, SquadState

AS_OF = datetime(2026, 8, 24, tzinfo=timezone.utc)
CLUBS = ["ARS", "AVL", "BOU", "BRE", "BHA", "CHE", "CRY"]


def _squad_player(eid: int, slot: int, price: int = 50) -> SquadPlayer:
    return SquadPlayer(
        element_id=eid, purchase_price=price, selling_price=price,
        squad_position=slot, multiplier=1, is_captain=False, is_vice_captain=False,
    )


def _make_universe(seed: int, bank: int = 0):
    """A minimal legal 15-man squad (2 GK/5 DEF/5 MID/3 FWD across enough
    clubs to respect the 3-per-club cap) plus a handful of extra candidate
    players in the same shape, all as a `pool` an optimizer call can use.
    Deterministic per `seed` so property-style tests can sweep many of them."""
    rng = random.Random(seed)
    composition = {"GK": 2, "DEF": 5, "MID": 5, "FWD": 3}
    squad_players: list[SquadPlayer] = []
    pool: list[Player] = []
    eid = 1
    slot = 1
    for pos, count in composition.items():
        for _ in range(count):
            club = CLUBS[eid % len(CLUBS)]
            price = rng.randint(40, 120)
            squad_players.append(_squad_player(eid, slot, price))
            pool.append(Player(element_id=eid, position=pos, club=club, now_cost=price))
            eid += 1
            slot += 1

    # extra transfer candidates, 3 per position, spread across clubs
    for pos, count in composition.items():
        for _ in range(3):
            club = CLUBS[eid % len(CLUBS)]
            price = rng.randint(40, 120)
            pool.append(Player(element_id=eid, position=pos, club=club, now_cost=price))
            eid += 1

    current = SquadState(as_of=AS_OF, players=tuple(squad_players), bank=bank)
    return current, pool


def _flat_projections(pool: list[Player], gws: list[int], value: float = 4.0) -> dict[int, dict[int, float]]:
    return {gw: {p.element_id: value for p in pool} for gw in gws}


def test_no_incentive_to_transfer_keeps_the_squad():
    current, pool = _make_universe(seed=1)
    projections = _flat_projections(pool, [3])  # every player worth the same -> no reason to move
    result = optimize_squad(current, pool, projections, horizon=[3], free_transfers=1)

    assert result.transfers_out == frozenset()
    assert result.transfers_in == frozenset()
    assert result.hits_taken == 0


def test_profitable_transfer_clears_the_horizon_hurdle():
    current, pool = _make_universe(seed=2, bank=100)
    # one current MID is worthless; one candidate MID is a star across the whole horizon
    weak_mid = next(sp.element_id for sp in current.players if sp.squad_position in range(8, 13))
    star_candidate = next(p.element_id for p in pool if p.position == "MID" and p.element_id not in {sp.element_id for sp in current.players})

    projections = _flat_projections(pool, [3, 4, 5], value=2.0)
    for gw in (3, 4, 5):
        projections[gw][weak_mid] = 0.0
        projections[gw][star_candidate] = 10.0

    result = optimize_squad(current, pool, projections, horizon=[3, 4, 5], free_transfers=1)

    assert star_candidate in result.transfers_in
    assert weak_mid in result.transfers_out


def test_bench_only_upgrade_scores_near_zero():
    current, pool = _make_universe(seed=3, bank=100)
    # identify a current bench player (never worth starting) and a slightly
    # better free-agent clone at the same position/price tier.
    bench_id = next(sp.element_id for sp in current.players if sp.squad_position == 12)  # last MID slot
    bench_player = next(p for p in pool if p.element_id == bench_id)
    clone_id = max(p.element_id for p in pool) + 1
    clone = Player(element_id=clone_id, position=bench_player.position, club=bench_player.club, now_cost=bench_player.now_cost)
    pool = list(pool) + [clone]

    gws = [3]
    projections = _flat_projections(pool, gws, value=6.0)  # everyone else is comfortably a starter
    projections[3][bench_id] = 0.1
    projections[3][clone_id] = 0.2  # marginally better, but still never a starter

    baseline = optimize_squad(current, pool, projections, horizon=gws, free_transfers=1, max_transfers=0)
    with_clone = optimize_squad(current, pool, projections, horizon=gws, free_transfers=1)

    # the only possible gain is the bench player's own marginal improvement,
    # which never enters the starting-XI objective at all
    assert with_clone.objective_value == pytest.approx(baseline.objective_value, abs=1e-6)


def test_horizon_control_changes_the_recommendation():
    current, pool = _make_universe(seed=4, bank=100)
    weak_fwd = next(sp.element_id for sp in current.players if sp.squad_position == 14)  # a FWD bench-ish slot
    late_bloomer = next(p.element_id for p in pool if p.position == "FWD" and p.element_id not in {sp.element_id for sp in current.players})

    projections = _flat_projections(pool, [3, 4], value=3.0)
    projections[3][weak_fwd] = 3.0
    projections[3][late_bloomer] = 2.9  # strictly worse in gw3 - no reason to move yet
    projections[4][weak_fwd] = 3.0
    projections[4][late_bloomer] = 12.0  # big edge in gw4

    short = optimize_squad(current, pool, projections, horizon=[3], free_transfers=1)
    long = optimize_squad(current, pool, projections, horizon=[3, 4], free_transfers=1)

    assert late_bloomer not in short.transfers_in
    assert late_bloomer in long.transfers_in


def test_max_transfers_cap_is_respected():
    current, pool = _make_universe(seed=5, bank=200)
    projections = _flat_projections(pool, [3])
    # make every candidate look better than every current player
    current_ids = {sp.element_id for sp in current.players}
    for eid in projections[3]:
        projections[3][eid] = 0.0 if eid in current_ids else 10.0

    result = optimize_squad(current, pool, projections, horizon=[3], free_transfers=1, max_transfers=2)
    assert len(result.transfers_out) <= 2
    assert len(result.transfers_in) <= 2


def test_hits_taken_beyond_free_transfers_are_reported():
    current, pool = _make_universe(seed=6, bank=200)
    current_ids = {sp.element_id for sp in current.players}
    projections = _flat_projections(pool, [3])
    for eid in projections[3]:
        projections[3][eid] = 0.0 if eid in current_ids else 10.0

    result = optimize_squad(current, pool, projections, horizon=[3], free_transfers=1, max_transfers=3)
    assert result.hits_taken == max(0, len(result.transfers_out) - 1)


@pytest.mark.parametrize("seed", range(10))
def test_solver_never_proposes_an_illegal_squad(seed):
    current, pool = _make_universe(seed=seed, bank=random.Random(seed).randint(0, 50))
    pool_by_id = {p.element_id: p for p in pool}
    rng = random.Random(seed + 1000)
    projections = {3: {p.element_id: rng.uniform(0, 12) for p in pool}}

    result = optimize_squad(current, pool, projections, horizon=[3], free_transfers=rng.randint(0, 5))

    assert len(result.squad) == 15
    counts: dict[str, int] = {}
    clubs: dict[str, int] = {}
    for eid in result.squad:
        p = pool_by_id[eid]
        counts[p.position] = counts.get(p.position, 0) + 1
        clubs[p.club] = clubs.get(p.club, 0) + 1
    assert counts == {"GK": 2, "DEF": 5, "MID": 5, "FWD": 3}
    assert all(n <= 3 for n in clubs.values())
    assert result.bank_after >= 0

    xi = result.starting_xi[3]
    assert len(xi) == 11
    assert xi <= result.squad
    xi_positions = {pool_by_id[eid].position for eid in xi}
    gk_count = sum(1 for eid in xi if pool_by_id[eid].position == "GK")
    def_count = sum(1 for eid in xi if pool_by_id[eid].position == "DEF")
    fwd_count = sum(1 for eid in xi if pool_by_id[eid].position == "FWD")
    assert gk_count == 1
    assert def_count >= 3
    assert fwd_count >= 1
    assert result.captain[3] in xi


def test_budget_uses_selling_price_not_now_cost():
    """§5.3/§5.5: budget for a transfer is `selling_price(out) + bank`,
    never `now_cost(out)`. Give one held player a now_cost well above its
    (correctly price-rise-adjusted) selling_price, and confirm the optimizer
    won't fund a transfer that's only affordable if it mistakenly used
    now_cost instead."""
    composition = {"GK": 2, "DEF": 5, "MID": 5, "FWD": 3}
    squad_players: list[SquadPlayer] = []
    pool: list[Player] = []
    eid = 1
    slot = 1
    target_id = None
    for pos, count in composition.items():
        for _ in range(count):
            club = CLUBS[eid % len(CLUBS)]
            if pos == "MID" and target_id is None:
                # bought at 50, risen to a now_cost of 70, but only half the
                # profit is banked and rounded down -> selling_price 60.
                target_id = eid
                squad_players.append(SquadPlayer(
                    element_id=eid, purchase_price=50, selling_price=60,
                    squad_position=slot, multiplier=1, is_captain=False, is_vice_captain=False,
                ))
                pool.append(Player(element_id=eid, position=pos, club=club, now_cost=70))
            else:
                squad_players.append(_squad_player(eid, slot, price=40))
                pool.append(Player(element_id=eid, position=pos, club=club, now_cost=40))
            eid += 1
            slot += 1

    current = SquadState(as_of=AS_OF, players=tuple(squad_players), bank=0)

    # priced at 65: affordable against the target's now_cost (70) but not
    # its selling_price (60), and no other held player can fund it either
    # (every other held player is worth only 40).
    candidate_id = eid
    candidate = Player(element_id=candidate_id, position="MID", club=CLUBS[0], now_cost=65)
    pool.append(candidate)

    projections = _flat_projections(pool, [3], value=2.0)
    projections[3][target_id] = 0.0
    projections[3][candidate_id] = 20.0  # overwhelming incentive, if only it were affordable

    result = optimize_squad(current, pool, projections, horizon=[3], free_transfers=1)

    assert candidate_id not in result.transfers_in
    assert target_id not in result.transfers_out


def test_template_risk_flag_fires_on_high_ownership_sale():
    ownership = {1: 55.0, 2: 3.0}
    flags = template_risk_flags(transfers_out=frozenset({1, 2}), ownership=ownership, threshold=20.0)

    assert 1 in flags
    assert 2 not in flags
