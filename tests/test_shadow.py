"""§6.2 shadow-team simulation."""

from __future__ import annotations

from datetime import datetime, timezone

from squad.optimize import OptimizationResult, Player
from squad.reconstruct import SquadPlayer, SquadState
from squad.shadow import apply_recommendation, realized_points

AS_OF = datetime(2026, 8, 28, tzinfo=timezone.utc)


def _minimal_squad() -> tuple[SquadState, dict[int, Player]]:
    """A tiny legal-shaped squad: 2 GK, 5 DEF, 5 MID, 3 FWD, ids 1-15,
    every player bought at 50 and unchanged since."""
    composition = {"GK": 2, "DEF": 5, "MID": 5, "FWD": 3}
    players = []
    pool: dict[int, Player] = {}
    eid = 1
    for pos, count in composition.items():
        for _ in range(count):
            players.append(SquadPlayer(
                element_id=eid, purchase_price=50, selling_price=50, squad_position=eid,
                multiplier=(2 if eid == 11 else 1) if eid <= 11 else 0,
                is_captain=eid == 11, is_vice_captain=eid == 10,
            ))
            pool[eid] = Player(element_id=eid, position=pos, club="ARS" if eid % 2 else "CHE", now_cost=50)
            eid += 1
    return SquadState(as_of=AS_OF, players=tuple(players), bank=0), pool


def test_apply_recommendation_preserves_squad_composition():
    state, pool = _minimal_squad()
    # sell element 9 (MID), buy element 16 (a new MID candidate)
    pool[16] = Player(element_id=16, position="MID", club="LIV", now_cost=55)
    new_squad = (frozenset(range(1, 16)) - {9}) | {16}
    xi = (frozenset(range(1, 12)) - {9}) | {16}
    bench = tuple(sorted(new_squad - xi))
    result = OptimizationResult(
        squad=new_squad, transfers_out=frozenset({9}), transfers_in=frozenset({16}),
        starting_xi={2: xi}, captain={2: 11}, bench_order=bench,
        bank_after=5, hits_taken=0, objective_value=0.0,
    )

    new_state = apply_recommendation(state, result, pool, {eid: p.now_cost for eid, p in pool.items()}, next_gw=2, as_of=AS_OF)

    ids = {sp.element_id for sp in new_state.players}
    assert ids == new_squad
    assert len(new_state.players) == 15
    counts: dict[str, int] = {}
    for sp in new_state.players:
        counts[pool[sp.element_id].position] = counts.get(pool[sp.element_id].position, 0) + 1
    assert counts == {"GK": 2, "DEF": 5, "MID": 5, "FWD": 3}


def test_apply_recommendation_bought_player_gets_todays_price_as_purchase_price():
    state, pool = _minimal_squad()
    pool[16] = Player(element_id=16, position="MID", club="LIV", now_cost=62)
    new_squad = (frozenset(range(1, 16)) - {9}) | {16}
    xi = (frozenset(range(1, 12)) - {9}) | {16}
    bench = tuple(sorted(new_squad - xi))
    result = OptimizationResult(
        squad=new_squad, transfers_out=frozenset({9}), transfers_in=frozenset({16}),
        starting_xi={2: xi}, captain={2: 11}, bench_order=bench,
        bank_after=5, hits_taken=0, objective_value=0.0,
    )

    new_state = apply_recommendation(state, result, pool, {eid: p.now_cost for eid, p in pool.items()}, next_gw=2, as_of=AS_OF)

    bought = next(sp for sp in new_state.players if sp.element_id == 16)
    assert bought.purchase_price == 62
    assert bought.selling_price == 62  # no price move possible yet, just bought


def test_apply_recommendation_captain_gets_double_multiplier_and_bench_gets_zero():
    state, pool = _minimal_squad()
    new_squad = frozenset(range(1, 16))
    xi = frozenset(range(1, 12))
    bench = (12, 13, 14, 15)
    result = OptimizationResult(
        squad=new_squad, transfers_out=frozenset(), transfers_in=frozenset(),
        starting_xi={2: xi}, captain={2: 5}, bench_order=bench,
        bank_after=0, hits_taken=0, objective_value=0.0,
    )

    new_state = apply_recommendation(state, result, pool, {eid: p.now_cost for eid, p in pool.items()}, next_gw=2, as_of=AS_OF)

    by_id = {sp.element_id: sp for sp in new_state.players}
    assert by_id[5].multiplier == 2 and by_id[5].is_captain
    for eid in bench:
        assert by_id[eid].multiplier == 0
    for eid in xi - {5}:
        assert by_id[eid].multiplier == 1


def test_apply_recommendation_no_transfer_holds_squad_and_prices():
    state, pool = _minimal_squad()
    new_squad = frozenset(range(1, 16))
    xi = frozenset(range(1, 12))
    result = OptimizationResult(
        squad=new_squad, transfers_out=frozenset(), transfers_in=frozenset(),
        starting_xi={2: xi}, captain={2: 11}, bench_order=(12, 13, 14, 15),
        bank_after=0, hits_taken=0, objective_value=0.0,
    )

    new_state = apply_recommendation(state, result, pool, {eid: 50 for eid in pool}, next_gw=2, as_of=AS_OF)

    assert {sp.element_id for sp in new_state.players} == set(range(1, 16))
    for sp in new_state.players:
        assert sp.selling_price == 50  # no price move


def test_apply_recommendation_refreshes_selling_price_on_price_rise():
    state, pool = _minimal_squad()
    new_squad = frozenset(range(1, 16))
    xi = frozenset(range(1, 12))
    result = OptimizationResult(
        squad=new_squad, transfers_out=frozenset(), transfers_in=frozenset(),
        starting_xi={2: xi}, captain={2: 11}, bench_order=(12, 13, 14, 15),
        bank_after=0, hits_taken=0, objective_value=0.0,
    )
    now_cost_by_id = {eid: 50 for eid in pool}
    now_cost_by_id[3] = 58  # element 3 has risen from 50 -> 58

    new_state = apply_recommendation(state, result, pool, now_cost_by_id, next_gw=2, as_of=AS_OF)

    risen = next(sp for sp in new_state.players if sp.element_id == 3)
    assert risen.selling_price == 50 + (58 - 50) // 2  # half profit, rounded down


def test_realized_points_counts_captain_double_and_ignores_bench():
    state, _ = _minimal_squad()  # captain is element 11, bench is 12-15
    actuals = {eid: 4 for eid in range(1, 16)}  # everyone scores 4
    actuals[11] = 10  # captain scores 10 -> counts as 20

    points = realized_points(state, actuals)

    # starters 1-10 (4 pts each, 40) + captain 11 (10*2=20) = 60; bench (12-15) contributes 0
    assert points == 60
