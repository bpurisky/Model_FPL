"""§6.2 shadow-team simulation: a second squad that starts identical to the
real team at gameweek 1 (the initial pick predates the model and isn't
something to second-guess) and then, every gameweek from gw2 on,
mechanically applies `squad.optimize.optimize_squad`'s own recommendation
with no human veto. Comparing the shadow team's results against the real
team's at gw13 answers §6's actual question: does the solver beat the
operator's own judgment.

This module only realizes one gameweek's recommendation into the next
`SquadState` — persisting the shadow team across weekly runs, and freezing
its recommendation immutably before each deadline, is `papertrade/freeze.py`'s
job (§6.1: the shadow team's committed move for a gameweek must never be
revised after the fact, same as the projections it's based on).
"""

from __future__ import annotations

from datetime import datetime
from typing import Mapping

from squad.optimize import OptimizationResult, Player, pair_transfers_by_position
from squad.reconstruct import SquadPlayer, SquadState, apply_price_move


def apply_recommendation(
    state: SquadState,
    result: OptimizationResult,
    pool_by_id: Mapping[int, Player],
    now_cost_by_id: Mapping[int, int],
    next_gw: int,
    as_of: datetime,
) -> SquadState:
    """Mechanically realizes `result` (already computed by `optimize_squad`
    against `state`) into the next `SquadState`: sells `transfers_out`, buys
    `transfers_in` at today's `now_cost_by_id` (paired by position, since
    squad composition is fixed per position — see
    `pair_transfers_by_position` — a bought player inherits the sold
    player's old squad_position slot), refreshes every kept player's
    selling price against `now_cost_by_id` (§5.3's half-profit-rounded-down
    rule), and lays out squad_position 1-11 as `result.starting_xi[next_gw]`
    (captain from `result.captain[next_gw]`, multiplier 2) and 12-15 as
    `result.bench_order` (multiplier 0).

    Deliberately no autosub simulation: a starter who blanks is not
    backfilled from the bench here. Full autosub logic (formation-legal
    substitution, captain-blank-reassigns-to-vice) is real FPL behavior but
    adds real complexity for a paper-trade evaluation whose own spec (§6.3)
    already asks readers to treat 13 squad-level observations as
    variance-dominated, not to over-index on exact point totals.
    """
    kept = {sp.element_id: sp for sp in state.players if sp.element_id not in result.transfers_out}
    pairs = pair_transfers_by_position(result.transfers_out, result.transfers_in, pool_by_id)
    slot_by_out_id = {
        out_id: next(sp.squad_position for sp in state.players if sp.element_id == out_id) for out_id, _ in pairs
    }

    bought: dict[int, SquadPlayer] = {}
    for out_id, in_id in pairs:
        price = now_cost_by_id[in_id]
        bought[in_id] = SquadPlayer(
            element_id=in_id, purchase_price=price, selling_price=price,
            squad_position=slot_by_out_id[out_id], multiplier=1, is_captain=False, is_vice_captain=False,
        )

    refreshed_kept = {
        eid: SquadPlayer(
            element_id=sp.element_id,
            purchase_price=sp.purchase_price,
            selling_price=apply_price_move(sp.purchase_price, now_cost_by_id.get(eid, sp.selling_price)),
            squad_position=sp.squad_position, multiplier=sp.multiplier,
            is_captain=sp.is_captain, is_vice_captain=sp.is_vice_captain,
        )
        for eid, sp in kept.items()
    }
    all_by_id = {**refreshed_kept, **bought}

    captain_id = result.captain[next_gw]
    xi = sorted(result.starting_xi[next_gw], key=lambda eid: (pool_by_id[eid].position != "GK", eid))
    vice_id = next((eid for eid in xi if eid != captain_id), captain_id)
    bench = list(result.bench_order)

    final_players = [
        SquadPlayer(
            element_id=eid, purchase_price=all_by_id[eid].purchase_price, selling_price=all_by_id[eid].selling_price,
            squad_position=slot, multiplier=2 if eid == captain_id else 1,
            is_captain=eid == captain_id, is_vice_captain=eid == vice_id,
        )
        for slot, eid in enumerate(xi, start=1)
    ] + [
        SquadPlayer(
            element_id=eid, purchase_price=all_by_id[eid].purchase_price, selling_price=all_by_id[eid].selling_price,
            squad_position=slot, multiplier=0, is_captain=False, is_vice_captain=False,
        )
        for slot, eid in enumerate(bench, start=12)
    ]

    return SquadState(as_of=as_of, players=tuple(final_players), bank=result.bank_after)


def realized_points(state: SquadState, actual_points_by_id: Mapping[int, int]) -> int:
    """Points the shadow squad actually scored: sum over its starting XI
    (multiplier 1, or 2 for the captain) of realized per-player points —
    no autosub credit for a bench player when a starter blanks, per this
    module's documented simplification."""
    return sum(actual_points_by_id.get(sp.element_id, 0) * sp.multiplier for sp in state.players if sp.multiplier > 0)
