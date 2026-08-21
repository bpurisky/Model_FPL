"""§5.4 the recommender: an integer linear program, not a sort. Maximizes
projected starting-XI points (captain doubled) summed over a horizon of H
gameweeks, minus one-time hit costs, subject to budget, club-limit, squad-
composition and legal-formation constraints. Value is marginal by
construction — the objective is XI points, so upgrading a bench player who
never starts contributes nothing unless it changes who starts.

This module takes a `projections` table (gw -> element_id -> points) as a
plain argument and never imports analytics.projections — the model-wiring
question of *whose* projections (historical backtest vs a live 2026/27
feed) is the caller's concern, and there is no live prediction pipeline
wired up yet (§5's own progress log flags this; likely Phase 5 territory).
Keeping this module decoupled means it stays testable against synthetic
projections regardless of when that wiring lands.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import pulp

from squad.reconstruct import SquadState

SQUAD_COMPOSITION = {"GK": 2, "DEF": 5, "MID": 5, "FWD": 3}
MAX_PER_CLUB = 3
XI_SIZE = 11
MIN_DEF_IN_XI = 3
MIN_FWD_IN_XI = 1
MAX_GK_IN_XI = 1

# A transfer that doesn't change the objective (e.g. swapping to an
# equally-projected player) is otherwise a free tie for the solver, which
# can then propose a pointless move. This tie-break is far smaller than any
# real projection gap the tests exercise (horizon deltas as small as 0.1),
# so it only ever resolves ties, never a genuine transfer decision.
TRANSFER_TIE_BREAK_EPSILON = 1e-4


@dataclass(frozen=True)
class Player:
    element_id: int
    position: str  # GK/DEF/MID/FWD
    club: str
    now_cost: int  # 0.1m units — the price paid if this player is bought


@dataclass(frozen=True)
class OptimizationResult:
    squad: frozenset[int]
    transfers_out: frozenset[int]
    transfers_in: frozenset[int]
    starting_xi: dict[int, frozenset[int]]  # gw -> element_ids
    captain: dict[int, int]  # gw -> element_id
    bench_order: tuple[int, ...]  # element_ids, best-first by next-gw projection
    bank_after: int
    hits_taken: int
    objective_value: float


def optimize_squad(
    current: SquadState,
    pool: Sequence[Player],
    projections: Mapping[int, Mapping[int, float]],
    horizon: Sequence[int],
    free_transfers: int,
    max_transfers: int | None = None,
    hit_cost: int = 4,
) -> OptimizationResult:
    """`pool` must include every currently-owned player (as a `Player`,
    for its position/club/now_cost) plus every transfer-in candidate.
    `horizon` is the ordered list of gameweeks to project the XI over —
    changing its length or contents is expected to change the
    recommendation (§5.5): a striker with a brutal next fixture but a soft
    run after it should look different at H=1 than H=6.
    """
    pool_by_id = {p.element_id: p for p in pool}
    current_ids = {sp.element_id for sp in current.players}
    selling_price = {sp.element_id: sp.selling_price for sp in current.players}

    missing = current_ids - pool_by_id.keys()
    if missing:
        raise ValueError(f"pool is missing currently-owned players: {sorted(missing)}")
    if not horizon:
        raise ValueError("horizon must be non-empty")

    prob = pulp.LpProblem("squad_transfer", pulp.LpMaximize)
    all_ids = sorted(pool_by_id)
    squad_var = {eid: pulp.LpVariable(f"squad_{eid}", cat="Binary") for eid in all_ids}

    for pos, count in SQUAD_COMPOSITION.items():
        prob += pulp.lpSum(squad_var[eid] for eid in all_ids if pool_by_id[eid].position == pos) == count
    prob += pulp.lpSum(squad_var.values()) == 15

    clubs = {pool_by_id[eid].club for eid in all_ids}
    for club in clubs:
        prob += pulp.lpSum(squad_var[eid] for eid in all_ids if pool_by_id[eid].club == club) <= MAX_PER_CLUB

    bought_cost = pulp.lpSum(pool_by_id[eid].now_cost * squad_var[eid] for eid in all_ids if eid not in current_ids)
    sold_value = pulp.lpSum(selling_price[eid] * (1 - squad_var[eid]) for eid in current_ids)
    prob += bought_cost <= current.bank + sold_value

    transfers_made = pulp.lpSum(1 - squad_var[eid] for eid in current_ids)
    if max_transfers is not None:
        prob += transfers_made <= max_transfers

    hits = pulp.LpVariable("hits", lowBound=0, cat="Integer")
    prob += hits >= transfers_made - free_transfers

    start_vars: dict[int, dict[int, pulp.LpVariable]] = {}
    captain_vars: dict[int, dict[int, pulp.LpVariable]] = {}
    objective_terms = []
    for gw in horizon:
        gw_proj = projections.get(gw, {})
        start = {eid: pulp.LpVariable(f"start_{gw}_{eid}", cat="Binary") for eid in all_ids}
        captain = {eid: pulp.LpVariable(f"cap_{gw}_{eid}", cat="Binary") for eid in all_ids}
        for eid in all_ids:
            prob += start[eid] <= squad_var[eid]
            prob += captain[eid] <= start[eid]
        prob += pulp.lpSum(start.values()) == XI_SIZE
        prob += pulp.lpSum(start[eid] for eid in all_ids if pool_by_id[eid].position == "GK") == MAX_GK_IN_XI
        prob += pulp.lpSum(start[eid] for eid in all_ids if pool_by_id[eid].position == "DEF") >= MIN_DEF_IN_XI
        prob += pulp.lpSum(start[eid] for eid in all_ids if pool_by_id[eid].position == "FWD") >= MIN_FWD_IN_XI
        prob += pulp.lpSum(captain.values()) == 1
        start_vars[gw] = start
        captain_vars[gw] = captain
        for eid in all_ids:
            pts = gw_proj.get(eid, 0.0)
            objective_terms.append(pts * start[eid])
            objective_terms.append(pts * captain[eid])

    prob += pulp.lpSum(objective_terms) - hit_cost * hits - TRANSFER_TIE_BREAK_EPSILON * transfers_made

    status = prob.solve(pulp.PULP_CBC_CMD(msg=False))
    if pulp.LpStatus[status] != "Optimal":
        raise RuntimeError(f"solver did not find an optimal solution: {pulp.LpStatus[status]}")

    squad = frozenset(eid for eid in all_ids if squad_var[eid].value() > 0.5)
    transfers_out = frozenset(current_ids - squad)
    transfers_in = frozenset(squad - current_ids)
    starting_xi = {gw: frozenset(eid for eid in all_ids if start_vars[gw][eid].value() > 0.5) for gw in horizon}
    captain = {gw: next(eid for eid in all_ids if captain_vars[gw][eid].value() > 0.5) for gw in horizon}

    bank_after = int(
        current.bank
        + sum(selling_price[eid] for eid in transfers_out)
        - sum(pool_by_id[eid].now_cost for eid in transfers_in)
    )

    next_gw = horizon[0]
    next_proj = projections.get(next_gw, {})
    bench = sorted(squad - starting_xi[next_gw], key=lambda eid: next_proj.get(eid, 0.0), reverse=True)

    return OptimizationResult(
        squad=squad,
        transfers_out=transfers_out,
        transfers_in=transfers_in,
        starting_xi=starting_xi,
        captain=captain,
        bench_order=tuple(bench),
        bank_after=bank_after,
        hits_taken=int(round(hits.value())),
        objective_value=pulp.value(prob.objective),
    )


def template_risk_flags(
    transfers_out: frozenset[int],
    ownership: Mapping[int, float],
    threshold: float = 20.0,
) -> dict[int, str]:
    """§5.4: a pure points model will happily sell a 55%-owned premium for
    a marginal expected-points gain — a poor trade against the field, since
    a blank week costs a differential-holder ground everyone else keeps,
    but costs a template-holder nothing. This flags, it doesn't block: the
    solver stays a pure points optimizer, and risk communication is an
    explicit layer on top of its output, not baked into the objective.
    """
    return {
        eid: f"selling a player owned by {ownership[eid]:.1f}% of squads is a high-variance move against the template"
        for eid in transfers_out
        if ownership.get(eid, 0.0) >= threshold
    }
