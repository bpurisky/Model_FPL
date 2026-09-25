"""§5.4 the recommender: an integer linear program, not a sort, planned
week by week over a horizon of H gameweeks.

Every gameweek of the horizon has its own squad, transfers, bank, free
transfers, starting XI, captain, vice-captain and bench order, linked to
the week before: a player bought in week 2 is in the squad from week 2 on,
money spent in week 1 is gone in week 2, and a free transfer not used this
week is there next week (up to the season's banking cap). So the solver
can wait a week for a fixture to turn, bank a transfer, or take a hit now
because it sets up two good weeks, where the single-decision model it
replaced could only make every move at once and hold the squad after.

The objective is projected points, decayed by `decay` per week of
distance (a projection three weeks out is worth less than next week's:
it is less accurate and more likely to be overtaken by news), with:

- the captain doubled, and the vice-captain and each bench slot counted
  at a small weight — roughly the chance they end up playing, which is
  what makes a bench upgrade worth *something* rather than exactly zero;
- a hit cost for every transfer beyond the free ones;
- `ft_value` for each free transfer still banked after the last week, so
  the solver does not spend a transfer on a marginal move just because
  the horizon ends.

The weights are FPL-Optimization-Tools' published defaults
(sertalpbilal/FPL-Optimization-Tools, comprehensive_settings.json),
checked here by season simulation (squad/simulate.py) rather than assumed.
Over 2023-24, 2024-25 and 2025-26 (gw2-38, the same starting squad and
the same walk-forward projections, three-week horizon) it outscored the
single-decision model it replaced every season — 2106 v 2098, 2258 v
2254, 2101 v 2043 — with half the hits (35 v 69). The margin is modest
against how far one season's path moves with small setting changes (up to
~130 points), which is why the defaults are taken as published rather
than tuned to three seasons. It is also far less sensitive to the
horizon: at one week the old model lost 94-130 points a season, this one
about the same as at three.

Only the first week's transfers are a decision; the rest are the plan
that made them worth making, and are re-solved next week with new
information. `OptimizationResult`'s first-week fields therefore mean what
they always meant, and the plan rides alongside.

This module takes a `projections` table (gw -> element_id -> points) as a
plain argument and never imports analytics.projections — whose
projections they are is the caller's concern — so it stays testable
against synthetic projections.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

import pulp

from squad.reconstruct import SquadState

SQUAD_COMPOSITION = {"GK": 2, "DEF": 5, "MID": 5, "FWD": 3}
MAX_PER_CLUB = 3
XI_SIZE = 11
MIN_DEF_IN_XI = 3
MIN_MID_IN_XI = 2
MIN_FWD_IN_XI = 1
MAX_GK_IN_XI = 1

# A transfer that doesn't change the objective (e.g. swapping to an
# equally-projected player) is otherwise a free tie for the solver, which
# can then propose a pointless move. This tie-break is far smaller than any
# real projection gap the tests exercise (horizon deltas as small as 0.1),
# so it only ever resolves ties, never a genuine transfer decision.
TRANSFER_TIE_BREAK_EPSILON = 1e-4

# FPL-Optimization-Tools' defaults (see module docstring).
DEFAULT_DECAY = 0.9
DEFAULT_FT_VALUE = 1.5
DEFAULT_VICE_WEIGHT = 0.1
# Substitute goalkeeper, then outfield bench slots 1-3.
DEFAULT_BENCH_WEIGHTS = (0.03, 0.21, 0.06, 0.002)
DEFAULT_MAX_BANKED = 5


@dataclass(frozen=True)
class Player:
    element_id: int
    position: str  # GK/DEF/MID/FWD
    club: str
    now_cost: int  # 0.1m units — the price paid if this player is bought


@dataclass(frozen=True)
class PlannedWeek:
    """A later week of the plan: what the solver expects to do then, given
    what it knows now."""

    gw: int
    transfers_out: frozenset[int]
    transfers_in: frozenset[int]
    hits: int
    free_transfers_before: int


@dataclass(frozen=True)
class OptimizationResult:
    squad: frozenset[int]  # after this week's transfers
    transfers_out: frozenset[int]  # this week's
    transfers_in: frozenset[int]
    starting_xi: dict[int, frozenset[int]]  # gw -> element_ids, from that week's planned squad
    captain: dict[int, int]  # gw -> element_id
    bench_order: tuple[int, ...]  # this week's: substitute GK, then outfield bench 1-3
    bank_after: int  # after this week's transfers
    hits_taken: int  # this week's
    objective_value: float
    vice_captain: dict[int, int] = field(default_factory=dict)  # gw -> element_id
    plan: tuple[PlannedWeek, ...] = ()  # the horizon's later weeks, in order
    free_transfers_after: int | None = None  # banked going into next week


def optimize_squad(
    current: SquadState,
    pool: Sequence[Player],
    projections: Mapping[int, Mapping[int, float]],
    horizon: Sequence[int],
    free_transfers: int,
    max_transfers: int | None = None,
    hit_cost: int = 4,
    *,
    decay: float = DEFAULT_DECAY,
    ft_value: float = DEFAULT_FT_VALUE,
    vice_weight: float = DEFAULT_VICE_WEIGHT,
    bench_weights: Sequence[float] = DEFAULT_BENCH_WEIGHTS,
    max_banked: int = DEFAULT_MAX_BANKED,
    plan_future_transfers: bool = True,
) -> OptimizationResult:
    """`pool` must include every currently-owned player (as a `Player`,
    for its position/club/now_cost) plus every transfer-in candidate.
    `horizon` is the ordered list of gameweeks to plan over — changing its
    length or contents is expected to change the plan (§5.5): a striker
    with a brutal next fixture but a soft run after it should look
    different at H=1 than H=6.

    `max_transfers` caps this week's transfers only (the freeze uses it to
    forbid hits while the model's history is thin); later weeks are plan,
    not action. `plan_future_transfers=False` holds the squad after this
    week, which with zero weights, no decay and no ft_value reproduces the
    single-decision model this replaced — kept so squad/simulate.py can
    measure one against the other.
    """
    pool_by_id = {p.element_id: p for p in pool}
    current_ids = {sp.element_id for sp in current.players}
    selling_price = {sp.element_id: sp.selling_price for sp in current.players}

    missing = current_ids - pool_by_id.keys()
    if missing:
        raise ValueError(f"pool is missing currently-owned players: {sorted(missing)}")
    if not horizon:
        raise ValueError("horizon must be non-empty")
    if len(bench_weights) != 4:
        raise ValueError("bench_weights needs four entries: substitute GK, then outfield bench 1-3")

    horizon = list(horizon)
    all_ids = sorted(pool_by_id)
    position = {e: pool_by_id[e].position for e in all_ids}
    clubs = sorted({p.club for p in pool})
    # A player bought and sold again inside the horizon sells for what he
    # cost (price changes are not modelled); one owned now sells at his
    # FPL selling price.
    sell_value = {e: selling_price.get(e, pool_by_id[e].now_cost) for e in all_ids}

    prob = pulp.LpProblem("squad_plan", pulp.LpMaximize)
    squad: dict[int, dict[int, pulp.LpVariable]] = {}
    buy: dict[int, dict[int, pulp.LpVariable]] = {}
    sell: dict[int, dict[int, pulp.LpVariable]] = {}
    start: dict[int, dict[int, pulp.LpVariable]] = {}
    captain: dict[int, dict[int, pulp.LpVariable]] = {}
    vice: dict[int, dict[int, pulp.LpVariable]] = {}
    bench: dict[int, dict[tuple[int, int], pulp.LpVariable]] = {}
    bank: dict[int, pulp.LpVariable] = {}
    hits: dict[int, pulp.LpVariable] = {}
    fts_after: dict[int, pulp.LpVariable] = {}
    fts_before: int | pulp.LpVariable = free_transfers  # a constant for week 0, a variable after
    objective = []

    for i, gw in enumerate(horizon):
        squad[i] = {e: pulp.LpVariable(f"x_{i}_{e}", cat="Binary") for e in all_ids}
        buy[i] = {e: pulp.LpVariable(f"buy_{i}_{e}", cat="Binary") for e in all_ids}
        sell[i] = {e: pulp.LpVariable(f"sell_{i}_{e}", cat="Binary") for e in all_ids}
        for e in all_ids:
            before = (1 if e in current_ids else 0) if i == 0 else squad[i - 1][e]
            prob += squad[i][e] == before + buy[i][e] - sell[i][e]
            prob += buy[i][e] + sell[i][e] <= 1
            if i > 0 and not plan_future_transfers:
                prob += buy[i][e] == 0
                prob += sell[i][e] == 0

        for pos, count in SQUAD_COMPOSITION.items():
            prob += pulp.lpSum(squad[i][e] for e in all_ids if position[e] == pos) == count
        for club in clubs:
            prob += pulp.lpSum(squad[i][e] for e in all_ids if pool_by_id[e].club == club) <= MAX_PER_CLUB

        bank_before = current.bank if i == 0 else bank[i - 1]
        bank[i] = pulp.LpVariable(f"bank_{i}", lowBound=0)
        prob += bank[i] == bank_before + pulp.lpSum(sell_value[e] * sell[i][e] for e in all_ids) - pulp.lpSum(
            pool_by_id[e].now_cost * buy[i][e] for e in all_ids
        )

        # Free transfers: `used` of them cover this week's transfers and
        # the rest are hits; next week gets what's left plus one, up to the
        # banking cap.
        n_transfers = pulp.lpSum(buy[i].values())
        used = pulp.LpVariable(f"ft_used_{i}", lowBound=0, cat="Integer")
        hits[i] = pulp.LpVariable(f"hits_{i}", lowBound=0, cat="Integer")
        prob += used <= fts_before
        prob += used <= n_transfers
        prob += hits[i] == n_transfers - used
        fts_after[i] = pulp.LpVariable(f"ft_after_{i}", lowBound=0, upBound=max_banked, cat="Integer")
        prob += fts_after[i] <= fts_before - used + 1
        fts_before = fts_after[i]
        if i == 0 and max_transfers is not None:
            prob += n_transfers <= max_transfers

        # The team sheet: every squad player either starts or holds exactly
        # one bench slot — slot 0 the substitute keeper, slots 1-3 outfield.
        start[i] = {e: pulp.LpVariable(f"start_{i}_{e}", cat="Binary") for e in all_ids}
        captain[i] = {e: pulp.LpVariable(f"cap_{i}_{e}", cat="Binary") for e in all_ids}
        vice[i] = {e: pulp.LpVariable(f"vice_{i}_{e}", cat="Binary") for e in all_ids}
        bench[i] = {
            (e, k): pulp.LpVariable(f"bench_{i}_{k}_{e}", cat="Binary")
            for e in all_ids
            for k in range(4)
            if (k == 0) == (position[e] == "GK")
        }
        for e in all_ids:
            slots = pulp.lpSum(bench[i][(e, k)] for k in range(4) if (e, k) in bench[i])
            prob += start[i][e] + slots == squad[i][e]
            prob += captain[i][e] + vice[i][e] <= start[i][e]
        for k in range(4):
            prob += pulp.lpSum(v for (_, slot), v in bench[i].items() if slot == k) == 1
        prob += pulp.lpSum(start[i].values()) == XI_SIZE
        prob += pulp.lpSum(start[i][e] for e in all_ids if position[e] == "GK") == MAX_GK_IN_XI
        prob += pulp.lpSum(start[i][e] for e in all_ids if position[e] == "DEF") >= MIN_DEF_IN_XI
        prob += pulp.lpSum(start[i][e] for e in all_ids if position[e] == "MID") >= MIN_MID_IN_XI
        prob += pulp.lpSum(start[i][e] for e in all_ids if position[e] == "FWD") >= MIN_FWD_IN_XI
        prob += pulp.lpSum(captain[i].values()) == 1
        prob += pulp.lpSum(vice[i].values()) == 1

        weight = decay**i
        gw_proj = projections.get(gw, {})
        for e in all_ids:
            pts = gw_proj.get(e, 0.0)
            if pts == 0.0:
                continue
            objective.append(weight * pts * (start[i][e] + captain[i][e] + vice_weight * vice[i][e]))
            objective.extend(
                weight * pts * bench_weights[k] * bench[i][(e, k)] for k in range(4) if (e, k) in bench[i]
            )
        objective.append(-weight * hit_cost * hits[i])
        objective.append(-TRANSFER_TIE_BREAK_EPSILON * n_transfers)

    objective.append(ft_value * fts_after[len(horizon) - 1])
    prob += pulp.lpSum(objective)

    status = prob.solve(pulp.PULP_CBC_CMD(msg=False))
    if pulp.LpStatus[status] != "Optimal":
        raise RuntimeError(f"solver did not find an optimal solution: {pulp.LpStatus[status]}")

    def chosen(variables: Mapping[int, pulp.LpVariable]) -> frozenset[int]:
        return frozenset(e for e, v in variables.items() if v.value() > 0.5)

    bench_first = tuple(
        next(e for (e, slot), v in bench[0].items() if slot == k and v.value() > 0.5) for k in range(4)
    )
    free_transfers_path = [free_transfers] + [int(round(fts_after[i].value())) for i in range(len(horizon))]
    return OptimizationResult(
        squad=chosen(squad[0]),
        transfers_out=chosen(sell[0]),
        transfers_in=chosen(buy[0]),
        starting_xi={gw: chosen(start[i]) for i, gw in enumerate(horizon)},
        captain={gw: next(iter(chosen(captain[i]))) for i, gw in enumerate(horizon)},
        bench_order=bench_first,
        bank_after=int(round(bank[0].value())),
        hits_taken=int(round(hits[0].value())),
        objective_value=pulp.value(prob.objective),
        vice_captain={gw: next(iter(chosen(vice[i]))) for i, gw in enumerate(horizon)},
        plan=tuple(
            PlannedWeek(
                gw=gw,
                transfers_out=chosen(sell[i]),
                transfers_in=chosen(buy[i]),
                hits=int(round(hits[i].value())),
                free_transfers_before=free_transfers_path[i],
            )
            for i, gw in enumerate(horizon)
            if i > 0
        ),
        free_transfers_after=free_transfers_path[1],
    )


def prune_pool(
    current: SquadState,
    pool: Sequence[Player],
    projections: Mapping[int, Mapping[int, float]],
    horizon: Sequence[int],
    per_position: int = 20,
) -> list[Player]:
    """The players worth giving the solver: everyone owned, plus per
    position the `per_position` best by projected points over the horizon
    and the `per_position` best by points per price (the cheap enablers a
    budget needs). The weekly plan has ten binaries per player per week;
    offering it 600 players it would never pick only makes it slow.
    """
    owned = {sp.element_id for sp in current.players}
    total = {p.element_id: sum(projections.get(gw, {}).get(p.element_id, 0.0) for gw in horizon) for p in pool}
    keep = set(owned)
    for pos in SQUAD_COMPOSITION:
        players = [p for p in pool if p.position == pos]
        keep.update(p.element_id for p in sorted(players, key=lambda p: -total[p.element_id])[:per_position])
        keep.update(
            p.element_id
            for p in sorted(players, key=lambda p: -total[p.element_id] / max(p.now_cost, 1))[:per_position]
        )
    return [p for p in pool if p.element_id in keep]


def pair_transfers_by_position(
    transfers_out: frozenset[int], transfers_in: frozenset[int], pool_by_id: Mapping[int, Player]
) -> list[tuple[int, int]]:
    """Squad composition is fixed per position (§5.4's composition
    constraint), so a transfer never changes the position mix — the number
    sold and bought at each position always matches exactly, which makes
    "who replaced whom" well-defined per position even though `optimize_squad`
    only returns the two sets, not a pairing. Pairing across positions
    instead (e.g. a plain zip of two id-sorted sets) produces nonsense like
    a sold goalkeeper "paired" with a bought defender — this is the one
    correct way to present or apply a multi-transfer recommendation.
    """
    pairs: list[tuple[int, int]] = []
    for pos in ["GK", "DEF", "MID", "FWD"]:
        outs = sorted(eid for eid in transfers_out if pool_by_id[eid].position == pos)
        ins = sorted(eid for eid in transfers_in if pool_by_id[eid].position == pos)
        pairs.extend(zip(outs, ins))
    return pairs


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
