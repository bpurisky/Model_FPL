"""Team rating and chip comparison for the Squad Optimizer.

Everything here is a *projection* from the same per-gameweek xP the
recommender uses, summed undecayed over the horizon (the optimizer's own
objective decays later weeks and weights bench slots; these figures are
plain expected points so a reader can add them up).

Definitions
-----------
- XI points of a gameweek: the xP of the 11 starters plus the captain's
  xP once more (captain scores double).
- `current_xi_xp`: the current squad held for the whole horizon, zero
  transfers, best XI and captain chosen each week.
- `recommended_xp`: the recommended plan's XI points over the horizon,
  minus `hit_cost` for every hit in the plan (this week's and later weeks').
- `optimum_xp`: the best single squad affordable with the current squad's
  selling value plus bank, held for the horizon, transfers and hits ignored
  (a wildcard). Never reported below the recommended plan's gross points,
  so the plan cannot beat "the optimum".
- `team_rating` = 100 * current_xi_xp / optimum_xp, clipped to 0-100.

Chips (gain = extra projected points versus not playing the chip):
- Bench Boost at gw: xP of the recommended squad's four bench players.
- Triple Captain at gw: the captain's xP (the extra 1x).
- Free Hit at gw: best unconstrained squad for that gw alone, minus the
  plan's XI points that gw.
- Wildcard: `optimum_xp - recommended_xp`; only meaningful at the first
  horizon gameweek, so no other gameweek is listed.

Availability: FPL gives every chip once per half-season (gw1-19, gw20-38);
a chip is available at a gw if the entry's history shows no use of it in
that gw's half.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from squad.optimize import OptimizationResult, Player, optimize_squad, prune_pool
from squad.reconstruct import SquadState

FIRST_HALF_LAST_GW = 19
# Display order; the ids are FPL's own API chip names.
CHIP_ORDER = ("bboost", "3xc", "freehit", "wildcard")

# Free-transfer allowance large enough to be "unlimited" in a solve.
_UNLIMITED = 15
# Plain expected points: no decay, no value for banked transfers.
_PLAIN = {"decay": 1.0, "ft_value": 0.0, "max_banked": _UNLIMITED}


def _xp(projections: Mapping[int, Mapping[int, float]], gw: int, eid: int) -> float:
    return projections.get(gw, {}).get(eid, 0.0)


def xi_points(result: OptimizationResult, projections: Mapping[int, Mapping[int, float]], gw: int) -> float:
    xi = sum(_xp(projections, gw, e) for e in result.starting_xi[gw])
    return xi + _xp(projections, gw, result.captain[gw])


def total_hits(result: OptimizationResult) -> int:
    return result.hits_taken + sum(week.hits for week in result.plan)


def recommended_xp(
    result: OptimizationResult,
    projections: Mapping[int, Mapping[int, float]],
    horizon: Sequence[int],
    hit_cost: int,
) -> float:
    return sum(xi_points(result, projections, gw) for gw in horizon) - hit_cost * total_hits(result)


def squad_by_gw(result: OptimizationResult, horizon: Sequence[int]) -> dict[int, frozenset[int]]:
    """The recommended squad in each horizon week: this week's squad with
    each later planned week's transfers applied in order."""
    squad = set(result.squad)
    out = {horizon[0]: frozenset(squad)}
    plan = {week.gw: week for week in result.plan}
    for gw in horizon[1:]:
        week = plan.get(gw)
        if week is not None:
            squad -= week.transfers_out
            squad |= week.transfers_in
        out[gw] = frozenset(squad)
    return out


def _unconstrained(
    current: SquadState,
    pool: Sequence[Player],
    projections: Mapping[int, Mapping[int, float]],
    horizon: Sequence[int],
) -> OptimizationResult:
    return optimize_squad(
        current,
        prune_pool(current, pool, projections, horizon),
        projections,
        horizon=horizon,
        free_transfers=_UNLIMITED,
        hit_cost=0,
        plan_future_transfers=False,
        **_PLAIN,
    )


def team_rating(
    current: SquadState,
    pool: Sequence[Player],
    projections: Mapping[int, Mapping[int, float]],
    horizon: Sequence[int],
    result: OptimizationResult,
    hit_cost: int,
) -> dict[str, float]:
    held = optimize_squad(
        current,
        prune_pool(current, pool, projections, horizon),
        projections,
        horizon=horizon,
        free_transfers=0,
        max_transfers=0,
        plan_future_transfers=False,
        **_PLAIN,
    )
    current_xp = sum(xi_points(held, projections, gw) for gw in horizon)
    wildcard = _unconstrained(current, pool, projections, horizon)
    wildcard_xp = sum(xi_points(wildcard, projections, gw) for gw in horizon)
    gross = sum(xi_points(result, projections, gw) for gw in horizon)
    optimum = max(wildcard_xp, gross, current_xp)
    return {
        "current_xi_xp": current_xp,
        "recommended_xp": recommended_xp(result, projections, horizon, hit_cost),
        "optimum_xp": optimum,
        "team_rating": min(100.0, max(0.0, 100.0 * current_xp / optimum)) if optimum > 0 else 0.0,
    }


def chips_available(chips_used: Sequence[tuple[str, int]], chip: str, gw: int) -> bool:
    first_half = gw <= FIRST_HALF_LAST_GW
    return not any(name == chip and (event <= FIRST_HALF_LAST_GW) == first_half for name, event in chips_used)


def chip_comparison(
    current: SquadState,
    pool: Sequence[Player],
    projections: Mapping[int, Mapping[int, float]],
    horizon: Sequence[int],
    result: OptimizationResult,
    rating: Mapping[str, float],
    chips_used: Sequence[tuple[str, int]] = (),
) -> dict[str, object]:
    """`{"chips": [{chip, gw, gain, available}], "best_gw": {chip: gw}}`."""
    squads = squad_by_gw(result, horizon)
    rows: list[dict[str, object]] = []
    for gw in horizon:
        squad_xp = sum(_xp(projections, gw, e) for e in squads[gw])
        xi_xp = sum(_xp(projections, gw, e) for e in result.starting_xi[gw])
        free_hit = _unconstrained(current, pool, projections, [gw])
        gains = {
            "bboost": squad_xp - xi_xp,
            "3xc": _xp(projections, gw, result.captain[gw]),
            "freehit": xi_points(free_hit, projections, gw) - xi_points(result, projections, gw),
        }
        for chip, gain in gains.items():
            rows.append({"chip": chip, "gw": gw, "gain": gain})
    rows.append({"chip": "wildcard", "gw": horizon[0], "gain": rating["optimum_xp"] - rating["recommended_xp"]})
    for row in rows:
        row["available"] = chips_available(chips_used, str(row["chip"]), int(row["gw"]))  # type: ignore[call-overload]
    rows.sort(key=lambda r: (CHIP_ORDER.index(str(r["chip"])), int(r["gw"])))  # type: ignore[call-overload]
    best = {
        chip: int(max((r for r in rows if r["chip"] == chip), key=lambda r: float(r["gain"]))["gw"])  # type: ignore[arg-type,call-overload]
        for chip in CHIP_ORDER
    }
    return {"chips": rows, "best_gw": best}
