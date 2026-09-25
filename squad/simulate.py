"""Season simulation: an optimizer measured by the points its decisions
actually scored, not by the projections it maximized.

A projection model is judged by walk-forward error; an optimizer can't be,
because its output is a decision. So this plays a whole archived season:
from `start_gw` on, each week the event model projects the next `horizon`
gameweeks from games before the deadline only, the optimizer decides, and
the squad scores what those players really scored that week — with FPL's
autosubs (bench order, formation minimums, goalkeeper for goalkeeper), the
vice-captain doubled when the captain doesn't play, hits deducted, free
transfers banked, and players sold at FPL's selling price (half of any
rise kept). Every variant starts from the same squad and sees the same
projections, so the difference between two runs is the decisions alone.

    uv run python -m squad.simulate            # every archived season, old v new

Not modelled: chips, and price changes inside a gameweek (each week's
price is that week's `value`).
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import polars as pl

from analytics.carryover import previous_season
from analytics.evaluate import SEASON_SCORING_CONFIG, build_difficulty_table, season_fixtures, season_prior_history
from analytics.projections import project_points
from analytics.scoreline import scoreline_table, strengths_before
from analytics.scoring import load_scoring_config
from backtest.backfill import NORMALIZED_DIR
from squad.optimize import (
    MIN_DEF_IN_XI,
    MIN_FWD_IN_XI,
    MIN_MID_IN_XI,
    Player,
    optimize_squad,
    prune_pool,
)
from squad.reconstruct import SquadPlayer, SquadState
from squad.transfers import accrue_free_transfers

logger = logging.getLogger("squad.simulate")

STARTING_BANK = 1000  # £100.0m
DEFAULT_HORIZON = 3

# The optimizer this module was written to replace: every transfer made
# this week and the squad held after, the bench and vice-captain worth
# nothing, no decay, and a banked transfer worth nothing either.
SINGLE_DECISION = dict(
    plan_future_transfers=False, decay=1.0, ft_value=0.0, vice_weight=0.0, bench_weights=(0.0, 0.0, 0.0, 0.0)
)
# ...and its team sheet, which the solver didn't choose: the bench sorted by
# next week's projection afterwards, and the vice-captain squad/shadow.py's
# old rule gave it — the first other starter, goalkeeper first. With zero
# weights the solver's own bench and vice would be arbitrary, which would
# make the old optimizer look worse than it was.
LEGACY_TEAM_SHEET = "legacy_team_sheet"
VARIANTS: dict[str, dict[str, Any]] = {
    "single decision (old)": {**SINGLE_DECISION, LEGACY_TEAM_SHEET: True},
    "weekly plan (new)": {},
}


def selling_price(purchase: int, now: int) -> int:
    """FPL keeps half of any rise, rounded down, and all of any fall."""
    return purchase + (now - purchase) // 2 if now > purchase else now


@dataclass
class SeasonData:
    season: str
    rows: pl.DataFrame
    config: dict[str, Any]
    difficulty: pl.DataFrame
    fixtures: pl.DataFrame
    prev_rows: pl.DataFrame | None
    prior: pl.DataFrame | None
    _projection_cache: dict[tuple[int, int], dict[int, float]] = field(default_factory=dict)

    @classmethod
    def load(cls, season: str) -> "SeasonData":
        prev_path = NORMALIZED_DIR / f"{previous_season(season)}.parquet"
        return cls(
            season=season,
            rows=pl.read_parquet(NORMALIZED_DIR / f"{season}.parquet").filter(pl.col("position") != "AM"),
            config=load_scoring_config(Path(SEASON_SCORING_CONFIG[season])),
            difficulty=build_difficulty_table(season),
            fixtures=season_fixtures(season),
            prev_rows=pl.read_parquet(prev_path) if prev_path.exists() else None,
            prior=season_prior_history(season),
        )

    @property
    def last_gw(self) -> int:
        return int(self.rows["gw"].max())

    def known_players(self, before_gw: int) -> pl.DataFrame:
        """Everyone seen before the deadline, at their latest club and position."""
        return (
            self.rows.filter(pl.col("gw") < before_gw)
            .sort("gw")
            .group_by("element_id")
            .agg(pl.col("position").last(), pl.col("team").last(), pl.col("is_promoted_club").last(), pl.col("value").last())
        )

    def prices(self, gw: int) -> dict[int, int]:
        """Each player's price at gw's deadline: that week's value, else his latest."""
        latest = self.rows.filter(pl.col("gw") <= gw).sort("gw").group_by("element_id").agg(pl.col("value").last())
        return dict(zip(latest["element_id"].to_list(), latest["value"].to_list()))

    def projections(self, decided_at: int, target_gw: int) -> dict[int, float]:
        """The event model's projection for target_gw as it stood before
        decided_at's deadline. Cached: every variant sees the same numbers."""
        key = (decided_at, target_gw)
        if key not in self._projection_cache:
            train = self.rows.filter(pl.col("gw") < decided_at)
            roster = self.known_players(decided_at).select("element_id", "position", "team", "is_promoted_club")
            strengths = strengths_before(train, self.fixtures, self.prev_rows, decided_at)
            df = project_points(
                train, roster, target_gw, self.config, self.difficulty,
                prior_history=self.prior, scoreline=scoreline_table(self.fixtures, strengths, target_gw),
            )
            self._projection_cache[key] = dict(zip(df["element_id"].to_list(), df["prediction"].to_list()))
        return self._projection_cache[key]

    def actuals(self, gw: int) -> dict[int, tuple[int, int]]:
        """element_id -> (points, minutes) in gw; absent means no game."""
        week = self.rows.filter(pl.col("gw") == gw).group_by("element_id").agg(
            pl.col("total_points").sum(), pl.col("minutes").sum()
        )
        return {r["element_id"]: (r["total_points"], r["minutes"]) for r in week.to_dicts()}


def score_week(
    xi: frozenset[int],
    bench_order: tuple[int, ...],
    captain: int,
    vice: int | None,
    positions: dict[int, str],
    actual: dict[int, tuple[int, int]],
) -> int:
    """What the team sheet really scored, after FPL's automatic substitutions."""
    played = {e for e in set(xi) | set(bench_order) if actual.get(e, (0, 0))[1] > 0}
    lineup = set(xi)
    gk_sub, *outfield_subs = bench_order
    for starter in sorted(xi, key=lambda e: positions[e] != "GK"):
        if starter in played:
            continue
        if positions[starter] == "GK":
            if gk_sub in played:
                lineup.remove(starter)
                lineup.add(gk_sub)
            continue
        for sub in outfield_subs:
            if sub in lineup or sub not in played:
                continue
            trial = (lineup - {starter}) | {sub}
            counts = {pos: sum(positions[e] == pos for e in trial) for pos in ("DEF", "MID", "FWD")}
            if counts["DEF"] >= MIN_DEF_IN_XI and counts["MID"] >= MIN_MID_IN_XI and counts["FWD"] >= MIN_FWD_IN_XI:
                lineup = trial
                break
    points = sum(actual.get(e, (0, 0))[0] for e in lineup)
    armband = captain if captain in played else (vice if vice is not None and vice in played else None)
    if armband is not None and armband in lineup:
        points += actual[armband][0]
    return points


@dataclass
class SeasonResult:
    variant: str
    season: str
    points: int
    hits: int
    transfers: int
    by_gw: list[int]


def initial_squad(data: SeasonData, start_gw: int, horizon: int) -> SquadState:
    """The best £100m squad for the opening weeks, by the same projections
    — one squad every variant then starts from."""
    prices = data.prices(start_gw)
    known = data.known_players(start_gw)
    pool = [
        Player(element_id=r["element_id"], position=r["position"], club=r["team"], now_cost=prices[r["element_id"]])
        for r in known.to_dicts()
        if r["element_id"] in prices
    ]
    gws = [g for g in range(start_gw, start_gw + horizon) if g <= data.last_gw]
    projections = {g: data.projections(start_gw, g) for g in gws}
    empty = SquadState(as_of=datetime.now(timezone.utc), players=(), bank=STARTING_BANK)
    result = optimize_squad(
        empty, prune_pool(empty, pool, projections, gws), projections, horizon=gws,
        free_transfers=15, max_banked=15, plan_future_transfers=False,
    )
    return SquadState(
        as_of=datetime.now(timezone.utc),
        players=tuple(
            SquadPlayer(element_id=e, purchase_price=prices[e], selling_price=prices[e], squad_position=i + 1,
                        multiplier=1, is_captain=False, is_vice_captain=False)
            for i, e in enumerate(sorted(result.squad))
        ),
        bank=result.bank_after,
    )


def simulate(
    data: SeasonData,
    variant: str,
    options: dict[str, Any],
    squad: SquadState,
    start_gw: int = 2,
    horizon: int = DEFAULT_HORIZON,
) -> SeasonResult:
    max_banked = data.config["free_transfers"]["max_banked"]
    options = dict(options)
    legacy_sheet = options.pop(LEGACY_TEAM_SHEET, False)
    free_transfers = 1
    purchase = {sp.element_id: sp.purchase_price for sp in squad.players}
    total, hits_total, transfers_total, by_gw = 0, 0, 0, []

    for gw in range(start_gw, data.last_gw + 1):
        prices = data.prices(gw)
        known = data.known_players(gw)
        positions = dict(zip(known["element_id"].to_list(), known["position"].to_list()))
        state = SquadState(
            as_of=squad.as_of,
            players=tuple(
                SquadPlayer(element_id=sp.element_id, purchase_price=purchase[sp.element_id],
                            selling_price=selling_price(purchase[sp.element_id], prices.get(sp.element_id, purchase[sp.element_id])),
                            squad_position=sp.squad_position, multiplier=1, is_captain=False, is_vice_captain=False)
                for sp in squad.players
            ),
            bank=squad.bank,
        )
        owned = {sp.element_id for sp in state.players}
        pool = [
            Player(element_id=e, position=positions[e], club=r_team, now_cost=prices.get(e, purchase.get(e, 0)))
            for e, r_team in zip(known["element_id"].to_list(), known["team"].to_list())
            if e in prices or e in owned
        ]
        gws = [g for g in range(gw, gw + horizon) if g <= data.last_gw]
        projections = {g: data.projections(gw, g) for g in gws}
        result = optimize_squad(
            state, prune_pool(state, pool, projections, gws), projections, horizon=gws,
            free_transfers=free_transfers, max_banked=max_banked, **options,
        )

        bench_order, vice = result.bench_order, result.vice_captain.get(gw)
        if legacy_sheet:
            gk_sub, *outfield = bench_order
            bench_order = (gk_sub, *sorted(outfield, key=lambda e: -projections[gw].get(e, 0.0)))
            xi_sorted = sorted(result.starting_xi[gw], key=lambda e: (positions[e] != "GK", e))
            vice = next(e for e in xi_sorted if e != result.captain[gw])
        points = score_week(
            result.starting_xi[gw], bench_order, result.captain[gw], vice, positions, data.actuals(gw),
        ) - 4 * result.hits_taken
        total += points
        by_gw.append(points)
        hits_total += result.hits_taken
        transfers_total += len(result.transfers_in)

        for e in result.transfers_in:
            purchase[e] = prices[e]
        free_transfers = accrue_free_transfers(free_transfers, len(result.transfers_in), max_banked)
        squad = SquadState(
            as_of=squad.as_of,
            players=tuple(
                SquadPlayer(element_id=e, purchase_price=purchase[e], selling_price=purchase[e], squad_position=i + 1,
                            multiplier=1, is_captain=False, is_vice_captain=False)
                for i, e in enumerate(sorted(result.squad))
            ),
            bank=result.bank_after,
        )
        logger.info("%s %s gw%d: %d pts (%d transfers, %d hits)", data.season, variant, gw, points,
                    len(result.transfers_in), result.hits_taken)

    return SeasonResult(variant, data.season, total, hits_total, transfers_total, by_gw)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--seasons", default=",".join(SEASON_SCORING_CONFIG))
    parser.add_argument("--start-gw", type=int, default=2)
    parser.add_argument("--horizon", type=int, default=DEFAULT_HORIZON)
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    for season in args.seasons.split(","):
        data = SeasonData.load(season)
        start = initial_squad(data, args.start_gw, args.horizon)
        for variant, options in VARIANTS.items():
            r = simulate(data, variant, options, start, args.start_gw, args.horizon)
            print(f"{season}  {variant:24s} {r.points:5d} pts  {r.transfers:3d} transfers  {r.hits:2d} hits")


if __name__ == "__main__":
    main()
