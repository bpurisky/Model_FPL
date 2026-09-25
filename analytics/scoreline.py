"""Team scoreline model: each fixture's expected goals for and against,
from team attack and defence strengths measured in xG.

The event model used to adjust for the fixture with one number, the Elo
difficulty on a 1-5 scale, scaling goals/assists/clean sheets up and goals
conceded/saves down by a flat 7.5% a step. A clean sheet is not linear in
anything, and an Elo gap says nothing about whether a side is strong
because it scores or because it doesn't concede. Here each team carries an
attack rate (xG for per match) and a defence rate (xG against per match),
and a fixture's expected goals are

    lambda_for(T v O) = attack(T) * defence(O) / league_average * home

which is the multiplicative Poisson form of Dixon & Coles (1997) that
AIrsenal also starts from, fitted by shrunk means rather than by
likelihood. Clean sheets and goals conceded then come from Poisson
probabilities, and goals, assists and saves from how far this fixture's
lambda sits from the team's own norm.

Double and blank gameweeks fall out of the same table: every quantity is a
sum over the team's fixtures that gameweek, and a team with none has
`n_fixtures` 0.

Point-in-time: strengths for gameweek g use only gameweeks before g (and
last season, which had finished), and fixtures are known in advance.
"""

from __future__ import annotations

import math

import polars as pl

# Home sides make about 13% more xG than away sides: 1.527 v 1.355 per match
# over 2024-25's single-fixture gameweeks. Split evenly either side of the
# neutral rate, so it moves both teams' expectations and not only one.
HOME_FACTOR = math.sqrt(1.527 / 1.355)

# Matches' worth of weight on the prior (last season's rate, or the
# relegated sides' for a promoted club) when measuring a team this season.
# The walk-forward barely distinguishes 4, 8 and 15 (RMSE 1.9554 / 1.9544 /
# 1.9540 over 2024-25 and 2025-26); 8 is the middle.
PRIOR_MATCHES = 8

# League xG per team per match, for a season with neither history nor a
# previous season to measure it from.
FALLBACK_RATE = 1.44


def team_match_rates(season_df: pl.DataFrame, fixtures: pl.DataFrame | None = None) -> pl.DataFrame:
    """team, gw, matches, xg_for, xg_against — one row per team-gameweek
    it played in, summed over a double gameweek's fixtures.

    `fixtures`: gw, team_h, team_a, for every fixture that has been
    played; `matches` is counted from it, so a gameweek's rows need no
    fixture count of their own (the live store carries none). Without it,
    `matches` is the archive's own per-row `n_fixtures`. xG for is
    the sum of the side's players' xG. xG against is the largest
    `expected_goals_conceded` any of its players carried, which is the xG
    conceded by whoever was on the pitch throughout (FPL counts xGC only
    while a player is on), so the whole match's worth.
    """
    if fixtures is None:
        matches = (
            season_df.filter(pl.col("n_fixtures") > 0)
            .group_by("team", "gw")
            .agg(pl.col("n_fixtures").max().cast(pl.Int64).alias("matches"))
        )
    else:
        matches = (
            pl.concat([fixtures.select("gw", pl.col("team_h").alias("team")), fixtures.select("gw", pl.col("team_a").alias("team"))])
            .group_by("team", "gw")
            .agg(pl.len().cast(pl.Int64).alias("matches"))
        )
    xg = season_df.group_by("team", "gw").agg(
        pl.col("expected_goals").sum().alias("xg_for"),
        pl.col("expected_goals_conceded").max().alias("xg_against"),
    )
    return matches.join(xg, on=["team", "gw"], how="inner")


def team_priors(prev_rates: pl.DataFrame | None, current_teams: list[str]) -> pl.DataFrame:
    """team, prior_for, prior_against per match for this season's teams.

    A team that was in the league last season starts from its own rates.
    A promoted club starts from the average of the sides it replaced — the
    ones in last season's table that are not in this one — which is the
    best available guess at a newly promoted side's level. With no
    previous season at all, every team starts at FALLBACK_RATE.
    """
    if prev_rates is None or prev_rates.height == 0:
        return pl.DataFrame({"team": current_teams, "prior_for": FALLBACK_RATE, "prior_against": FALLBACK_RATE})
    per_team = prev_rates.group_by("team").agg(
        (pl.col("xg_for").sum() / pl.col("matches").sum()).alias("prior_for"),
        (pl.col("xg_against").sum() / pl.col("matches").sum()).alias("prior_against"),
    )
    replaced = per_team.filter(~pl.col("team").is_in(current_teams))
    newcomer_for = replaced["prior_for"].mean() if replaced.height else per_team["prior_for"].mean()
    newcomer_against = replaced["prior_against"].mean() if replaced.height else per_team["prior_against"].mean()
    return (
        pl.DataFrame({"team": current_teams})
        .join(per_team, on="team", how="left")
        .with_columns(
            pl.col("prior_for").fill_null(newcomer_for),
            pl.col("prior_against").fill_null(newcomer_against),
        )
    )


def team_strengths(rates: pl.DataFrame, priors: pl.DataFrame, before_gw: int, prior_matches: float = PRIOR_MATCHES) -> pl.DataFrame:
    """team, attack, defence: xG for and against per match from this
    season's gameweeks before `before_gw`, shrunk toward the prior by
    `prior_matches` matches' worth of weight."""
    so_far = rates.filter(pl.col("gw") < before_gw).group_by("team").agg(
        pl.col("matches").sum(), pl.col("xg_for").sum(), pl.col("xg_against").sum()
    )
    return (
        priors.join(so_far, on="team", how="left")
        .with_columns(pl.col("matches", "xg_for", "xg_against").fill_null(0))
        .select(
            "team",
            ((pl.col("xg_for") + prior_matches * pl.col("prior_for")) / (pl.col("matches") + prior_matches)).alias("attack"),
            ((pl.col("xg_against") + prior_matches * pl.col("prior_against")) / (pl.col("matches") + prior_matches)).alias("defence"),
        )
    )


def _expected_floor_half(lam: float, per: int = 2) -> float:
    """E[floor(K / per)] for K ~ Poisson(lam): the expected number of
    goals-conceded deductions."""
    total, p = 0.0, math.exp(-lam)
    for k in range(0, 30):
        if k:
            p *= lam / k
        total += (k // per) * p
    return total


def scoreline_table(fixtures: pl.DataFrame, strengths: pl.DataFrame, gw: int, per: int = 2) -> pl.DataFrame:
    """team, n_fixtures, attack_mult, defence_mult, expected_cs,
    expected_gc_deductions for gameweek `gw`.

    `fixtures`: gw, team_h, team_a (team names). `strengths`: team_strengths.

    attack_mult is the sum over the team's fixtures of lambda_for over its
    own attack rate — so 1.0 is one ordinary match, 2.2 a double gameweek
    of kind ones. defence_mult is the same for lambda_against and its
    defence rate. expected_cs sums P(0 conceded), expected_gc_deductions
    E[floor(conceded / per)]. Teams without a fixture are absent; the
    caller reads that as n_fixtures 0.
    """
    league = strengths["attack"].mean()
    s = {r["team"]: r for r in strengths.to_dicts()}
    out: dict[str, dict[str, float]] = {}
    for f in fixtures.filter(pl.col("gw") == gw).to_dicts():
        home, away = f["team_h"], f["team_a"]
        if home not in s or away not in s:
            continue
        lam_home = s[home]["attack"] * s[away]["defence"] / league * HOME_FACTOR
        lam_away = s[away]["attack"] * s[home]["defence"] / league / HOME_FACTOR
        for team, lam_for, lam_against in ((home, lam_home, lam_away), (away, lam_away, lam_home)):
            row = out.setdefault(team, {"n_fixtures": 0, "attack_mult": 0.0, "defence_mult": 0.0,
                                        "expected_cs": 0.0, "expected_gc_deductions": 0.0})
            row["n_fixtures"] += 1
            row["attack_mult"] += lam_for / s[team]["attack"]
            row["defence_mult"] += lam_against / s[team]["defence"]
            row["expected_cs"] += math.exp(-lam_against)
            row["expected_gc_deductions"] += _expected_floor_half(lam_against, per)
    schema = {"team": pl.Utf8, "n_fixtures": pl.Int64, "attack_mult": pl.Float64, "defence_mult": pl.Float64,
              "expected_cs": pl.Float64, "expected_gc_deductions": pl.Float64}
    return pl.DataFrame([{"team": t, **v} for t, v in out.items()], schema=schema)


def strengths_before(
    season_df: pl.DataFrame,
    fixtures: pl.DataFrame,
    prev_season_df: pl.DataFrame | None,
    before_gw: int,
) -> pl.DataFrame:
    """team_strengths for this season's teams as they stood before
    `before_gw`: the one call a caller with a season's rows and fixtures
    needs. `fixtures` may include unplayed fixtures; only those in
    gameweeks before `before_gw` are counted as played. `prev_season_df`
    is last season's archived rows (they carry `n_fixtures`), or None."""
    teams = sorted(set(fixtures["team_h"].to_list()) | set(fixtures["team_a"].to_list()))
    prev_rates = team_match_rates(prev_season_df) if prev_season_df is not None else None
    rates = team_match_rates(season_df.filter(pl.col("gw") < before_gw), fixtures.filter(pl.col("gw") < before_gw))
    return team_strengths(rates, team_priors(prev_rates, teams), before_gw)
