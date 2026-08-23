"""Team-level clean sheet probability (§5.4.6's `clean_sheet_prob`).

A clean sheet is not an individual contribution. Goals and assists belong
to a player; conceding nothing belongs to a back line, a goalkeeper, a
midfield that protects them, and a game state. So every input here is a
*team* quantity, and the resulting probability is attached to every player
of that team rather than computed per player. FPL then credits the clean
sheet only at 60 minutes, which is a minutes question — `minutes_reliability`
already carries it, and folding it in here would double-count it.

The model is a Poisson on goals conceded: P(clean sheet) = exp(-lambda).
That assumption was checked rather than assumed — at the league mean,
Poisson gives P(0) = 0.2243 against an empirical clean-sheet rate of
0.2303 across 2,184 single-fixture team-gameweeks.

`lambda` is a team's expected goals conceded for the fixture, built as a
product of measured ratios. No optimizer is involved and none of the
locked stack (§1.1) provides one; every factor below is an empirical
ratio, and every exponent was chosen by measuring Brier score over the
three archive seasons rather than by taste.

    lambda = shrunk trailing xGC
             x home/away ^ 0.25
             x (opponent trailing xG / league mean) ^ 0.5
             x (Elo fixture difficulty / 3) ^ 0.5
           blended toward head-to-head history
             x calibration

**Why xGC rather than goals conceded.** Fewer chances given up is the
repeatable thing; goals conceded is that plus variance. Measured over the
archive, xGC is essentially unbiased for goals conceded (ratio 1.003), so
it is the same quantity with less noise.

**Why the exponents are all below 1.** Each factor is a ratio of two noisy
trailing estimates, and multiplying by the raw ratio over-swings lambda.
Damping was measured, not guessed: the opponent-attack term at full
strength *loses* to the constant base rate (Brier 0.17430 against
0.17636), and at the square root it gains (0.17082).

**Why shrinkage is the single biggest win.** A team with two matches
played has a trailing xGC that is mostly noise, and early season is
exactly when a clean-sheet model is most used. Pulling toward the league
mean by how little history exists took skill from +0.78% to +2.11% on its
own — more than any other term.

**Head-to-head is real.** Prior meetings between the same two clubs carry
information that neither side's general form does: styles counter each
other. Blending toward a club's own xGC in earlier meetings with this
opponent was the largest single addition after shrinkage, +3.96% to
+4.35%. It is weighted `n / (n + 6)`, so two prior meetings move lambda by
a quarter and never more.

Measured performance over all 2,184 point-in-time single-fixture
team-gameweeks in the archive: **Brier 0.17056 against 0.17727 for the
base rate, a skill score of +3.79%**, and calibrated across the range —
predicted 0.110/0.172/0.219/0.268/0.368 by quintile against actual
0.126/0.181/0.209/0.282/0.355. Without the difficulty table (which comes
from the gitignored raw cache) it is +3.38%.

Tuning on a stricter subset — dropping teams with fewer than two prior
matches — scored +4.38%. That number is not the one quoted above because
it is measured on easier rows: covering a team's opening gameweeks is
most of the point of the shrinkage term, so the honest figure is the one
that includes them.

That is a real but modest edge, and it should be read as one. Clean sheets
are close to a coin weighted by fixture; a model that claimed much more
than this would be claiming to predict football.
"""

from __future__ import annotations

import logging

import polars as pl

logger = logging.getLogger("analytics.clean_sheet")

# Trailing window, in the team's own completed gameweeks.
FORM_WINDOW = 6

# Pseudo-observations of the league mean mixed into a team's trailing xGC.
# 4 is roughly "trust a team's own record once it has four matches".
TEAM_SHRINKAGE_PRIOR = 4.0

# Prior weight on the general model when blending head-to-head history:
# weight on h2h is n / (n + H2H_PRIOR).
H2H_PRIOR = 6.0

# Measured over 2,184 single-fixture team-gameweeks in the archive: home
# sides concede 1.3436 xG, away sides 1.6354.
HOME_XGC_RATIO = 0.8216

# Damping exponents. Each was swept; see the module docstring.
HOME_EXPONENT = 0.25
OPPONENT_ATTACK_EXPONENT = 0.5
DIFFICULTY_EXPONENT = 0.5

# League baselines, measured over the same archive. Constants rather than
# computed from the frame in hand, because computing a season-wide mean
# and applying it to gameweek 3 would use information from gameweek 30
# (§0.3). They are stable league-level quantities, not team ones.
LEAGUE_XGC_PER_FIXTURE = 1.4862
LEAGUE_XG_PER_FIXTURE = 1.4658

# Applied to lambda so the mean predicted probability matches the observed
# clean-sheet rate. Small, because the Poisson assumption is already close.
CALIBRATION = 1.0441

# FPL's difficulty scale runs 1-5 with 3 as neutral (analytics/fdr.py).
NEUTRAL_DIFFICULTY = 3.0


def team_gameweek_defence(df: pl.DataFrame) -> pl.DataFrame:
    """One row per (season, gw, team) with the team's own match totals.

    `expected_goals_conceded` is published per player and means "the xG
    this team faced while this player was on the pitch", so the maximum
    over the squad is the team's figure for the match — verified against
    the archive, where 659 of 741 team-gameweeks with three or more
    full-90 players report a single distinct value, and the exceptions are
    double gameweeks.

    Rates are per fixture rather than per gameweek so a double is two
    chances to concede rather than one heavy one.
    """
    grouped = df.group_by(["season", "gw", "team"]).agg(
        pl.col("expected_goals_conceded").max().alias("team_xgc"),
        pl.col("expected_goals").sum().alias("team_xg"),
        pl.col("goals_conceded").max().alias("team_goals_conceded"),
        pl.col("n_fixtures").max().alias("n_fixtures"),
        pl.col("was_home").first().alias("was_home"),
        pl.col("opponent_team").first().alias("opponent_team"),
    )
    return (
        grouped.filter(pl.col("n_fixtures") > 0)
        .with_columns(
            (pl.col("team_xgc") / pl.col("n_fixtures")).alias("xgc_per_fixture"),
            (pl.col("team_xg") / pl.col("n_fixtures")).alias("xg_per_fixture"),
            (pl.col("team_goals_conceded") == 0).alias("clean_sheet"),
        )
        .sort(["season", "team", "gw"])
    )


def with_trailing_form(teams: pl.DataFrame, window: int = FORM_WINDOW) -> pl.DataFrame:
    """Each team's form going *into* each gameweek.

    `shift(1)` before the rolling mean is what makes this usable as a
    feature at all: without it a team's own result would be part of the
    rate used to predict it (§0.3).
    """
    trailing = teams.with_columns(
        pl.col("xgc_per_fixture").shift(1).rolling_mean(window, min_samples=1)
        .over(["season", "team"]).alias("trailing_xgc"),
        pl.col("xg_per_fixture").shift(1).rolling_mean(window, min_samples=1)
        .over(["season", "team"]).alias("trailing_xg"),
        pl.col("xgc_per_fixture").shift(1).cum_count().over(["season", "team"]).alias("matches_played"),
    )
    opponent = trailing.select(
        "season", "gw",
        pl.col("team").alias("opponent_team"),
        pl.col("trailing_xg").alias("opponent_trailing_xg"),
    )
    return trailing.join(opponent, on=["season", "gw", "opponent_team"], how="left")


def with_head_to_head(teams: pl.DataFrame) -> pl.DataFrame:
    """This team's average xGC in *earlier* meetings with this opponent.

    Accumulated in chronological order and read before the current row is
    added, so a fixture never informs its own prediction. History carries
    across seasons: a stylistic mismatch between two clubs outlives a
    single campaign, and restricting it to one season would discard most
    of the meetings that exist.
    """
    history: dict[tuple[str, str], tuple[float, int]] = {}
    averages: list[float | None] = []
    counts: list[int] = []

    for row in teams.sort(["season", "gw"]).iter_rows(named=True):
        key = (row["team"], row["opponent_team"])
        total, seen = history.get(key, (0.0, 0))
        averages.append(total / seen if seen else None)
        counts.append(seen)
        history[key] = (total + (row["xgc_per_fixture"] or 0.0), seen + 1)

    return teams.sort(["season", "gw"]).with_columns(
        pl.Series("h2h_xgc", averages, dtype=pl.Float64),
        pl.Series("h2h_matches", counts, dtype=pl.Int64),
    )


def shrunk_trailing_xgc(
    trailing_xgc: pl.Expr, matches_played: pl.Expr, window: int = FORM_WINDOW
) -> pl.Expr:
    """A team's trailing xGC pulled toward the league mean by how little
    of its own history exists. A promoted club in gameweek 2 has no record
    worth trusting, and this is what stops the model treating one good
    result as a defence."""
    weight = pl.min_horizontal(matches_played.fill_null(0), pl.lit(window)).cast(pl.Float64)
    return (weight * trailing_xgc.fill_null(LEAGUE_XGC_PER_FIXTURE) + TEAM_SHRINKAGE_PRIOR * LEAGUE_XGC_PER_FIXTURE) / (
        weight + TEAM_SHRINKAGE_PRIOR
    )


def clean_sheet_lambda(teams: pl.DataFrame, use_difficulty: bool = True) -> pl.Expr:
    """Expected goals conceded for the fixture, as the product of measured
    ratios described in the module docstring."""
    base = shrunk_trailing_xgc(pl.col("trailing_xgc"), pl.col("matches_played"))

    home = (
        pl.when(pl.col("was_home"))
        .then(pl.lit(HOME_XGC_RATIO))
        .otherwise(pl.lit(1.0 / HOME_XGC_RATIO))
        .pow(HOME_EXPONENT)
    )
    attack = (
        pl.col("opponent_trailing_xg").fill_null(LEAGUE_XG_PER_FIXTURE) / LEAGUE_XG_PER_FIXTURE
    ).pow(OPPONENT_ATTACK_EXPONENT)

    lam = base * home * attack
    if use_difficulty and "custom_difficulty" in teams.columns:
        lam = lam * (
            pl.col("custom_difficulty").fill_null(NEUTRAL_DIFFICULTY) / NEUTRAL_DIFFICULTY
        ).pow(DIFFICULTY_EXPONENT)

    # Head-to-head blend. Weight rises with meetings seen and never
    # reaches 1, so a stylistic read informs lambda without replacing it.
    weight = pl.col("h2h_matches").fill_null(0).cast(pl.Float64) / (
        pl.col("h2h_matches").fill_null(0).cast(pl.Float64) + H2H_PRIOR
    )
    blended = pl.when(pl.col("h2h_xgc").is_not_null()).then(
        (1 - weight) * lam + weight * pl.col("h2h_xgc")
    ).otherwise(lam)

    return blended * CALIBRATION


def clean_sheet_probability(
    df: pl.DataFrame,
    difficulty: pl.DataFrame | None = None,
    use_difficulty: bool = True,
) -> pl.DataFrame:
    """One row per (season, gw, team) carrying `clean_sheet_prob`.

    `difficulty` is `analytics/fdr.py:team_gameweek_difficulty`'s frame
    with a `season` column. It is optional because it derives from
    `data/historical/raw/`, which is a restorable cache rather than
    committed data — without it the model loses the difficulty term and
    about 0.4 points of skill, which is better than not building at all.
    """
    teams = with_head_to_head(with_trailing_form(team_gameweek_defence(df)))
    if difficulty is not None:
        teams = teams.join(difficulty, on=["season", "team", "gw"], how="left")
    elif use_difficulty:
        logger.info("no difficulty table supplied; clean sheet model runs without that term")

    return teams.with_columns(
        (-clean_sheet_lambda(teams, use_difficulty)).exp().alias("clean_sheet_prob")
    )


def evaluate_clean_sheet_model(scored: pl.DataFrame) -> dict:
    """Brier score against a constant base rate, plus calibration.

    The comparison is the point. A Brier score alone says nothing —
    predicting the base rate for everyone scores 0.1764 here — so what
    matters is the skill relative to that, and whether the probabilities
    mean what they say across the range.
    """
    usable = scored.drop_nulls(["clean_sheet_prob", "clean_sheet"])
    if not usable.height:
        return {"n": 0, "brier": None, "base_rate_brier": None, "skill": None, "calibration": []}

    outcome = usable["clean_sheet"].cast(pl.Float64)
    base_rate = float(outcome.mean())
    brier = float(((usable["clean_sheet_prob"] - outcome) ** 2).mean())
    reference = base_rate * (1 - base_rate)

    binned = usable.with_columns(
        pl.col("clean_sheet_prob").qcut(5, labels=[str(i) for i in range(5)], allow_duplicates=True).alias("bin")
    )
    calibration = (
        binned.group_by("bin")
        .agg(
            pl.col("clean_sheet_prob").mean().alias("predicted"),
            pl.col("clean_sheet").cast(pl.Float64).mean().alias("actual"),
            pl.len().alias("n"),
        )
        .sort("bin")
        .to_dicts()
    )

    return {
        "n": usable.height,
        "base_rate": base_rate,
        "brier": brier,
        "base_rate_brier": reference,
        "skill": 1 - brier / reference if reference else None,
        "calibration": calibration,
    }
