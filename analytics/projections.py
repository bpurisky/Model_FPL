"""Event-level model (§4.2): separate heads for appearance/minutes, goals,
assists, clean sheets, defensive contribution, saves, and bonus, combined
through analytics.scoring.compute_points-equivalent arithmetic into a
points projection.

Every head is a trailing-rate estimator — the same statistical philosophy
established in backtest/baselines.py (§3.4), generalized per-component via
analytics/features.py, and difficulty-adjusted using the custom Elo-based
FDR (analytics/fdr.py) rather than FPL's own static rating. This is a
deliberately simple, transparent statistical model, not machine learning —
scikit-learn/xgboost aren't in the locked stack (§1.1) and this phase's bar
is beating three naive baselines (§4.4), not state-of-the-art accuracy.

Difficulty adjustment is applied only where the direction is unambiguous:
favorable for the attacking side on goals/assists/clean-sheets, adverse for
the defending side on goals conceded and GK saves (more shots faced against
a stronger attack). Defensive contributions and bonus are left as plain
trailing rates — a difficulty story for those is weaker and not worth the
extra unproven assumption.

Minutes get their own three-way P(blank)/P(short)/P(60+) distribution
(analytics.features.minutes_distribution) rather than a single mean-minutes
figure thresholded after the fact — see that function's docstring for why
(Jensen's inequality on a step function).

Every other head is a rate *per appearance*, multiplied by this week's
P(plays) from the minutes head. Averaging per gameweek instead folds
availability into ability: a starter back from a month out ranks as a
quarter of himself, and last season's games (analytics.carryover) would
bring last season's role with them — his old club, his end-of-season
rotation. Split, availability comes only from the recent minutes window
and ability from the long one, which is what lets the ability window reach
back into last season at all. Goals and assists trail xG and xA rather
than realized counts. Walk-forward, 2024-25 and 2025-26 pooled, against the
per-gameweek, realized-count, five-game model this replaced: MAE 1.0506 ->
0.9938, RMSE 2.1185 -> 1.9844, within-position Spearman 0.7287 -> 0.7515
(per gameweek and position, averaged); 2023-24, which has no previous
season to carry, improves on all three as well.
"""

from __future__ import annotations

from typing import Any, Callable

import polars as pl

from analytics import features

# Games of per-appearance history behind every scoring rate, reaching back
# into last season when this one is short. The walk-forward flattens out
# between 10 and 20 (MAE 0.9967 / 0.9938 / 0.9931, Spearman 0.7510 /
# 0.7515 / 0.7517); 15 is the middle of it, and the best of the three on
# 5+ point returns. The minutes window stays short — see
# project_event_vectors.
DEFAULT_WINDOW = 15
DEFAULT_MINUTES_WINDOW = 3
DIFFICULTY_SCALE_STEP = 0.075  # matches backtest.baselines' constant

# Goals conceded is shared across an entire back line and heavily
# influenced by single-match variance (a fluke 3-0), so an individual
# defender's trailing mean of it carries real signal but also a lot of
# noise relative to a cleaner binary stat like clean_sheets. Confirmed by
# ablation, not assumed: at full weight the goals-conceded term passes the
# MAE bar (§4.4) but pulls within-position Spearman below
# fixture_adjusted_trailing_mean's, entirely concentrated in DEF (0.590 vs
# 0.606); at zero weight Spearman comfortably clears the bar but MAE no
# longer beats the baselines. 0.6-0.85 is a wide, robust plateau clearing
# both bars with margin; this is the middle of it, not a fitted edge.
#
# The sweep behind that sentence is now kept rather than only asserted:
# `web/export/shrinkage.py` re-runs the walk-forward across 0.0-1.0 and
# `shrinkage.json` records it. Reading it back sharpens one thing the
# paragraph above leaves ambiguous -- *which* within-position Spearman.
# On §4.4's criterion as stated, the unweighted mean across positions,
# the range clearing both bars is 0.5-0.9, and the 0.6-0.85 claim holds
# comfortably. On the stricter reading this comment itself uses, DEF
# alone, it is only 0.5-0.6 -- and 0.7 sits just past it, at DEF rho
# 0.6028 against the baseline's 0.6055.
#
# Both are true of different measurements. 0.7 stays: the acceptance
# criterion is the stated one, and the DEF gap at 0.7 is 0.003 against a
# 0.028 MAE gain. But the trade is real, monotone in both directions, and
# now on screen rather than in this paragraph.
#
# (The figures above are the per-gameweek, realized-goals, five-game model.
# On the per-appearance xG model with carryover every point of the 0.0-1.0
# sweep clears all three bars, DEF alone included — 0.6359 at 0.7 against
# the baseline's 0.6055 — so the stricter reading no longer disagrees. The
# trade itself remains: MAE still falls and Spearman still falls as the
# weight rises. shrinkage.json has the current numbers.)
#
# Every caller that does not name a shrinkage gets this one, so the
# published model is exactly the model these numbers were measured on.
# The parameter exists for one reason: §5.4.7 wants the plateau *on
# screen* rather than only in this comment, and `web/export/shrinkage.py`
# re-runs the walk-forward across the range to draw it. A sweep that
# edited the module constant would be a sweep that changed the model for
# everyone who imported it mid-run.
#
# Superseded by the scoreline model (analytics.scoreline). Goals conceded
# is now a Poisson expectation from team strengths, not a defender's own
# trailing mean, so the noise the shrinkage damped is no longer there, and
# the weight is 1.0: the expectation itself. The walk-forward has no
# optimum to offer in its place: MAE keeps falling past 1.0 (0.9825 at
# 1.0, 0.9749 at 1.3) only because a heavier deduction drags every
# projection toward the skewed distribution's median, while 5+ point
# returns get worse (RMSE 5.454 -> 5.491) and Spearman doesn't move. The
# sweep on screen still runs 0.0-1.0.
GOALS_CONCEDED_SHRINKAGE = 1.0

# Columns projected as simple trailing means (via analytics.features.trailing_feature,
# which includes the promoted-club pooled-prior fallback for missing history).
_TRAILING_COLUMNS = [
    "goals_scored", "assists", "clean_sheets", "goals_conceded", "saves",
    "bonus", "yellow_cards", "red_cards", "own_goals", "penalties_missed", "penalties_saved",
]

# The goals and assists heads read these instead of goals_scored/assists:
# a realized goal count is mostly finishing variance, and xG is the part of
# it that repeats. Every season in the archive and the live store carry
# them.
_XG_COLUMNS = ["expected_goals", "expected_assists"]


def _favorable_scale(difficulty: float) -> float:
    """>1 for an easier-than-average fixture, <1 for harder."""
    return 1 + (3 - difficulty) * DIFFICULTY_SCALE_STEP


def _adverse_scale(difficulty: float) -> float:
    """>1 for a harder-than-average fixture, <1 for easier."""
    return 1 + (difficulty - 3) * DIFFICULTY_SCALE_STEP


def _dc_threshold_met_expr(dc_thresholds: dict[str, int | None]) -> pl.Expr:
    expr = pl.lit(False)
    for position, threshold in dc_thresholds.items():
        if threshold is None:
            continue
        expr = pl.when(pl.col("position") == position).then(pl.col("defensive_contribution") >= threshold).otherwise(expr)
    return expr.cast(pl.Float64)


def project_event_vectors(
    train_df: pl.DataFrame,
    target_roster: pl.DataFrame,
    target_gw: int,
    config: dict[str, Any],
    window: int = DEFAULT_WINDOW,
    minutes_window: int = DEFAULT_MINUTES_WINDOW,
    prior_history: pl.DataFrame | None = None,
    availability: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """target_roster needs: element_id, position, team, is_promoted_club,
    and (optionally) custom_difficulty — see project_points, which joins
    that in before calling this. Returns target_roster plus one
    `{col}_trailing` column per component and `p_blank`/`p_short`/`p_full`.

    Minutes gets its own, shorter window than the scoring-rate components
    (§4.2: "this dominates everything else and deserves the most
    attention"). Rotation and injury status are far more time-sensitive
    than a player's underlying scoring rate — a player about to lose their
    starting place should already rank low on next week's *minutes*
    expectation well before enough games pass for their goals/assists rate
    to reflect it. Empirically confirmed, not just argued: widening this
    window specifically hurts within-position rank correlation (§4.4) even
    though it doesn't move MAE much, exactly the signature of smoothing
    away exactly the information that separates "nailed on" from "rotation
    risk" among otherwise-similar scorers.

    `prior_history` is last season's rows re-keyed to this season
    (analytics.carryover), placed before gw1. They extend the scoring-rate
    windows only: minutes are this season's alone, since last season's
    minutes describe a role the player may no longer have (carrying them
    cost 0.02 of early-season Spearman in the walk-forward), and the
    pooled prior is still built from this season's train_df.

    `availability` is FPL's own flag as it stood before the deadline:
    element_id -> chance_of_playing_next_round (0/25/50/75/100, null when
    the player carries no flag). A flagged player's chances of a short and
    of a full appearance are both scaled by it, and the difference goes to
    P(blank) — the trailing minutes window can only learn an injury after
    the games it costs, and the flag knows before the first one. An
    unflagged player is left to his minutes history.
    """
    rate_df = train_df
    if prior_history is not None and prior_history.height > 0:
        shared = [c for c in train_df.columns if c in prior_history.columns]
        rate_df = pl.concat([prior_history.select(shared), train_df.select(shared)], how="vertical_relaxed")

    minutes_dist = features.minutes_distribution(train_df, target_gw - 1, minutes_window)
    df = target_roster.join(minutes_dist, on="element_id", how="left")
    # A player with zero prior history (first ever gameweek in the dataset,
    # not just this season) gets a mildly-pessimistic default rather than
    # the promoted-club pooling machinery — see module docstring on scope.
    df = df.with_columns(
        pl.col("p_blank").fill_null(0.5),
        pl.col("p_short").fill_null(0.2),
        pl.col("p_full").fill_null(0.3),
    )
    if availability is not None:
        chance = (pl.col("chance_of_playing_next_round").cast(pl.Float64) / 100).fill_null(1.0)
        df = (
            df.join(availability.select("element_id", "chance_of_playing_next_round"), on="element_id", how="left")
            .with_columns((pl.col("p_short") * chance).alias("p_short"), (pl.col("p_full") * chance).alias("p_full"))
            .with_columns((1 - pl.col("p_short") - pl.col("p_full")).alias("p_blank"))
            .drop("chance_of_playing_next_round")
        )

    # Per appearance, then scaled by this week's P(plays) — see the module
    # docstring. `{col}_trailing` stays a per-gameweek expectation, so
    # everything downstream reads it exactly as before.
    rate_cols = _TRAILING_COLUMNS + _XG_COLUMNS
    appearances = rate_df.filter(pl.col("minutes") > 0)
    pool = train_df.filter(pl.col("minutes") > 0)
    for col in rate_cols:
        df = features.trailing_feature(appearances, df, target_gw, window, col, pool_df=pool)
    df = df.with_columns([(pl.col(f"{c}_trailing") * (1 - pl.col("p_blank"))).alias(f"{c}_trailing") for c in rate_cols])

    dc_cfg = config.get("defensive_contribution")
    dc_source = rate_df.filter(pl.col("defensive_contribution").is_not_null()) if dc_cfg else None
    if dc_cfg and dc_source is not None and dc_source.height > 0:
        dc_source = dc_source.with_columns(_dc_threshold_met_expr(dc_cfg["thresholds"]).alias("dc_threshold_met"))
        dc_rate = features.trailing_mean(dc_source, target_gw - 1, window, "dc_threshold_met")
        df = df.join(
            dc_rate.select("element_id", pl.col("dc_threshold_met_trailing").alias("p_dc_threshold_trailing")),
            on="element_id",
            how="left",
        )
    else:
        df = df.with_columns(pl.lit(None, dtype=pl.Float64).alias("p_dc_threshold_trailing"))
    df = df.with_columns(pl.col("p_dc_threshold_trailing").fill_null(0.0))

    return df


def expected_points_by_component(
    row: dict[str, Any],
    config: dict[str, Any],
    goals_conceded_shrinkage: float = GOALS_CONCEDED_SHRINKAGE,
) -> dict[str, float]:
    """The projected-event-vector equivalent of
    analytics.scoring.compute_points_by_component: each head's expectation,
    broken out by the same bucket names, so a predicted bucket can be
    compared directly against its realized counterpart (§3.5, §4.4's true
    per-component error decomposition — see backtest.report). Uses the same
    per-season config compute_points does, so a rule change affects the
    model exactly the way it affects the ground truth it's validated
    against."""
    if row.get("n_fixtures") is not None:
        return _scoreline_components(row, config, goals_conceded_shrinkage)
    position = row["position"]
    difficulty = row.get("custom_difficulty") or 3.0
    fav = _favorable_scale(difficulty)
    adv = _adverse_scale(difficulty)
    minutes_cfg = config["minutes"]

    components = {
        "minutes": row["p_blank"] * minutes_cfg["none"] + row["p_short"] * minutes_cfg["short"] + row["p_full"] * minutes_cfg["full"],
        "goals": row["expected_goals_trailing"] * fav * config["goals_scored"][position],
        "assists": row["expected_assists_trailing"] * fav * config["assists"][position],
        "clean_sheets": row["clean_sheets_trailing"] * fav * config["clean_sheets"][position],
        "goals_conceded": 0.0,
        "saves": 0.0,
        "cards_and_other": (
            row["own_goals_trailing"] * config["own_goals"]
            + row["penalties_missed_trailing"] * config["penalties_missed"]
            + row["penalties_saved_trailing"] * config["penalties_saved"]
            + row["yellow_cards_trailing"] * config["yellow_cards"]
            + row["red_cards_trailing"] * config["red_cards"]
        ),
        "defensive_contribution": 0.0,
        "bonus": row["bonus_trailing"],  # direct pass-through, matching compute_points
    }

    gc_cfg = config["goals_conceded"]
    if position in gc_cfg["positions"]:
        expected_conceded = row["goals_conceded_trailing"] * adv
        components["goals_conceded"] = goals_conceded_shrinkage * (expected_conceded / gc_cfg["per"]) * gc_cfg["points"]

    if position == "GK":
        saves_cfg = config["saves"]
        expected_saves = row["saves_trailing"] * adv
        if saves_cfg["mode"] == "per_n":
            components["saves"] = (expected_saves / saves_cfg["n"]) * saves_cfg["points"]
        else:  # flat_plus_bonus: close-range/big-chance splits aren't in the historical data
            components["saves"] = expected_saves * saves_cfg["flat_rate"]

    dc_cfg = config.get("defensive_contribution")
    if dc_cfg:
        components["defensive_contribution"] = row.get("p_dc_threshold_trailing", 0.0) * dc_cfg["points"]

    return components


def _scoreline_components(
    row: dict[str, Any],
    config: dict[str, Any],
    goals_conceded_shrinkage: float,
) -> dict[str, float]:
    """expected_points_by_component with the fixture read from
    analytics.scoreline instead of the Elo difficulty: the row carries its
    team's n_fixtures, attack_mult, defence_mult, expected_cs and
    expected_gc_deductions for the target gameweek.

    Every `{col}_trailing` is one ordinary match's expectation. Rates the
    fixture moves (goals, assists, saves) are scaled by the multiplier,
    which already sums over a double gameweek; the ones it doesn't (minutes,
    bonus, cards, defensive contribution) by n_fixtures. Clean sheets and
    goals conceded are the Poisson expectations themselves, earned only by
    a player who stays on — P(60+) per match.
    """
    position = row["position"]
    n = row["n_fixtures"]
    minutes_cfg = config["minutes"]
    components = {
        "minutes": n * (row["p_blank"] * minutes_cfg["none"] + row["p_short"] * minutes_cfg["short"] + row["p_full"] * minutes_cfg["full"]),
        "goals": row["expected_goals_trailing"] * row["attack_mult"] * config["goals_scored"][position],
        "assists": row["expected_assists_trailing"] * row["attack_mult"] * config["assists"][position],
        "clean_sheets": row["p_full"] * row["expected_cs"] * config["clean_sheets"][position],
        "goals_conceded": 0.0,
        "saves": 0.0,
        "cards_and_other": n * (
            row["own_goals_trailing"] * config["own_goals"]
            + row["penalties_missed_trailing"] * config["penalties_missed"]
            + row["penalties_saved_trailing"] * config["penalties_saved"]
            + row["yellow_cards_trailing"] * config["yellow_cards"]
            + row["red_cards_trailing"] * config["red_cards"]
        ),
        "defensive_contribution": 0.0,
        "bonus": n * row["bonus_trailing"],
    }
    gc_cfg = config["goals_conceded"]
    if position in gc_cfg["positions"]:
        components["goals_conceded"] = goals_conceded_shrinkage * row["p_full"] * row["expected_gc_deductions"] * gc_cfg["points"]
    if position == "GK":
        saves_cfg = config["saves"]
        expected_saves = row["saves_trailing"] * row["defence_mult"]
        if saves_cfg["mode"] == "per_n":
            components["saves"] = (expected_saves / saves_cfg["n"]) * saves_cfg["points"]
        else:
            components["saves"] = expected_saves * saves_cfg["flat_rate"]
    dc_cfg = config.get("defensive_contribution")
    if dc_cfg:
        components["defensive_contribution"] = n * row.get("p_dc_threshold_trailing", 0.0) * dc_cfg["points"]
    return components


def expected_points_from_projection(
    row: dict[str, Any],
    config: dict[str, Any],
    goals_conceded_shrinkage: float = GOALS_CONCEDED_SHRINKAGE,
) -> float:
    """The projected total: sum(expected_points_by_component(...).values())."""
    return sum(expected_points_by_component(row, config, goals_conceded_shrinkage).values())


def with_scoreline(roster: pl.DataFrame, scoreline: pl.DataFrame) -> pl.DataFrame:
    """`roster` with its team's analytics.scoreline columns for the target
    gameweek, which expected_points_by_component then reads the fixture
    from. A team missing from the table has no fixture that gameweek."""
    return roster.join(scoreline, on="team", how="left").with_columns(
        pl.col("n_fixtures", "attack_mult", "defence_mult", "expected_cs", "expected_gc_deductions").fill_null(0)
    )


def project_points(
    train_df: pl.DataFrame,
    target_roster: pl.DataFrame,
    target_gw: int,
    config: dict[str, Any],
    difficulty_table: pl.DataFrame,
    window: int = DEFAULT_WINDOW,
    minutes_window: int = DEFAULT_MINUTES_WINDOW,
    goals_conceded_shrinkage: float = GOALS_CONCEDED_SHRINKAGE,
    on_projection: Callable[[int, pl.DataFrame], None] | None = None,
    prior_history: pl.DataFrame | None = None,
    availability: pl.DataFrame | None = None,
    scoreline: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """The full model, in the same (train_df, target_roster, target_gw) ->
    DataFrame[element_id, prediction] shape as backtest.baselines' three
    functions — bind `config`/`difficulty_table` with functools.partial to
    register this as another entry in a walk_forward `baselines` dict for a
    direct, apples-to-apples comparison (§4.4).

    `difficulty_table` is analytics.fdr.team_gameweek_difficulty's output
    for the whole season — safe to pass un-truncated for any target_gw
    since its values are pre-match by construction (see fdr.py).

    `on_projection`, if given, receives (target_gw, event vectors) before
    they are collapsed to points, so a caller that also needs the
    per-component detail (analytics.evaluate.run_evaluation) can take it
    from this pass instead of projecting the same gameweek a second time.
    """
    gw_difficulty = difficulty_table.filter(pl.col("gw") == target_gw).select("team", "custom_difficulty")
    roster = target_roster.join(gw_difficulty, on="team", how="left").with_columns(pl.col("custom_difficulty").fill_null(3.0))
    if scoreline is not None:
        roster = with_scoreline(roster, scoreline)

    projected = project_event_vectors(train_df, roster, target_gw, config, window, minutes_window, prior_history, availability)
    if on_projection is not None:
        on_projection(target_gw, projected)
    predictions = [
        expected_points_from_projection(row, config, goals_conceded_shrinkage)
        for row in projected.to_dicts()
    ]
    return pl.DataFrame({"element_id": projected["element_id"], "prediction": predictions})
