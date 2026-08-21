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
"""

from __future__ import annotations

from typing import Any

import polars as pl

from analytics import features

DEFAULT_WINDOW = 5
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
GOALS_CONCEDED_SHRINKAGE = 0.7

# Columns projected as simple trailing means (via analytics.features.trailing_feature,
# which includes the promoted-club pooled-prior fallback for missing history).
_TRAILING_COLUMNS = [
    "goals_scored", "assists", "clean_sheets", "goals_conceded", "saves",
    "bonus", "yellow_cards", "red_cards", "own_goals", "penalties_missed", "penalties_saved",
]


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
    """
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

    for col in _TRAILING_COLUMNS:
        df = features.trailing_feature(train_df, df, target_gw, window, col)

    dc_cfg = config.get("defensive_contribution")
    dc_source = train_df.filter(pl.col("defensive_contribution").is_not_null()) if dc_cfg else None
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


def expected_points_by_component(row: dict[str, Any], config: dict[str, Any]) -> dict[str, float]:
    """The projected-event-vector equivalent of
    analytics.scoring.compute_points_by_component: each head's expectation,
    broken out by the same bucket names, so a predicted bucket can be
    compared directly against its realized counterpart (§3.5, §4.4's true
    per-component error decomposition — see backtest.report). Uses the same
    per-season config compute_points does, so a rule change affects the
    model exactly the way it affects the ground truth it's validated
    against."""
    position = row["position"]
    difficulty = row.get("custom_difficulty") or 3.0
    fav = _favorable_scale(difficulty)
    adv = _adverse_scale(difficulty)
    minutes_cfg = config["minutes"]

    components = {
        "minutes": row["p_blank"] * minutes_cfg["none"] + row["p_short"] * minutes_cfg["short"] + row["p_full"] * minutes_cfg["full"],
        "goals": row["goals_scored_trailing"] * fav * config["goals_scored"][position],
        "assists": row["assists_trailing"] * fav * config["assists"][position],
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
        components["goals_conceded"] = GOALS_CONCEDED_SHRINKAGE * (expected_conceded / gc_cfg["per"]) * gc_cfg["points"]

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


def expected_points_from_projection(row: dict[str, Any], config: dict[str, Any]) -> float:
    """The projected total: sum(expected_points_by_component(...).values())."""
    return sum(expected_points_by_component(row, config).values())


def project_points(
    train_df: pl.DataFrame,
    target_roster: pl.DataFrame,
    target_gw: int,
    config: dict[str, Any],
    difficulty_table: pl.DataFrame,
    window: int = DEFAULT_WINDOW,
    minutes_window: int = DEFAULT_MINUTES_WINDOW,
) -> pl.DataFrame:
    """The full model, in the same (train_df, target_roster, target_gw) ->
    DataFrame[element_id, prediction] shape as backtest.baselines' three
    functions — bind `config`/`difficulty_table` with functools.partial to
    register this as another entry in a walk_forward `baselines` dict for a
    direct, apples-to-apples comparison (§4.4).

    `difficulty_table` is analytics.fdr.team_gameweek_difficulty's output
    for the whole season — safe to pass un-truncated for any target_gw
    since its values are pre-match by construction (see fdr.py).
    """
    gw_difficulty = difficulty_table.filter(pl.col("gw") == target_gw).select("team", "custom_difficulty")
    roster = target_roster.join(gw_difficulty, on="team", how="left").with_columns(pl.col("custom_difficulty").fill_null(3.0))

    projected = project_event_vectors(train_df, roster, target_gw, config, window, minutes_window)
    predictions = [expected_points_from_projection(row, config) for row in projected.to_dicts()]
    return pl.DataFrame({"element_id": projected["element_id"], "prediction": predictions})
