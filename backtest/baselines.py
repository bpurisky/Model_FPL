"""Three benchmarks the eventual event model must beat (§3.4). Implemented
before the model exists, and against a real data hazard worth naming:

FPL's `form` field (baseline 2) is itself a live snapshot value — nobody
recorded what it actually read before each historical deadline, and per §0.1
that data is now permanently unrecoverable. `fpl_form_approx_baseline` below
replicates FPL's own documented definition (mean points per match over the
trailing 30 days) from our reconstructed per-gameweek data instead. This is
an approximation of the historical field, not the literal value managers saw
— documented here rather than silently passed off as the real thing (§8).

All three baselines share one fallback path for players with no trailing
history: the promoted-club prior (§3.2) — never a null, never last season's
relegated club by name. See `_fill_missing_with_prior`.
"""

from __future__ import annotations

from datetime import timedelta

import polars as pl

# How much a fixture difficulty step (FPL's 1-5 rating) scales the trailing
# mean, relative to difficulty=3 (scale=1.0). Deliberately gentle — this is
# a hurdle baseline, not a real difficulty model (that's §4.3, Phase 2).
DIFFICULTY_SCALE_STEP = 0.075

DEFAULT_WINDOW_GAMEWEEKS = 5
DEFAULT_WINDOW_DAYS = 30


def _trailing_points_as_of(df: pl.DataFrame, upto_gw: int, window: int) -> pl.DataFrame:
    """element_id -> mean total_points over each player's last `window`
    appearances in gameweeks <= upto_gw."""
    filtered = df.filter(pl.col("gw") <= upto_gw).sort(["element_id", "gw"])
    tails = filtered.group_by("element_id", maintain_order=True).tail(window)
    return tails.group_by("element_id").agg(
        pl.col("total_points").mean().alias("trailing_mean"),
        pl.len().alias("n_games"),
    )


def _fill_missing_with_prior(joined: pl.DataFrame, train_df: pl.DataFrame, target_gw: int) -> pl.DataFrame:
    """Fills null `trailing_mean` (players with no prior appearances).

    Promoted-club players pool from their promoted-club peers in the same
    position (§3.2's "pooled promoted-club baseline"). Anyone else missing
    — a genuine new-to-the-league signing, say — pools from same-position,
    same-promoted-club-status peers generally. A final coalesce to the
    all-players-in-position mean covers the degenerate case where even that
    pool is empty (only possible this early that no prior gameweek exists
    at all). Never a null; never a name-matched relegated-club lookup.
    """
    history = train_df.filter(pl.col("gw") < target_gw)

    position_pool = history.group_by(["position", "is_promoted_club"]).agg(pl.col("total_points").mean().alias("pool_mean"))
    all_position_pool = history.group_by("position").agg(pl.col("total_points").mean().alias("all_pool_mean"))

    out = joined.join(position_pool, on=["position", "is_promoted_club"], how="left")
    out = out.join(all_position_pool, on="position", how="left")
    out = out.with_columns(pl.coalesce([pl.col("trailing_mean"), pl.col("pool_mean"), pl.col("all_pool_mean"), pl.lit(0.0)]).alias("trailing_mean"))
    return out.drop("pool_mean", "all_pool_mean")


def _core_trailing_prediction(train_df: pl.DataFrame, target_roster: pl.DataFrame, target_gw: int, window: int) -> pl.DataFrame:
    trailing = _trailing_points_as_of(train_df, target_gw - 1, window)
    joined = target_roster.join(trailing, on="element_id", how="left")
    return _fill_missing_with_prior(joined, train_df, target_gw)


def trailing_mean_baseline(
    train_df: pl.DataFrame, target_roster: pl.DataFrame, target_gw: int, window: int = DEFAULT_WINDOW_GAMEWEEKS
) -> pl.DataFrame:
    """Baseline 1: player's mean points over the prior `window` gameweeks."""
    result = _core_trailing_prediction(train_df, target_roster, target_gw, window)
    return result.select("element_id", pl.col("trailing_mean").alias("prediction"))


def fpl_form_approx_baseline(
    train_df: pl.DataFrame, target_roster: pl.DataFrame, target_gw: int, window_days: int = DEFAULT_WINDOW_DAYS
) -> pl.DataFrame:
    """Baseline 2: approximation of FPL's `form` field — see module docstring."""
    history = train_df.filter(pl.col("gw") < target_gw)
    as_of = history["kickoff_time"].max()

    if as_of is None:
        windowed = history  # empty; nothing has been played yet
    else:
        cutoff = as_of - timedelta(days=window_days)
        windowed = history.filter(pl.col("kickoff_time") > cutoff)

    trailing = windowed.group_by("element_id").agg(pl.col("total_points").mean().alias("trailing_mean"))
    joined = target_roster.join(trailing, on="element_id", how="left")
    filled = _fill_missing_with_prior(joined, train_df, target_gw)
    return filled.select("element_id", pl.col("trailing_mean").alias("prediction"))


def fixture_adjusted_trailing_mean_baseline(
    train_df: pl.DataFrame,
    target_roster: pl.DataFrame,
    target_gw: int,
    window: int = DEFAULT_WINDOW_GAMEWEEKS,
    difficulty_scale_step: float = DIFFICULTY_SCALE_STEP,
) -> pl.DataFrame:
    """Baseline 3: baseline 1 scaled by the upcoming fixture's difficulty.

    `target_roster.opponent_difficulty` is the target gameweek's own,
    already-published FPL difficulty rating (1=easiest, 5=hardest) for that
    player's fixture — known well before the deadline, not an outcome, so
    using it here isn't leakage. Missing (blank gameweek, postponed fixture)
    defaults to difficulty=3, i.e. no adjustment.
    """
    result = _core_trailing_prediction(train_df, target_roster, target_gw, window)
    scale = 1 + (3 - pl.col("opponent_difficulty").fill_null(3)) * difficulty_scale_step
    result = result.with_columns((pl.col("trailing_mean") * scale).alias("prediction"))
    return result.select("element_id", "prediction")


BASELINES = {
    "trailing_mean": trailing_mean_baseline,
    "fpl_form_approx": fpl_form_approx_baseline,
    "fixture_adjusted_trailing_mean": fixture_adjusted_trailing_mean_baseline,
}
