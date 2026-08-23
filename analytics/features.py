"""Point-in-time feature construction (§4.2).

Generalizes the trailing-rate/pooled-prior pattern established in
backtest/baselines.py (there specific to total_points) so projections.py
can build one feature per event-model head without repeating the pooling
and promoted-club-fallback logic for each. Phase 1's baselines.py is left
untouched — these are new, general-purpose functions, not a refactor of
already-validated code.
"""

from __future__ import annotations

import polars as pl


def trailing_mean(df: pl.DataFrame, upto_gw: int, window: int, value_col: str) -> pl.DataFrame:
    """element_id -> mean of `value_col` over each player's last `window`
    appearances in gameweeks <= upto_gw, plus n_games for context."""
    filtered = df.filter(pl.col("gw") <= upto_gw).sort(["element_id", "gw"])
    tails = filtered.group_by("element_id", maintain_order=True).tail(window)
    return tails.group_by("element_id").agg(
        pl.col(value_col).mean().alias(f"{value_col}_trailing"),
        pl.len().alias("n_games"),
    )


def minutes_distribution(df: pl.DataFrame, upto_gw: int, window: int, short_threshold: int = 60) -> pl.DataFrame:
    """element_id -> empirical P(blank), P(short appearance), P(60+ minutes)
    over the trailing window. Used directly as the minutes-points
    expectation (§4.2 — "this dominates everything else") rather than
    thresholding a single mean-minutes figure, which would systematically
    misstate the expectation of a step function (Jensen's inequality)."""
    filtered = df.filter(pl.col("gw") <= upto_gw).sort(["element_id", "gw"])
    tails = filtered.group_by("element_id", maintain_order=True).tail(window)
    return tails.group_by("element_id").agg(
        (pl.col("minutes") == 0).mean().alias("p_blank"),
        ((pl.col("minutes") > 0) & (pl.col("minutes") < short_threshold)).mean().alias("p_short"),
        (pl.col("minutes") >= short_threshold).mean().alias("p_full"),
        pl.len().alias("n_games"),
    )


def fill_missing_with_pooled_prior(
    joined: pl.DataFrame, train_df: pl.DataFrame, target_gw: int, value_col: str
) -> pl.DataFrame:
    """Fills a null trailing feature (§3.2's promoted-club prior,
    generalized to any column): promoted-club players pool from their
    promoted-club peers in the same position; anyone else missing pools
    from same-position, same-promoted-club-status peers generally; a final
    coalesce to the all-players-in-position mean covers the case where even
    that pool is empty. Never a null.

    `joined` must have `{value_col}_trailing`, `position`, `is_promoted_club`.
    """
    history = train_df.filter(pl.col("gw") < target_gw)
    trailing_col = f"{value_col}_trailing"

    position_pool = history.group_by(["position", "is_promoted_club"]).agg(pl.col(value_col).mean().alias("pool_mean"))
    all_position_pool = history.group_by("position").agg(pl.col(value_col).mean().alias("all_pool_mean"))

    out = joined.join(position_pool, on=["position", "is_promoted_club"], how="left")
    out = out.join(all_position_pool, on="position", how="left")
    out = out.with_columns(
        pl.coalesce([pl.col(trailing_col), pl.col("pool_mean"), pl.col("all_pool_mean"), pl.lit(0.0)]).alias(trailing_col)
    )
    return out.drop("pool_mean", "all_pool_mean")


def trailing_feature(df: pl.DataFrame, target_roster: pl.DataFrame, target_gw: int, window: int, value_col: str) -> pl.DataFrame:
    """The full pipeline for one column: trailing mean, joined onto the
    target roster, with the pooled prior filling gaps. Returns
    `target_roster` plus one new column: `{value_col}_trailing`."""
    trailing = trailing_mean(df, target_gw - 1, window, value_col).drop("n_games")
    joined = target_roster.join(trailing, on="element_id", how="left")
    return fill_missing_with_pooled_prior(joined, df, target_gw, value_col)


def trailing_minutes_reliability(
    df: pl.DataFrame, window: int, short_threshold: int = 60
) -> pl.DataFrame:
    """Per-row point-in-time `p_full`: the share of the previous `window`
    gameweeks in which this player reached `short_threshold` minutes.

    The same definition as `minutes_distribution`'s `p_full`, evaluated for
    every row at once instead of for one gameweek at a time — the panel
    needs it across three seasons and 85,000 rows, and calling the
    gameweek-at-a-time version in a loop would rebuild the same frame 114
    times. Reused rather than redefined so the exported column inherits
    that head's published Brier score (0.1007 over 83,035 predictions)
    instead of quietly becoming a second, unvalidated statistic.

    `shift(1)` is what keeps it point-in-time (§0.3): the gameweek being
    described must not contribute to the rate describing it. The first
    gameweek of a player's season is therefore null — unknown, not zero,
    since a player with no history is not a player who never plays.
    """
    return df.sort(["season", "element_id", "gw"]).with_columns(
        (pl.col("minutes") >= short_threshold)
        .cast(pl.Float64)
        .shift(1)
        .rolling_mean(window, min_samples=1)
        .over(["season", "element_id"])
        .alias("minutes_reliability")
    )
