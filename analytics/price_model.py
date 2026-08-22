"""§4.1/§6.4/§6.5's price-change model: predicts which players are likely
to rise or fall in price from `analytics/trending.py`'s pressure signal,
and evaluates that prediction's hit rate against real observed price
changes — with an explicit, wide confidence interval (§6.4: market signals
have no historical analogue and gw13 is their *first* evaluation window,
not a settled one; never report a bare point estimate).

No scipy in the locked stack (§1.1), so the confidence interval is a
standard normal approximation for a binomial proportion — matching
`backtest/report.py`'s own precedent of computing Spearman without scipy.

The rise/fall thresholds below are deliberately unfitted round numbers,
not calibrated against real data — there isn't any yet (§6.4 again: this
whole model's evaluation window only opens once prices have actually had
time to move). Revisit them once `evaluate_price_model` has accumulated
enough real observations to tune against; don't mistake the current
values for a considered choice.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import polars as pl

from analytics.deltas import compute_deltas, reference_timestamps, state_as_of
from analytics.trending import price_change_pressure

Z_95 = 1.959963985  # standard normal critical value for a 95% CI

DEFAULT_RISE_THRESHOLD = 0.5
DEFAULT_FALL_THRESHOLD = -0.5


@dataclass(frozen=True)
class PriceModelEvaluation:
    n: int  # players with both a prediction and a known actual outcome
    n_moves_predicted: int  # of those, how many were predicted to actually move (rise or fall)
    hit_rate: float | None  # None, never a fabricated 0.0, when n_moves_predicted == 0
    ci_low: float | None
    ci_high: float | None


def predict_price_changes(
    pressure: pl.DataFrame, rise_threshold: float = DEFAULT_RISE_THRESHOLD, fall_threshold: float = DEFAULT_FALL_THRESHOLD
) -> pl.DataFrame:
    """`pressure`: `analytics.trending.price_change_pressure`'s output
    (element_id, net_transfers, price_change_pressure). Returns element_id,
    price_change_pressure, predicted_direction ("rise"/"fall"/"stable")."""
    return pressure.select(
        "element_id",
        "price_change_pressure",
        pl.when(pl.col("price_change_pressure") >= rise_threshold)
        .then(pl.lit("rise"))
        .when(pl.col("price_change_pressure") <= fall_threshold)
        .then(pl.lit("fall"))
        .otherwise(pl.lit("stable"))
        .alias("predicted_direction"),
    )


def actual_price_direction(distilled_dir: Path, before: datetime, after: datetime) -> pl.DataFrame:
    """element_id, actual_direction ("rise"/"fall"/"stable") from real
    `now_cost` movement between two points in time — only players with a
    known state at both ends are included (inner join), never a guessed
    baseline."""
    before_state = state_as_of(distilled_dir, before).select("element_id", pl.col("now_cost").alias("now_cost_before"))
    after_state = state_as_of(distilled_dir, after).select("element_id", pl.col("now_cost").alias("now_cost_after"))
    joined = before_state.join(after_state, on="element_id", how="inner")
    return joined.select(
        "element_id",
        pl.when(pl.col("now_cost_after") > pl.col("now_cost_before"))
        .then(pl.lit("rise"))
        .when(pl.col("now_cost_after") < pl.col("now_cost_before"))
        .then(pl.lit("fall"))
        .otherwise(pl.lit("stable"))
        .alias("actual_direction"),
    )


def _normal_approx_ci(hits: int, n: int, z: float = Z_95) -> tuple[float, float]:
    """Fine for the sample sizes this model will realistically see (dozens
    to low hundreds of "moved" predictions per week) — wide and honest at
    small n rather than pretending precision it doesn't have."""
    if n == 0:
        return (0.0, 1.0)
    p = hits / n
    margin = z * ((p * (1 - p)) / n) ** 0.5
    return (max(0.0, p - margin), min(1.0, p + margin))


def evaluate_price_model(predictions: pl.DataFrame, actuals: pl.DataFrame) -> PriceModelEvaluation:
    """§6.5's "reports its hit rate with a stated confidence interval": of
    every player predicted to actually *move* (rise or fall — a "stable"
    prediction is trivially right most of the time and isn't the claim
    being tested), how often did the real direction match. Never
    fabricates a rate when zero predictions called for a move.
    """
    joined = predictions.join(actuals, on="element_id", how="inner")
    moved = joined.filter(pl.col("predicted_direction") != "stable")
    n_moves = moved.height
    if n_moves == 0:
        return PriceModelEvaluation(n=joined.height, n_moves_predicted=0, hit_rate=None, ci_low=None, ci_high=None)
    hits = moved.filter(pl.col("predicted_direction") == pl.col("actual_direction")).height
    ci_low, ci_high = _normal_approx_ci(hits, n_moves)
    return PriceModelEvaluation(n=joined.height, n_moves_predicted=n_moves, hit_rate=hits / n_moves, ci_low=ci_low, ci_high=ci_high)


def run_price_model_evaluation(
    distilled_dir: Path,
    prediction_ts: datetime,
    evaluation_ts: datetime,
    pressure_window: str = "1h",
    rise_threshold: float = DEFAULT_RISE_THRESHOLD,
    fall_threshold: float = DEFAULT_FALL_THRESHOLD,
) -> PriceModelEvaluation:
    """The live-data path: predict from pressure as of `prediction_ts`,
    then check the real outcome as of `evaluation_ts` (typically 24h later
    — FPL price changes land roughly once a day). Ties together
    analytics/deltas.py, analytics/trending.py, and this module's own
    predict/evaluate pair.
    """
    refs = reference_timestamps(prediction_ts)
    if pressure_window not in refs:
        refs[pressure_window] = prediction_ts  # caller-supplied window not one of the standard three
    deltas = compute_deltas(distilled_dir, prediction_ts, refs)
    if deltas.height == 0:
        return PriceModelEvaluation(n=0, n_moves_predicted=0, hit_rate=None, ci_low=None, ci_high=None)

    pressure = price_change_pressure(deltas, window=pressure_window)
    predictions = predict_price_changes(pressure, rise_threshold, fall_threshold)
    actuals = actual_price_direction(distilled_dir, prediction_ts, evaluation_ts)
    return evaluate_price_model(predictions, actuals)
