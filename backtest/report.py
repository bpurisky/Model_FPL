"""Metrics for a walk-forward result set (§3.5): MAE, RMSE, within-position
Spearman rank correlation, calibration, and an error decomposition.

No scipy in the locked stack, so Spearman is computed directly: Pearson
correlation of the ranks, with polars' own (ties-averaged) `.rank()`.
Its significance (n and a two-sided p-value, §5.3.2) is built from the
regularized incomplete beta in the same spirit -- see the "significance"
block below.
"""

from __future__ import annotations

import json
from math import exp, lgamma, log, log1p
from pathlib import Path

import polars as pl


def mae(df: pl.DataFrame) -> float:
    if df.height == 0:
        return float("nan")
    return float(df["error"].abs().mean())


def rmse(df: pl.DataFrame) -> float:
    if df.height == 0:
        return float("nan")
    return float((df["error"] ** 2).mean() ** 0.5)


def _pearson(x: pl.Series, y: pl.Series) -> float:
    n = x.len()
    if n < 2:
        return float("nan")
    mean_x, mean_y = x.mean(), y.mean()
    cov = ((x - mean_x) * (y - mean_y)).sum() / (n - 1)
    std_x, std_y = x.std(), y.std()
    if not std_x or not std_y:
        return float("nan")
    return float(cov / (std_x * std_y))


def spearman(x: pl.Series, y: pl.Series) -> float:
    return _pearson(x.rank(), y.rank())


# --------------------------------------------------------------------------
# significance
#
# §5.3.2's correlations.json wants rho, n and a p-value side by side: rho
# alone is unreadable without knowing whether it came from 40 rows or
# 40,000. There is no scipy in the locked stack, so the Student-t CDF is
# built here from the regularized incomplete beta -- the exact identity
#
#     P(|T_v| >= |t|)  =  I_{v/(v+t^2)}(v/2, 1/2)
#
# rather than a normal approximation, which is wrong in exactly the
# small-n case the p-value is there to flag.
# --------------------------------------------------------------------------

_BETACF_TINY = 1e-300
_BETACF_EPS = 3e-16
_BETACF_MAX_ITER = 300


def _log_beta(a: float, b: float) -> float:
    return lgamma(a) + lgamma(b) - lgamma(a + b)


def _betacf(a: float, b: float, x: float) -> float:
    """Continued fraction for the incomplete beta, by modified Lentz."""
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < _BETACF_TINY:
        d = _BETACF_TINY
    d = 1.0 / d
    h = d
    for m in range(1, _BETACF_MAX_ITER + 1):
        m2 = 2 * m
        # even step
        num = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + num * d
        if abs(d) < _BETACF_TINY:
            d = _BETACF_TINY
        c = 1.0 + num / c
        if abs(c) < _BETACF_TINY:
            c = _BETACF_TINY
        d = 1.0 / d
        h *= d * c
        # odd step
        num = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + num * d
        if abs(d) < _BETACF_TINY:
            d = _BETACF_TINY
        c = 1.0 + num / c
        if abs(c) < _BETACF_TINY:
            c = _BETACF_TINY
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < _BETACF_EPS:
            break
    return h


def regularized_incomplete_beta(a: float, b: float, x: float) -> float:
    """I_x(a, b). Uses the reflection I_x(a,b) = 1 - I_{1-x}(b,a) on the
    side where the continued fraction converges slowly."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    front = exp(a * log(x) + b * log1p(-x) - _log_beta(a, b))
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1.0 - x) / b


def two_sided_t_p_value(t: float, dof: float) -> float:
    """P(|T| >= |t|) for Student's t with `dof` degrees of freedom."""
    if dof <= 0 or t != t:  # no df, or t is NaN
        return float("nan")
    if t in (float("inf"), float("-inf")):
        return 0.0
    return regularized_incomplete_beta(dof / 2.0, 0.5, dof / (dof + t * t))


def spearman_p_value(rho: float, n: int) -> float:
    """Two-sided p for a Spearman rho over n pairs, via the usual
    t = rho * sqrt((n-2)/(1-rho^2)) with n-2 degrees of freedom.

    Approximate in the presence of ties, and FPL points are almost
    nothing but ties -- most players score 0, 1 or 2 in a gameweek. It is
    reported because §5.3.2 asks for it, and because at these sample
    sizes it is read as "not a small-sample artefact", not as a
    hypothesis test. Do not read a difference between two tiny p-values
    as meaning anything.
    """
    if rho != rho or n < 3:
        return float("nan")
    denom = 1.0 - rho * rho
    if denom <= 0.0:  # |rho| == 1: t is infinite
        return 0.0
    t = rho * ((n - 2) / denom) ** 0.5
    return two_sided_t_p_value(t, n - 2)


def spearman_with_significance(x: pl.Series, y: pl.Series) -> dict[str, float]:
    """Spearman rho alongside the n it was computed over and its p-value."""
    rho = spearman(x, y)
    n = x.len()
    return {"rho": rho, "n": n, "p_value": spearman_p_value(rho, n)}


def spearman_within_position(df: pl.DataFrame) -> dict[str, float]:
    """Spearman(prediction, total_points) computed separately per position,
    plus an unweighted mean across positions as a single summary figure."""
    per_position: dict[str, float] = {}
    for position in sorted(df["position"].drop_nulls().unique().to_list()):
        subset = df.filter(pl.col("position") == position)
        per_position[position] = spearman(subset["prediction"], subset["total_points"])
    valid = [v for v in per_position.values() if v == v]  # drop NaN
    per_position["mean"] = sum(valid) / len(valid) if valid else float("nan")
    return per_position


def spearman_within_position_significance(df: pl.DataFrame) -> dict[str, dict[str, float]]:
    """`spearman_within_position` with n and a p-value per position.

    Deliberately carries no "mean" row, unlike its rho-only sibling: the
    unweighted mean of four correlations is a readable summary, but it is
    not a statistic any sampling distribution describes, so there is no
    p-value to put beside it. Reporting one would invent a number.
    """
    out: dict[str, dict[str, float]] = {}
    for position in sorted(df["position"].drop_nulls().unique().to_list()):
        subset = df.filter(pl.col("position") == position)
        out[position] = spearman_with_significance(subset["prediction"], subset["total_points"])
    return out


def calibration_curve(df: pl.DataFrame, n_bins: int = 10) -> list[dict]:
    """Deciles of `prediction`: mean predicted vs. mean actual per bin. A
    well-calibrated baseline tracks the diagonal; systematic gaps show bias."""
    if df.height == 0:
        return []
    binned = df.with_columns(pl.col("prediction").qcut(n_bins, labels=[str(i) for i in range(n_bins)], allow_duplicates=True).alias("bin"))
    agg = binned.group_by("bin").agg(
        pl.col("prediction").mean().alias("mean_prediction"),
        pl.col("total_points").mean().alias("mean_actual"),
        pl.len().alias("n"),
    )
    return agg.sort("bin").to_dicts()


def error_by_event_occurrence(df: pl.DataFrame) -> dict[str, dict]:
    """A decomposition of error by event type (§3.5), scoped to what a
    scalar points baseline can actually support: MAE split by which kind of
    event occurred, rather than by predicted-vs-actual event components
    (that needs an event-level model — Phase 2). Once projections.py exists,
    this should be replaced with a true per-component decomposition."""
    buckets = {
        "blank_(0_minutes)": pl.col("minutes") == 0,
        "played_no_goal_involvement": (pl.col("minutes") > 0) & (pl.col("goals_scored") == 0) & (pl.col("assists") == 0),
        "goal_involvement": (pl.col("goals_scored") > 0) | (pl.col("assists") > 0),
        "clean_sheet": pl.col("clean_sheets") > 0,
        "bonus_earned": pl.col("bonus") > 0,
    }
    result = {}
    for label, predicate in buckets.items():
        subset = df.filter(predicate)
        result[label] = {"mae": mae(subset), "n": subset.height}
    return result


def summarize(df: pl.DataFrame, group_cols: list[str]) -> dict:
    if df.height == 0:
        return {}
    summary: dict = {}
    for key_values, group in df.group_by(group_cols, maintain_order=True):
        key = key_values[0] if len(group_cols) == 1 else tuple(key_values)
        summary[str(key)] = {
            "n": group.height,
            "mae": mae(group),
            "rmse": rmse(group),
            "spearman_within_position": spearman_within_position(group),
            "spearman_significance": spearman_within_position_significance(group),
            "error_by_event": error_by_event_occurrence(group),
        }
    return summary


def build_report(results: pl.DataFrame) -> dict:
    """Per-season-per-baseline, per-baseline-pooled-across-seasons, and a
    top-level pooled-everything summary. Reproducible from a single command
    against the committed data/historical/*.parquet (§3.6, §8)."""
    if results.height == 0:
        return {"per_season_baseline": {}, "pooled_baseline": {}, "n_rows": 0}
    return {
        "n_rows": results.height,
        "per_season_baseline": summarize(results, ["season", "baseline"]),
        "pooled_baseline": summarize(results, ["baseline"]),
        "calibration": {
            baseline: calibration_curve(results.filter(pl.col("baseline") == baseline))
            for baseline in results["baseline"].unique().to_list()
        },
    }


def write_report(report: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")


def component_decomposition_mae(predicted_components: list[dict[str, float]], actual_components: list[dict[str, float]]) -> dict[str, float]:
    """A true per-component error decomposition (§3.5, §4.4): MAE between
    the event model's predicted point contribution and the realized one,
    per event-type bucket (minutes, goals, assists, clean_sheets,
    goals_conceded, saves, defensive_contribution, bonus, cards_and_other).

    Supersedes error_by_event_occurrence (Phase 1's proxy, kept for the
    scalar-only baselines that have no per-component prediction to compare
    against) now that analytics/projections.py exists to predict one.
    Requires `predicted_components`/`actual_components` from
    analytics.evaluate.run_component_decomposition — same length, same
    bucket keys, row-aligned.
    """
    if not predicted_components:
        return {}
    keys = predicted_components[0].keys()
    result = {}
    for key in keys:
        diffs = [abs(p[key] - a[key]) for p, a in zip(predicted_components, actual_components)]
        result[key] = sum(diffs) / len(diffs)
    return result


def minutes_head_metrics(predicted_minutes_dist: list[dict[str, float]], actual_minutes: list[int], short_threshold: int = 60) -> dict[str, float]:
    """§4.4: "Minutes head evaluated separately with its own metrics" —
    §4.2 calls minutes prediction the highest-leverage, most-failure-prone
    component, so it gets its own scorecard rather than being folded into
    the pooled MAE. Brier score per bucket (lower is better-calibrated;
    0 is perfect, 1 is maximally wrong) plus MAE of a derived expected-
    minutes scalar against actual minutes played.
    """
    n = len(actual_minutes)
    if n == 0:
        return {"brier_blank": float("nan"), "brier_short": float("nan"), "brier_full": float("nan"), "mae_expected_minutes": float("nan"), "n": 0}

    brier_blank = sum((row["p_blank"] - (1.0 if m == 0 else 0.0)) ** 2 for row, m in zip(predicted_minutes_dist, actual_minutes)) / n
    brier_short = sum((row["p_short"] - (1.0 if 0 < m < short_threshold else 0.0)) ** 2 for row, m in zip(predicted_minutes_dist, actual_minutes)) / n
    brier_full = sum((row["p_full"] - (1.0 if m >= short_threshold else 0.0)) ** 2 for row, m in zip(predicted_minutes_dist, actual_minutes)) / n

    # A short appearance's expected minutes uses the FPL short-band
    # midpoint as a rough per-bucket reference point, not a claim about the
    # true conditional mean — this scalar exists only to give minutes MAE a
    # comparable "how far off in actual minutes" figure, not to drive scoring.
    expected_minutes = [row["p_short"] * (short_threshold / 2) + row["p_full"] * 90 for row in predicted_minutes_dist]
    mae_expected_minutes = sum(abs(e - m) for e, m in zip(expected_minutes, actual_minutes)) / n

    return {
        "brier_blank": brier_blank,
        "brier_short": brier_short,
        "brier_full": brier_full,
        "mae_expected_minutes": mae_expected_minutes,
        "n": n,
    }
