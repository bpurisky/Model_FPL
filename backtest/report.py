"""Metrics for a walk-forward result set (§3.5): MAE, RMSE, within-position
Spearman rank correlation, calibration, and an error decomposition.

No scipy in the locked stack, so Spearman is computed directly: Pearson
correlation of the ranks, with polars' own (ties-averaged) `.rank()`.
"""

from __future__ import annotations

import json
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
