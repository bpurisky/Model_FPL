"""§3.5 metrics and §3.6 / §8 reproducibility of the report output."""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl
import pytest

from backtest.report import (
    build_report,
    calibration_curve,
    component_decomposition_mae,
    mae,
    minutes_head_metrics,
    rmse,
    spearman,
    write_report,
)


def _results_df() -> pl.DataFrame:
    # predictions perfectly track actuals for FWDs, are noise for MIDs —
    # gives a clean, known-answer case for MAE/RMSE/Spearman.
    rows = [
        {"season": "2024-25", "gw": 2, "baseline": "trailing_mean", "element_id": 1, "position": "FWD", "prediction": 5.0, "total_points": 5.0, "minutes": 90, "goals_scored": 1, "assists": 0, "clean_sheets": 0, "bonus": 1, "bps": 20},
        {"season": "2024-25", "gw": 2, "baseline": "trailing_mean", "element_id": 2, "position": "FWD", "prediction": 2.0, "total_points": 2.0, "minutes": 90, "goals_scored": 0, "assists": 0, "clean_sheets": 0, "bonus": 0, "bps": 10},
        {"season": "2024-25", "gw": 2, "baseline": "trailing_mean", "element_id": 3, "position": "FWD", "prediction": 8.0, "total_points": 8.0, "minutes": 90, "goals_scored": 2, "assists": 0, "clean_sheets": 0, "bonus": 3, "bps": 40},
        {"season": "2024-25", "gw": 2, "baseline": "trailing_mean", "element_id": 4, "position": "MID", "prediction": 5.0, "total_points": 1.0, "minutes": 90, "goals_scored": 0, "assists": 0, "clean_sheets": 0, "bonus": 0, "bps": 5},
    ]
    for row in rows:
        row["error"] = row["prediction"] - row["total_points"]
    return pl.DataFrame(rows)


def test_mae_and_rmse_known_values():
    df = _results_df()
    fwd_only = df.filter(pl.col("position") == "FWD")
    assert mae(fwd_only) == pytest.approx(0.0)  # FWD predictions are exact
    assert rmse(fwd_only) == pytest.approx(0.0)

    mid_only = df.filter(pl.col("position") == "MID")
    assert mae(mid_only) == pytest.approx(4.0)  # |5 - 1|


def test_spearman_perfect_rank_agreement():
    df = _results_df().filter(pl.col("position") == "FWD")
    assert spearman(df["prediction"], df["total_points"]) == pytest.approx(1.0)


def test_mae_empty_dataframe_is_nan_not_an_error():
    empty = _results_df().filter(pl.col("season") == "nonexistent")
    assert mae(empty) != mae(empty)  # NaN != NaN


def test_calibration_curve_shape():
    df = _results_df()
    curve = calibration_curve(df, n_bins=2)
    assert len(curve) <= 2
    for bucket in curve:
        assert "mean_prediction" in bucket and "mean_actual" in bucket and "n" in bucket


def test_build_report_has_per_season_and_pooled_sections():
    df = _results_df()
    report = build_report(df)
    assert report["n_rows"] == df.height
    assert "trailing_mean" in report["pooled_baseline"]
    assert "('2024-25', 'trailing_mean')" in report["per_season_baseline"]


def test_report_round_trips_through_json(tmp_path: Path):
    report = build_report(_results_df())
    out_path = tmp_path / "report.json"
    write_report(report, out_path)
    reloaded = json.loads(out_path.read_text(encoding="utf-8"))
    assert reloaded["n_rows"] == report["n_rows"]


def test_build_report_empty_input_does_not_raise():
    report = build_report(pl.DataFrame())
    assert report["n_rows"] == 0


def test_component_decomposition_mae_per_bucket():
    predicted = [{"goals": 0.4, "minutes": 1.8}, {"goals": 0.1, "minutes": 1.5}]
    actual = [{"goals": 4.0, "minutes": 2.0}, {"goals": 0.0, "minutes": 0.0}]
    result = component_decomposition_mae(predicted, actual)
    assert result["goals"] == pytest.approx((3.6 + 0.1) / 2)
    assert result["minutes"] == pytest.approx((0.2 + 1.5) / 2)


def test_component_decomposition_mae_empty_input():
    assert component_decomposition_mae([], []) == {}


def test_minutes_head_metrics_perfect_predictions_score_zero_brier():
    predicted = [{"p_blank": 1.0, "p_short": 0.0, "p_full": 0.0}, {"p_blank": 0.0, "p_short": 0.0, "p_full": 1.0}]
    actual_minutes = [0, 90]
    result = minutes_head_metrics(predicted, actual_minutes)
    assert result["brier_blank"] == pytest.approx(0.0)
    assert result["brier_full"] == pytest.approx(0.0)
    assert result["n"] == 2


def test_minutes_head_metrics_uniformly_wrong_predictions_score_maximal_brier():
    predicted = [{"p_blank": 0.0, "p_short": 0.0, "p_full": 1.0}]  # confidently predicts 90 min
    actual_minutes = [0]  # player didn't play at all
    result = minutes_head_metrics(predicted, actual_minutes)
    assert result["brier_full"] == pytest.approx(1.0)  # maximally wrong
    assert result["brier_blank"] == pytest.approx(1.0)


def test_minutes_head_metrics_empty_input():
    result = minutes_head_metrics([], [])
    assert result["n"] == 0
