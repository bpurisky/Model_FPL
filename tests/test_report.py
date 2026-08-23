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
    regularized_incomplete_beta,
    rmse,
    spearman,
    spearman_p_value,
    spearman_with_significance,
    spearman_within_position,
    spearman_within_position_significance,
    two_sided_t_p_value,
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


# --- significance (§5.3.2 wants rho, n and p together) --------------------


def test_regularized_incomplete_beta_matches_closed_forms():
    """I_x(a,1) = x^a and I_x(1,b) = 1-(1-x)^b are exact, so they check the
    continued fraction rather than restating it."""
    for x in (0.01, 0.25, 0.5, 0.9, 0.99):
        for a in (0.5, 2.0, 7.0, 40.0):
            assert regularized_incomplete_beta(a, 1.0, x) == pytest.approx(x**a, abs=1e-12)
            assert regularized_incomplete_beta(1.0, a, x) == pytest.approx(1 - (1 - x) ** a, abs=1e-12)
    assert regularized_incomplete_beta(5.0, 5.0, 0.5) == pytest.approx(0.5, abs=1e-12)
    # reflection: I_x(a,b) == 1 - I_{1-x}(b,a)
    assert regularized_incomplete_beta(2.0, 3.0, 0.3) == pytest.approx(
        1 - regularized_incomplete_beta(3.0, 2.0, 0.7), abs=1e-12
    )


def test_regularized_incomplete_beta_saturates_outside_the_unit_interval():
    assert regularized_incomplete_beta(2.0, 3.0, 0.0) == 0.0
    assert regularized_incomplete_beta(2.0, 3.0, 1.0) == 1.0


@pytest.mark.parametrize(
    "dof,t,p",
    [(1, 12.706, 0.05), (2, 4.303, 0.05), (10, 2.228, 0.05), (30, 2.042, 0.05), (10, 3.169, 0.01), (100, 1.984, 0.05)],
)
def test_two_sided_t_p_value_matches_published_critical_values(dof, t, p):
    """Textbook two-tailed critical values. Tolerance is 1e-4 because the
    published t values are themselves rounded to three decimals."""
    assert two_sided_t_p_value(t, dof) == pytest.approx(p, abs=1e-4)


def test_t_p_value_is_one_at_zero_and_zero_at_infinity():
    assert two_sided_t_p_value(0.0, 12) == pytest.approx(1.0)
    assert two_sided_t_p_value(float("inf"), 12) == 0.0


def test_spearman_p_value_needs_three_pairs():
    """n-2 degrees of freedom: at n=2 there is no t distribution to read,
    and a rho of 1.0 over two points means nothing."""
    assert spearman_p_value(1.0, 2) != spearman_p_value(1.0, 2)  # NaN
    assert spearman_p_value(float("nan"), 50) != spearman_p_value(float("nan"), 50)


def test_spearman_p_value_at_the_extremes():
    assert spearman_p_value(1.0, 50) == 0.0  # t is infinite, not an error
    assert spearman_p_value(-1.0, 50) == 0.0  # sign-independent: two-sided
    assert spearman_p_value(0.0, 50) == pytest.approx(1.0)


def test_spearman_with_significance_reports_n_alongside_rho():
    """rho on its own cannot be read -- 0.6 over 8 rows and 0.6 over 8,000
    are different claims. That is the whole reason n is in the payload."""
    df = _results_df().filter(pl.col("position") == "FWD")

    result = spearman_with_significance(df["prediction"], df["total_points"])

    assert result["rho"] == pytest.approx(1.0)
    assert result["n"] == 3
    assert result["p_value"] == 0.0


def test_spearman_significance_carries_no_mean_row():
    """Its rho-only sibling means across positions, which is a readable
    summary but not a sampling statistic -- so there is deliberately no
    p-value beside it."""
    df = _results_df()

    rho_only = spearman_within_position(df)
    with_sig = spearman_within_position_significance(df)

    assert "mean" in rho_only
    assert "mean" not in with_sig
    assert set(with_sig) == {"FWD", "MID"}
    assert set(with_sig["FWD"]) == {"rho", "n", "p_value"}


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
