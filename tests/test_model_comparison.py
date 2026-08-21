"""§4.4: the event model, run through the identical walk-forward harness as
the three Phase 1 baselines, on all three real backtest seasons. Requires
the committed data/historical/*.parquet and the raw fixtures/teams cache
(skipped without them — see backtest/backfill.py for how to regenerate)."""

from __future__ import annotations

import pytest

from analytics.evaluate import run_comparison, run_component_decomposition
from backtest.backfill import NORMALIZED_DIR, RAW_CACHE_DIR
from backtest.report import build_report, component_decomposition_mae, minutes_head_metrics

pytestmark = pytest.mark.skipif(
    not all((NORMALIZED_DIR / f"{s}.parquet").exists() for s in ("2023-24", "2024-25", "2025-26"))
    or not all((RAW_CACHE_DIR / s / "fixtures.csv").exists() for s in ("2023-24", "2024-25", "2025-26")),
    reason="requires the committed backfilled data and raw fixtures/teams cache",
)

_BASELINE_NAMES = ("trailing_mean", "fpl_form_approx", "fixture_adjusted_trailing_mean")


@pytest.fixture(scope="module")
def comparison_report():
    results = run_comparison()
    return build_report(results)


def test_event_model_runs_without_leakage_across_all_seasons(comparison_report):
    assert comparison_report["n_rows"] > 0
    assert "event_model" in comparison_report["pooled_baseline"]


def test_event_model_beats_all_three_baselines_on_pooled_mae(comparison_report):
    pooled = comparison_report["pooled_baseline"]
    model_mae = pooled["event_model"]["mae"]
    for baseline in _BASELINE_NAMES:
        assert model_mae < pooled[baseline]["mae"], f"event_model MAE {model_mae:.4f} does not beat {baseline} MAE {pooled[baseline]['mae']:.4f}"


def test_event_model_beats_fixture_adjusted_trailing_mean_on_within_position_rank_correlation(comparison_report):
    pooled = comparison_report["pooled_baseline"]
    model_spearman = pooled["event_model"]["spearman_within_position"]["mean"]
    baseline_spearman = pooled["fixture_adjusted_trailing_mean"]["spearman_within_position"]["mean"]
    assert model_spearman > baseline_spearman


def test_minutes_head_reported_separately(comparison_report):
    """§4.4: "Minutes head evaluated separately with its own metrics" —
    error_by_event_occurrence's blank bucket, at minimum, confirms the
    generic walk-forward report already carries the minutes signal. The
    dedicated Brier-score scorecard is tested directly below."""
    pooled = comparison_report["pooled_baseline"]["event_model"]
    assert "blank_(0_minutes)" in pooled["error_by_event"]


@pytest.fixture(scope="module")
def decomposition():
    return run_component_decomposition()


def test_component_decomposition_covers_every_event_type(decomposition):
    """§3.5 / §4.4: "a decomposition of error by event type (minutes vs
    goals vs clean sheets vs bonus)" — now that the event model exists,
    this is a true predicted-vs-actual comparison per bucket, not the
    Phase 1 occurrence-based proxy."""
    result = component_decomposition_mae(decomposition["predicted_components"], decomposition["actual_components"])
    for bucket in ("minutes", "goals", "assists", "clean_sheets", "goals_conceded", "saves", "defensive_contribution", "bonus"):
        assert bucket in result
        assert result[bucket] >= 0


def test_minutes_head_has_its_own_scorecard(decomposition):
    result = minutes_head_metrics(decomposition["predicted_minutes_dist"], decomposition["actual_minutes"])
    assert result["n"] > 0
    # A working minutes head should beat a coin-flip Brier score (0.25 for
    # a constant p=0.5 guess) on the dominant blank/full buckets.
    assert result["brier_blank"] < 0.25
    assert result["brier_full"] < 0.25
