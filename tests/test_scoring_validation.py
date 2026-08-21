"""§4.1 acceptance: replaying a completed historical gameweek under its own
season's config must reproduce official points for >=95% of players.
Requires the committed data/historical/*.parquet (skipped without it)."""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from analytics.scoring import load_scoring_config, validate_against_actual
from backtest.backfill import NORMALIZED_DIR

CONFIG_2024_25 = load_scoring_config(Path("config/scoring_2024_25.yaml"))
CONFIG_2025_26 = load_scoring_config(Path("config/scoring_2025_26.yaml"))

pytestmark = pytest.mark.skipif(
    not (NORMALIZED_DIR / "2024-25.parquet").exists() or not (NORMALIZED_DIR / "2025-26.parquet").exists(),
    reason="requires the committed backfilled data",
)


def test_scoring_reproduces_2024_25_gw10_official_points():
    df = pl.read_parquet(NORMALIZED_DIR / "2024-25.parquet").filter(pl.col("gw") == 10)
    report = validate_against_actual(df, CONFIG_2024_25)
    assert report.match_rate >= 0.95, f"only {report.match_rate:.1%} match; sample discrepancies:\n{report.discrepancies.head(20)}"


def test_scoring_reproduces_2025_26_gw10_official_points():
    """2025-26 additionally exercises the defensive contribution rule
    (introduced this season) as part of the reproduction check."""
    df = pl.read_parquet(NORMALIZED_DIR / "2025-26.parquet").filter(pl.col("gw") == 10)
    report = validate_against_actual(df, CONFIG_2025_26)
    assert report.match_rate >= 0.95, f"only {report.match_rate:.1%} match; sample discrepancies:\n{report.discrepancies.head(20)}"


def test_scoring_reproduces_full_2024_25_season():
    """Not just one gameweek — the whole season, for a tighter signal on
    where any systematic discrepancy actually lives."""
    df = pl.read_parquet(NORMALIZED_DIR / "2024-25.parquet")
    report = validate_against_actual(df, CONFIG_2024_25)
    assert report.match_rate >= 0.95, f"only {report.match_rate:.1%} match across the full season"


def test_scoring_reproduces_full_2025_26_season():
    df = pl.read_parquet(NORMALIZED_DIR / "2025-26.parquet")
    report = validate_against_actual(df, CONFIG_2025_26)
    assert report.match_rate >= 0.95, f"only {report.match_rate:.1%} match across the full season"
