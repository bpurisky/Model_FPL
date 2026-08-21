"""§3.3: `feature.available_at < gameweek.deadline_time` for every feature,
asserted before training/evaluation. A violation raises; it does not warn.
Three deliberate leakage scenarios, each must fail loudly."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from backtest.leakage import Feature, LeakageError, assert_no_leakage

DEADLINE = datetime(2025, 3, 1, 17, 30, tzinfo=timezone.utc)


def test_valid_features_pass():
    features = [
        Feature(name="trailing_mean", element_id=1, value=4.2, available_at=DEADLINE - timedelta(days=3)),
        Feature(name="trailing_mean", element_id=2, value=1.0, available_at=DEADLINE - timedelta(hours=1)),
    ]
    assert_no_leakage(features, DEADLINE)  # must not raise


def test_deliberate_leak_using_the_target_gameweeks_own_match_data():
    """A feature 'available' exactly at the deadline — e.g. built from the
    very match it's meant to predict — must be rejected, not waved through."""
    features = [Feature(name="leaky", element_id=1, value=10.0, available_at=DEADLINE)]
    with pytest.raises(LeakageError):
        assert_no_leakage(features, DEADLINE)


def test_deliberate_leak_using_a_future_gameweek():
    """A feature built from gameweek N+2 while predicting gameweek N+1."""
    features = [Feature(name="leaky", element_id=1, value=10.0, available_at=DEADLINE + timedelta(days=7))]
    with pytest.raises(LeakageError):
        assert_no_leakage(features, DEADLINE)


def test_deliberate_leak_using_a_season_aggregate():
    """A feature built from full-season totals — necessarily includes
    gameweeks that haven't happened yet relative to an early-season deadline."""
    season_end = DEADLINE + timedelta(days=90)
    features = [Feature(name="season_total_goals", element_id=1, value=20.0, available_at=season_end)]
    with pytest.raises(LeakageError):
        assert_no_leakage(features, DEADLINE)


def test_error_message_identifies_the_offending_feature():
    features = [Feature(name="leaky_feature", element_id=42, value=1.0, available_at=DEADLINE)]
    with pytest.raises(LeakageError, match="leaky_feature"):
        assert_no_leakage(features, DEADLINE, context="gw5")
