"""§3.4: the three baselines, plus the promoted-club prior fallback (§3.2)
that all three share for players with no trailing history."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import polars as pl
import pytest

from backtest.baselines import (
    fixture_adjusted_trailing_mean_baseline,
    fpl_form_approx_baseline,
    trailing_mean_baseline,
)

KICKOFF = datetime(2025, 1, 1, tzinfo=timezone.utc)


def _train_row(element_id, gw, points, position="MID", promoted=False, days_offset=0):
    return {
        "element_id": element_id,
        "gw": gw,
        "total_points": points,
        "position": position,
        "is_promoted_club": promoted,
        "kickoff_time": KICKOFF + timedelta(days=days_offset),
    }


def _roster_row(element_id, position="MID", promoted=False, difficulty=3):
    return {
        "element_id": element_id,
        "position": position,
        "team": "Whatever",
        "is_promoted_club": promoted,
        "opponent_difficulty": difficulty,
        "kickoff_time": KICKOFF,
    }


def test_trailing_mean_uses_last_window_appearances():
    train_df = pl.DataFrame([_train_row(1, gw, pts, days_offset=gw) for gw, pts in enumerate([2, 4, 6, 8, 10], start=1)])
    roster = pl.DataFrame([_roster_row(1)])
    result = trailing_mean_baseline(train_df, roster, target_gw=6, window=5)
    assert result.filter(pl.col("element_id") == 1)["prediction"][0] == pytest.approx(6.0)


def test_trailing_mean_window_trims_to_most_recent():
    train_df = pl.DataFrame([_train_row(1, gw, pts, days_offset=gw) for gw, pts in enumerate([2, 4, 6, 8, 10], start=1)])
    roster = pl.DataFrame([_roster_row(1)])
    result = trailing_mean_baseline(train_df, roster, target_gw=6, window=3)
    assert result.filter(pl.col("element_id") == 1)["prediction"][0] == pytest.approx(8.0)  # mean(6, 8, 10)


def test_fpl_form_approx_excludes_matches_older_than_30_days():
    train_df = pl.DataFrame(
        [
            _train_row(1, 1, 100, days_offset=0),   # 40 days before the last match: excluded
            _train_row(1, 2, 2, days_offset=35),
            _train_row(1, 3, 4, days_offset=39),
            _train_row(1, 4, 6, days_offset=40),    # most recent -> defines the as_of anchor
        ]
    )
    roster = pl.DataFrame([_roster_row(1)])
    result = fpl_form_approx_baseline(train_df, roster, target_gw=5, window_days=30)
    # window is (40-30, 40] days offset => only gw2 (35) and gw3 (39) qualify; gw4 itself
    # is part of history since target_gw=5 filters gw<5, so gw4 (offset 40) qualifies too.
    assert result.filter(pl.col("element_id") == 1)["prediction"][0] == pytest.approx((2 + 4 + 6) / 3)


def test_fixture_adjusted_scales_up_for_easy_fixture_down_for_hard():
    train_df = pl.DataFrame([_train_row(1, gw, 6, days_offset=gw) for gw in range(1, 6)])
    easy_roster = pl.DataFrame([_roster_row(1, difficulty=1)])
    hard_roster = pl.DataFrame([_roster_row(1, difficulty=5)])
    neutral_roster = pl.DataFrame([_roster_row(1, difficulty=3)])

    easy = fixture_adjusted_trailing_mean_baseline(train_df, easy_roster, target_gw=6)["prediction"][0]
    hard = fixture_adjusted_trailing_mean_baseline(train_df, hard_roster, target_gw=6)["prediction"][0]
    neutral = fixture_adjusted_trailing_mean_baseline(train_df, neutral_roster, target_gw=6)["prediction"][0]

    assert easy > neutral > hard
    assert neutral == pytest.approx(6.0)  # difficulty 3 is a no-op scale


def test_promoted_club_player_with_no_history_gets_pooled_prior_not_null_or_zero():
    train_df = pl.DataFrame(
        [
            _train_row(10, 1, 8, position="FWD", promoted=True, days_offset=1),
            _train_row(10, 2, 4, position="FWD", promoted=True, days_offset=2),
            _train_row(20, 1, 2, position="MID", promoted=False, days_offset=1),  # different position, shouldn't pollute
        ]
    )
    # element 99: a promoted-club FWD with zero appearances so far (e.g. only just broke into the XI).
    roster = pl.DataFrame([_roster_row(99, position="FWD", promoted=True)])
    result = trailing_mean_baseline(train_df, roster, target_gw=3, window=5)
    prediction = result.filter(pl.col("element_id") == 99)["prediction"][0]
    assert prediction == pytest.approx((8 + 4) / 2)  # pooled from the promoted-club FWD peer, not 0 and not null
