"""§4.2's generalized trailing-rate/pooled-prior pattern (analytics/features.py)."""

from __future__ import annotations

import polars as pl
import pytest

from analytics.features import fill_missing_with_pooled_prior, minutes_distribution, trailing_feature, trailing_mean


def _row(element_id, gw, minutes, goals=0, position="MID", promoted=False):
    return {"element_id": element_id, "gw": gw, "minutes": minutes, "goals_scored": goals, "position": position, "is_promoted_club": promoted}


def test_trailing_mean_matches_hand_computed_value():
    df = pl.DataFrame([_row(1, gw, 90, goals=pts) for gw, pts in enumerate([0, 1, 2, 1, 0], start=1)])
    result = trailing_mean(df, upto_gw=5, window=5, value_col="goals_scored")
    assert result.filter(pl.col("element_id") == 1)["goals_scored_trailing"][0] == pytest.approx(0.8)  # mean(0,1,2,1,0)


def test_trailing_mean_window_trims_to_most_recent():
    df = pl.DataFrame([_row(1, gw, 90, goals=pts) for gw, pts in enumerate([0, 1, 2, 1, 4], start=1)])
    result = trailing_mean(df, upto_gw=5, window=2, value_col="goals_scored")
    assert result.filter(pl.col("element_id") == 1)["goals_scored_trailing"][0] == pytest.approx(2.5)  # mean(1, 4)


def test_minutes_distribution_sums_to_one_and_matches_hand_computed_rates():
    minutes = [0, 90, 45, 90, 0]  # blank, full, short, full, blank -> p_blank=0.4, p_short=0.2, p_full=0.4
    df = pl.DataFrame([_row(1, gw, m) for gw, m in enumerate(minutes, start=1)])
    result = minutes_distribution(df, upto_gw=5, window=5).filter(pl.col("element_id") == 1).row(0, named=True)
    assert result["p_blank"] == pytest.approx(0.4)
    assert result["p_short"] == pytest.approx(0.2)
    assert result["p_full"] == pytest.approx(0.4)
    assert result["p_blank"] + result["p_short"] + result["p_full"] == pytest.approx(1.0)


def test_fill_missing_with_pooled_prior_uses_promoted_club_peers():
    train_df = pl.DataFrame(
        [
            _row(10, 1, 90, goals=1, position="FWD", promoted=True),
            _row(10, 2, 90, goals=3, position="FWD", promoted=True),
            _row(20, 1, 90, goals=0, position="MID", promoted=False),
        ]
    )
    # element 99: promoted-club FWD with no trailing row of its own.
    joined = pl.DataFrame([{"element_id": 99, "position": "FWD", "is_promoted_club": True, "goals_scored_trailing": None}])
    filled = fill_missing_with_pooled_prior(joined, train_df, target_gw=3, value_col="goals_scored")
    assert filled["goals_scored_trailing"][0] == pytest.approx(2.0)  # mean(1, 3) from the promoted-club FWD peer


def test_trailing_feature_end_to_end():
    train_df = pl.DataFrame([_row(1, gw, 90, goals=pts, position="FWD") for gw, pts in enumerate([1, 1, 1], start=1)])
    roster = pl.DataFrame([{"element_id": 1, "position": "FWD", "is_promoted_club": False}])
    result = trailing_feature(train_df, roster, target_gw=4, window=5, value_col="goals_scored")
    assert result["goals_scored_trailing"][0] == pytest.approx(1.0)


# --- minutes_reliability (the panel's model column) ------------------------


def test_trailing_minutes_reliability_is_point_in_time():
    """The gameweek being described must not contribute to the rate
    describing it, or the column would be reporting the future."""
    from analytics.features import trailing_minutes_reliability

    df = pl.DataFrame({
        "season": ["2025-26"] * 4,
        "element_id": [1] * 4,
        "gw": [1, 2, 3, 4],
        "minutes": [90, 90, 0, 90],
    })

    out = trailing_minutes_reliability(df, window=3).sort("gw")

    assert out["minutes_reliability"][0] is None, "nothing precedes the first gameweek"
    assert out["minutes_reliability"][1] == pytest.approx(1.0)
    assert out["minutes_reliability"][2] == pytest.approx(1.0), "the blank has not happened yet"
    assert out["minutes_reliability"][3] == pytest.approx(2 / 3)


def test_trailing_minutes_reliability_matches_minutes_distribution():
    """It is `minutes_distribution`'s `p_full` evaluated for every row at
    once. Reused rather than redefined, so the exported column inherits
    that head's published Brier score instead of becoming a second,
    unvalidated statistic — this is what pins the two together."""
    from analytics.features import minutes_distribution, trailing_minutes_reliability

    df = pl.DataFrame({
        "season": ["2025-26"] * 6,
        "element_id": [7] * 6,
        "gw": [1, 2, 3, 4, 5, 6],
        "minutes": [90, 30, 90, 0, 75, 90],
    })

    vectorized = trailing_minutes_reliability(df, window=3).sort("gw")
    for target_gw in range(2, 7):
        reference = minutes_distribution(df, upto_gw=target_gw - 1, window=3)
        expected = reference.filter(pl.col("element_id") == 7)["p_full"][0]
        actual = vectorized.filter(pl.col("gw") == target_gw)["minutes_reliability"][0]
        assert actual == pytest.approx(expected), f"disagreed at gw{target_gw}"


def test_a_player_with_no_history_is_null_not_zero():
    """A player with no history is not a player who never plays (§5.3.3)."""
    from analytics.features import trailing_minutes_reliability

    df = pl.DataFrame({
        "season": ["2025-26"], "element_id": [1], "gw": [1], "minutes": [90],
    })

    assert trailing_minutes_reliability(df, window=3)["minutes_reliability"][0] is None


def test_the_season_boundary_resets_the_window():
    """Partitioned by season as well as player: last May's minutes say
    nothing about a new campaign, and a window straddling the boundary
    would carry them in."""
    from analytics.features import trailing_minutes_reliability

    df = pl.DataFrame({
        "season": ["2024-25", "2024-25", "2025-26"],
        "element_id": [1, 1, 1],
        "gw": [37, 38, 1],
        "minutes": [90, 90, 90],
    })

    out = trailing_minutes_reliability(df, window=3).sort(["season", "gw"])

    assert out["minutes_reliability"][2] is None, "the new season starts unknown"
