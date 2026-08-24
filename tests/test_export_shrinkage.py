"""`shrinkage.json` — §5.4.7's plateau panel.

The expensive part of this exporter is the sweep itself: one walk-forward
per shrinkage value. So the tests here drive `build_shrinkage` over a
single small season frame rather than the real archive, and check the
things that would be wrong regardless of the data — that the sweep varies
what it says it varies, that the bars are the export's verdict rather than
the surface's, and that the default is not quietly moved.

The one test that does use real numbers is the last, and it is the point
of the whole file: the panel must not be able to render a plateau that
flatters the constant.
"""

from __future__ import annotations

import json
import io
from pathlib import Path

import polars as pl
import pytest

from analytics.projections import (
    GOALS_CONCEDED_SHRINKAGE,
    expected_points_by_component,
    expected_points_from_projection,
)
from web.export.shrinkage import FOCUS_POSITION, SWEEP


def test_the_default_shrinkage_has_not_moved():
    """Every number elsewhere in the repo was measured at this value.

    The parameter added for the sweep is only safe because the default is
    untouched; a changed default would silently restate the scorecard,
    the board and the paper trade.
    """
    assert GOALS_CONCEDED_SHRINKAGE == 0.7


def _row() -> dict:
    """A defender who concedes, with everything else zeroed so the only
    term that can move is the one under test."""
    return {
        "position": "DEF",
        "custom_difficulty": 3.0,
        "p_blank": 0.0,
        "p_short": 0.0,
        "p_full": 1.0,
        "goals_scored_trailing": 0.0,
        "assists_trailing": 0.0,
        "clean_sheets_trailing": 0.0,
        "goals_conceded_trailing": 2.0,
        "own_goals_trailing": 0.0,
        "penalties_missed_trailing": 0.0,
        "penalties_saved_trailing": 0.0,
        "yellow_cards_trailing": 0.0,
        "red_cards_trailing": 0.0,
        "saves_trailing": 0.0,
        "bonus_trailing": 0.0,
        "p_dc_threshold_trailing": 0.0,
    }


def test_shrinkage_scales_only_the_goals_conceded_term():
    from analytics.scoring import load_scoring_config

    config = load_scoring_config(Path("config/scoring_2025_26.yaml"))
    full = expected_points_by_component(_row(), config, 1.0)
    half = expected_points_by_component(_row(), config, 0.5)

    assert half["goals_conceded"] == pytest.approx(full["goals_conceded"] * 0.5)
    for head, value in full.items():
        if head != "goals_conceded":
            assert half[head] == pytest.approx(value), f"{head} moved and should not have"


def test_zero_shrinkage_removes_the_term_entirely():
    from analytics.scoring import load_scoring_config

    config = load_scoring_config(Path("config/scoring_2025_26.yaml"))
    assert expected_points_by_component(_row(), config, 0.0)["goals_conceded"] == 0.0


def test_the_parameter_defaults_to_the_published_constant():
    """A caller that does not name a shrinkage must get the shipped model.

    This is what lets `project_points` grow a parameter without changing
    any published number.
    """
    from analytics.scoring import load_scoring_config

    config = load_scoring_config(Path("config/scoring_2025_26.yaml"))
    assert expected_points_from_projection(
        _row(), config
    ) == pytest.approx(expected_points_from_projection(_row(), config, GOALS_CONCEDED_SHRINKAGE))


def test_the_sweep_covers_both_endpoints_and_the_claimed_plateau():
    """The argument for 0.7 is that neither endpoint works, so a sweep
    that did not reach them would not be able to make it."""
    assert SWEEP[0] == 0.0
    assert SWEEP[-1] == 1.0
    for claimed in (0.6, 0.7, 0.85):
        assert claimed in SWEEP, f"{claimed} is named in the ablation comment"


def test_focus_position_is_where_the_ablation_says_the_damage_is():
    assert FOCUS_POSITION == "DEF"


@pytest.mark.skipif(
    not Path("data/web/v1/shrinkage.json").exists(),
    reason="shrinkage.json is built by `python -m web.export shrinkage`",
)
def test_the_committed_sweep_reports_both_readings_and_they_disagree():
    """The point of the panel, asserted against the committed file.

    §4.4's criterion as written is the mean within-position rank
    correlation, and on that reading the shipped 0.7 clears both bars. The
    ablation comment in `analytics/projections.py` frames the same trade
    on DEF alone, and on *that* reading 0.7 is past the range. Both are
    true of different measurements, and a panel that rendered only the
    flattering one would be choosing a measurement to suit a constant.

    If this ever stops failing to agree — if the two readings converge —
    the panel's copy needs rewriting, so the disagreement is asserted
    rather than assumed.
    """
    file = json.load(io.open("data/web/v1/shrinkage.json", encoding="utf-8"))
    points = {p["shrinkage"]: p for p in file["points"]}
    default = points[file["default"]]

    assert default["beats_mae_bar"], "the shipped default must clear the MAE bar"
    assert default["beats_spearman_bar"], "and §4.4's criterion as stated"
    assert not default["beats_focus_bar"], (
        "the DEF-only reading is the stricter one and 0.7 sits past it; "
        "if that changed, the panel copy is now wrong"
    )

    stated = [s for s, p in points.items() if p["beats_mae_bar"] and p["beats_spearman_bar"]]
    focus = [s for s, p in points.items() if p["beats_mae_bar"] and p["beats_focus_bar"]]
    assert len(stated) > len(focus), "the DEF reading must be the stricter one"

    # Monotone in both directions is what makes it a trade rather than an
    # optimum, and it is the sentence the panel leads with.
    maes = [points[s]["mae"] for s in sorted(points)]
    focuses = [points[s]["spearman_focus"] for s in sorted(points)]
    assert maes == sorted(maes, reverse=True), "MAE should improve monotonically"
    assert focuses == sorted(focuses, reverse=True), "DEF rho should degrade monotonically"


def test_build_shrinkage_runs_over_an_injected_frame():
    """The whole exporter, over two gameweeks, so the wiring is covered
    without paying for ten walk-forwards."""
    from web.export.shrinkage import build_shrinkage

    from backtest.backfill import NORMALIZED_DIR

    frame = pl.read_parquet(NORMALIZED_DIR / "2025-26.parquet").filter(pl.col("gw") <= 3)
    file = build_shrinkage(sweep=(0.0, 1.0), season_dfs={"2025-26": frame})

    assert len(file.points) == 2
    assert file.default == GOALS_CONCEDED_SHRINKAGE
    assert file.focus_position == FOCUS_POSITION
    # Raising the weight must move the MAE, or the sweep is measuring
    # nothing.
    assert file.points[0].mae != file.points[1].mae
