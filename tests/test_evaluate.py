"""§6.3-6.5 evaluation. `fetch_real_gw_points` is live/async and exercised
by actually running it, not unit-tested here."""

from __future__ import annotations

from datetime import datetime, timezone

import polars as pl
import pytest

from analytics.price_model import PriceModelEvaluation
from papertrade.actuals import _SCHEMA
from papertrade.evaluate import evaluate_gw_player_level, evaluate_squad_level, launch_gate_report
from papertrade.freeze import write_freeze
from squad.live import FLOAT_STAT_COLUMNS, INT_STAT_COLUMNS
from squad.reconstruct import SquadPlayer, SquadState, squad_state_to_dict

AS_OF = datetime(2026, 8, 28, tzinfo=timezone.utc)


def _actual_row(gw: int, eid: int, position: str, total_points: int) -> dict:
    # Built from the column lists, not longhand -- see tests/test_actuals.py:_row
    # for why a literal here goes silently null when the schema widens.
    return {
        "gw": gw, "element_id": eid, "position": position, "team": "Team A", "is_promoted_club": False,
        **{col: 0 for col in INT_STAT_COLUMNS},
        **{col: 0.0 for col in FLOAT_STAT_COLUMNS},
        **{"minutes": 90, "total_points": total_points},
    }


def test_evaluate_gw_player_level_computes_mae(tmp_path):
    write_freeze(2, {"projections": {"2": {"101": 5.0, "102": 3.0}}}, freezes_dir=tmp_path)
    actuals = pl.DataFrame([_actual_row(2, 101, "MID", 7), _actual_row(2, 102, "DEF", 3)], schema=_SCHEMA)

    result = evaluate_gw_player_level(2, freezes_dir=tmp_path, actuals=actuals)

    assert result["n"] == 2
    assert result["mae"] == pytest.approx((abs(5.0 - 7) + abs(3.0 - 3)) / 2)


def test_evaluate_gw_player_level_raises_without_actuals(tmp_path):
    write_freeze(2, {"projections": {"2": {"101": 5.0}}}, freezes_dir=tmp_path)
    empty_actuals = pl.DataFrame(schema=_SCHEMA)

    with pytest.raises(ValueError, match="gw2"):
        evaluate_gw_player_level(2, freezes_dir=tmp_path, actuals=empty_actuals)


def _squad_state(bank: int = 0) -> SquadState:
    players = [
        SquadPlayer(element_id=eid, purchase_price=50, selling_price=50, squad_position=eid,
                    multiplier=(2 if eid == 1 else (1 if eid <= 11 else 0)), is_captain=eid == 1, is_vice_captain=eid == 2)
        for eid in range(1, 16)
    ]
    return SquadState(as_of=AS_OF, players=tuple(players), bank=bank)


def test_evaluate_squad_level_compares_real_and_shadow(tmp_path):
    write_freeze(2, {"shadow_state_after": squad_state_to_dict(_squad_state())}, freezes_dir=tmp_path)
    actuals = pl.DataFrame(
        [_actual_row(2, eid, "MID", 4 if eid != 1 else 10) for eid in range(1, 16)], schema=_SCHEMA
    )

    result = evaluate_squad_level({2: 55}, actuals=actuals, freezes_dir=tmp_path)

    # starters 2-11 (4 each = 40) + captain 1 (10*2=20) = 60; bench 12-15 contribute 0
    assert result["per_gw"] == [{"gw": 2, "real_points": 55, "shadow_points": 60}]
    assert result["cumulative_real_points"] == 55
    assert result["cumulative_shadow_points"] == 60
    assert result["shadow_minus_real"] == 5
    assert "variance" in result["warning"].lower()


def test_evaluate_squad_level_skips_gameweeks_missing_a_freeze_or_actuals(tmp_path):
    # gw2 has a freeze but no actuals; gw3 has neither
    write_freeze(2, {"shadow_state_after": squad_state_to_dict(_squad_state())}, freezes_dir=tmp_path)

    result = evaluate_squad_level({2: 55, 3: 60}, actuals=pl.DataFrame(schema=_SCHEMA), freezes_dir=tmp_path)

    assert result["n_gameweeks"] == 0
    assert result["per_gw"] == []


def test_launch_gate_report_insufficient_data_before_13_gameweeks():
    report = launch_gate_report({2: {}}, {"n_gameweeks": 1})

    assert report["ready_to_launch"] is False
    assert report["gameweeks_evaluated"] == 1
    assert report["criteria"]["beats_fixture_adjusted_trailing_mean_mae"]["status"] == "insufficient data"


def test_launch_gate_report_never_fabricates_a_pass_with_no_data():
    report = launch_gate_report({}, {"n_gameweeks": 0})

    assert report["ready_to_launch"] is False
    assert all(c["status"] != "PASS" for c in report["criteria"].values())
    # no PriceModelEvaluation was passed in -> treated as zero moves predicted, never a fabricated pass
    assert report["criteria"]["price_change_model_reports_hit_rate_with_ci"]["status"] == "insufficient data"


def test_launch_gate_report_price_model_pass_when_hit_rate_above_half():
    price_eval = PriceModelEvaluation(n=10, n_moves_predicted=6, hit_rate=0.8, ci_low=0.5, ci_high=0.95)
    report = launch_gate_report({}, {"n_gameweeks": 0}, price_eval=price_eval)
    assert report["criteria"]["price_change_model_reports_hit_rate_with_ci"]["status"] == "PASS"


def test_launch_gate_report_price_model_fail_when_hit_rate_at_or_below_half():
    price_eval = PriceModelEvaluation(n=10, n_moves_predicted=6, hit_rate=0.33, ci_low=0.1, ci_high=0.6)
    report = launch_gate_report({}, {"n_gameweeks": 0}, price_eval=price_eval)
    assert report["criteria"]["price_change_model_reports_hit_rate_with_ci"]["status"] == "FAIL"
