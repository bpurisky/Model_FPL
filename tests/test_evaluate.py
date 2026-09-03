"""§6.3-6.5 evaluation. `fetch_real_gw_points` is live/async and exercised
by actually running it, not unit-tested here."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import polars as pl
import pytest

from analytics.price_model import PriceModelEvaluation
from papertrade.actuals import _SCHEMA
from papertrade.evaluate import (
    collect_freeze_provenance,
    evaluate_gw_player_level,
    evaluate_player_level,
    evaluate_squad_level,
    freeze_degeneracy,
    launch_gate_report,
    projection_degeneracy,
)
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


def test_launch_gate_report_reports_baseline_gap_as_soon_as_any_live_data_exists():
    """The gate has no fixed gameweek minimum any more -- one real gameweek
    is enough for a criterion to report its real answer. The baseline
    criteria specifically never reach PASS regardless of gameweek count
    (backtest/baselines.py isn't wired to live data yet), but they still
    move off "insufficient data" the moment there is any live evidence."""
    report = launch_gate_report({2: {}}, {"n_gameweeks": 1})

    assert report["ready_to_launch"] is False
    assert report["gameweeks_evaluated"] == 1
    assert report["criteria"]["beats_fixture_adjusted_trailing_mean_mae"]["status"] == "not wired to live baselines yet"


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


# --------------------------------------------------------------------------
# degeneracy guard (§6.5): the automated version of the gw2 judgement
# --------------------------------------------------------------------------


def test_projection_degeneracy_flags_an_all_identical_freeze():
    """The gw2 shape: every player assigned the same number."""
    result = projection_degeneracy({str(eid): 0.8 for eid in range(1, 601)})

    assert result["is_degenerate"] is True
    assert result["n"] == 600
    assert result["n_distinct"] == 1
    assert result["variance"] == 0.0
    assert result["modal_value"] == 0.8
    assert result["modal_share"] == 1.0
    assert "identical" in result["reason"]


def test_projection_degeneracy_passes_a_freeze_with_real_spread():
    result = projection_degeneracy({str(eid): float(eid % 9) for eid in range(1, 601)})

    assert result["is_degenerate"] is False
    assert result["is_near_degenerate"] is False
    assert result["variance"] > 0
    assert result["reason"] == ""


def test_projection_degeneracy_reports_but_does_not_exclude_a_near_degenerate_freeze():
    """97% on the pooled prior is the same failure at less than full
    strength. It is surfaced, not excluded -- there is no evidence behind
    any particular cutoff, and dropping partial signal is the worse error.
    """
    projections = {str(eid): 0.8 for eid in range(1, 583)}
    projections.update({str(eid): float(eid % 7) + 1.5 for eid in range(583, 601)})

    result = projection_degeneracy(projections)

    assert result["is_degenerate"] is False  # not excluded
    assert result["is_near_degenerate"] is True  # but flagged
    assert result["modal_share"] >= 0.95


def test_projection_degeneracy_treats_missing_projections_as_missing_not_degenerate():
    """"No projections recorded" and "projections carrying no signal" are
    different facts; only the second licenses throwing a gameweek away."""
    result = projection_degeneracy({})

    assert result["is_missing"] is True
    assert result["is_degenerate"] is False


def test_the_real_gw2_freeze_is_detected_as_degenerate():
    """Not a synthetic case: a real freeze holding 0.8 for all 600
    players, written 2026-08-21T21:03Z -- seven days before its own
    deadline and three hours after gw1 kicked off, so the model had
    zero gameweeks of 2026-27 to train on.

    It lived at papertrade/freezes/gw2.json until it was moved here.
    Leaving it there would have made write_freeze raise FileExistsError
    inside gw2's real window, which cmd_freeze swallows as expected
    steady state -- the gameweek would have been lost behind a green
    build. Archived rather than deleted because it is the observation
    the guard exists to have caught automatically, and a synthetic
    fixture cannot corroborate that."""
    freezes_dir = Path(__file__).parent / "fixtures" / "degenerate_freeze"

    result = freeze_degeneracy(2, freezes_dir=freezes_dir)

    assert result["is_degenerate"] is True
    assert result["n_distinct"] == 1
    assert result["variance"] == 0.0


def _degenerate_freeze(gw: int, n: int = 20) -> dict:
    return {"projections": {str(gw): {str(eid): 0.8 for eid in range(1, n + 1)}}}


def _real_freeze(gw: int, n: int = 20) -> dict:
    return {"projections": {str(gw): {str(eid): float(eid % 5) + 0.5 for eid in range(1, n + 1)}}}


def test_evaluate_player_level_excludes_a_degenerate_gameweek(tmp_path):
    write_freeze(2, _degenerate_freeze(2), freezes_dir=tmp_path)
    write_freeze(3, _real_freeze(3), freezes_dir=tmp_path)
    actuals = pl.DataFrame(
        [_actual_row(gw, eid, "MID", eid % 6) for gw in (2, 3) for eid in range(1, 21)], schema=_SCHEMA
    )

    result = evaluate_player_level([2, 3], freezes_dir=tmp_path, actuals=actuals)

    assert sorted(result["included"]) == [3]
    assert [e["gw"] for e in result["excluded"]] == [2]
    assert result["skipped"] == []


def test_evaluate_player_level_separates_skipped_from_excluded(tmp_path):
    """A gameweek with no freeze at all is skipped, not excluded: nothing
    was predicted, as opposed to something predicted that said nothing."""
    write_freeze(2, _degenerate_freeze(2), freezes_dir=tmp_path)
    actuals = pl.DataFrame([_actual_row(2, eid, "MID", 4) for eid in range(1, 21)], schema=_SCHEMA)

    result = evaluate_player_level([2, 9], freezes_dir=tmp_path, actuals=actuals)

    assert [e["gw"] for e in result["excluded"]] == [2]
    assert [s["gw"] for s in result["skipped"]] == [9]


def test_excluded_gameweeks_do_not_count_toward_the_evaluated_total(tmp_path):
    """§6.5's gate counts live evidence. A null observation is not
    evidence, and must not shrink the denominator silently either."""
    squad_eval = {"n_gameweeks": 0}
    excluded = [{"gw": 2, "reason": "all 600 projections are effectively identical"}]

    gate = launch_gate_report({3: {"mae": 1.0}}, squad_eval, excluded_gws=excluded)

    assert gate["gameweeks_evaluated"] == 1  # gw3 only, not gw2
    assert gate["gameweeks_excluded"] == excluded
    detail = gate["criteria"]["beats_fixture_adjusted_trailing_mean_mae"]["detail"]
    assert "gw2" in detail and "1 gameweek evaluated" in detail


def test_evaluate_squad_level_excludes_a_degenerate_gameweek(tmp_path):
    """The shadow XI and captain for a degenerate gameweek were chosen by
    optimizing against one identical number, so the realized points
    measure tie-breaking, not the model."""
    state = squad_state_to_dict(_squad_state())
    write_freeze(2, {**_degenerate_freeze(2), "shadow_state_after": state}, freezes_dir=tmp_path)
    write_freeze(3, {**_real_freeze(3), "shadow_state_after": state}, freezes_dir=tmp_path)
    actuals = pl.DataFrame(
        [_actual_row(gw, eid, "MID", 4 if eid != 1 else 10) for gw in (2, 3) for eid in range(1, 16)],
        schema=_SCHEMA,
    )

    result = evaluate_squad_level({2: 55, 3: 55}, actuals=actuals, freezes_dir=tmp_path)

    assert [row["gw"] for row in result["per_gw"]] == [3]
    assert [e["gw"] for e in result["excluded_gameweeks"]] == [2]
    assert result["n_gameweeks"] == 1


# --------------------------------------------------------------------------
# §6.5 criteria 3 and 4, read off the freezes themselves
# --------------------------------------------------------------------------


def _provenanced_freeze(gw: int, manual_correction=None, leakage=True) -> dict:
    payload = {**_real_freeze(gw), "model_git_sha": "a1b2c3d", "manual_correction": manual_correction}
    if leakage:
        payload["leakage_check"] = {"ran": True, "passed": True, "n_features": 600}
    return payload


def test_collect_freeze_provenance_reads_the_recorded_fields(tmp_path):
    write_freeze(3, _provenanced_freeze(3), freezes_dir=tmp_path)

    provenance = collect_freeze_provenance([3], freezes_dir=tmp_path)

    assert provenance[0]["leakage_verified"] is True
    assert provenance[0]["manual_correction"] is None
    assert provenance[0]["records_manual_correction_field"] is True
    assert provenance[0]["model_git_sha"] == "a1b2c3d"


def test_leakage_criterion_is_not_tracked_for_a_freeze_that_predates_the_check(tmp_path):
    """gw2 was frozen before the assertion was wired in. It cannot be
    verified retroactively — the claim is about what was available before
    a deadline that has already passed."""
    write_freeze(2, _real_freeze(2), freezes_dir=tmp_path)  # no leakage_check field
    provenance = collect_freeze_provenance([2], freezes_dir=tmp_path)

    gate = launch_gate_report({}, {"n_gameweeks": 0}, freeze_provenance=provenance)

    criterion = gate["criteria"]["no_leakage_assertion_fired"]
    assert criterion["status"] == "not tracked"
    assert "gw2" in criterion["detail"]
    assert "retroactively" in criterion["detail"]


def test_leakage_criterion_passes_as_soon_as_a_single_gameweek_is_clean(tmp_path):
    """No fixed gameweek minimum any more -- one verified freeze is enough
    for this criterion to report PASS live."""
    write_freeze(3, _provenanced_freeze(3), freezes_dir=tmp_path)
    provenance = collect_freeze_provenance([3], freezes_dir=tmp_path)

    gate = launch_gate_report({3: {}}, {"n_gameweeks": 1}, freeze_provenance=provenance)

    criterion = gate["criteria"]["no_leakage_assertion_fired"]
    assert criterion["status"] == "PASS"
    assert "ran and passed" in criterion["detail"]


def test_manual_correction_criterion_fails_when_a_correction_is_declared(tmp_path):
    write_freeze(3, _provenanced_freeze(3, manual_correction="rebuilt shadow squad by hand"), freezes_dir=tmp_path)
    provenance = collect_freeze_provenance([3], freezes_dir=tmp_path)

    gate = launch_gate_report({3: {}}, {"n_gameweeks": 1}, freeze_provenance=provenance)

    criterion = gate["criteria"]["squad_reconstruction_ran_without_manual_correction"]
    assert criterion["status"] == "FAIL"
    assert "rebuilt shadow squad by hand" in criterion["detail"]


def test_manual_correction_criterion_untracked_for_a_freeze_without_the_field(tmp_path):
    """A freeze predating the field makes no claim either way, and is not
    assumed clean."""
    write_freeze(2, _real_freeze(2), freezes_dir=tmp_path)
    provenance = collect_freeze_provenance([2], freezes_dir=tmp_path)

    gate = launch_gate_report({}, {"n_gameweeks": 1}, freeze_provenance=provenance)

    criterion = gate["criteria"]["squad_reconstruction_ran_without_manual_correction"]
    assert criterion["status"] == "not tracked"
    assert "immutable" in criterion["detail"]


def test_manual_correction_criterion_passes_as_soon_as_a_clean_gameweek_exists(tmp_path):
    """No fixed gameweek minimum any more -- one freeze that declares no
    manual correction is enough for this criterion to report PASS live."""
    write_freeze(3, _provenanced_freeze(3), freezes_dir=tmp_path)
    provenance = collect_freeze_provenance([3], freezes_dir=tmp_path)

    gate = launch_gate_report({3: {}}, {"n_gameweeks": 1}, freeze_provenance=provenance)

    criterion = gate["criteria"]["squad_reconstruction_ran_without_manual_correction"]
    assert criterion["status"] == "PASS"
    assert "declare no manual correction" in criterion["detail"]


def test_gate_never_fabricates_a_pass_from_absent_provenance():
    """No freezes at all must not read as "nothing went wrong"."""
    gate = launch_gate_report({}, {"n_gameweeks": 0}, freeze_provenance=[])

    assert gate["criteria"]["no_leakage_assertion_fired"]["status"] == "insufficient data"
    assert gate["criteria"]["squad_reconstruction_ran_without_manual_correction"]["status"] == "insufficient data"
    assert gate["ready_to_launch"] is False
