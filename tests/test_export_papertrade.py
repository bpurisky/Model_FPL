"""§5.11.1/§5.16 D14 over `papertrade.json`: the empty state that is the
real state today, one real evaluated gameweek, degenerate exclusion, and
the contract round-trip."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import polars as pl

from analytics.price_model import PriceModelEvaluation
from papertrade.actuals import _SCHEMA
from papertrade.freeze import write_freeze
from squad.reconstruct import SquadPlayer, SquadState, squad_state_to_dict
from web.export.contract import PaperTradeFile
from web.export.papertrade import build_papertrade

AS_OF = datetime(2026, 8, 28, tzinfo=timezone.utc)
_ZERO_PRICE_EVAL = PriceModelEvaluation(n=0, n_moves_predicted=0, hit_rate=None, ci_low=None, ci_high=None)


def _actual_row(gw: int, eid: int, position: str, total_points: int) -> dict:
    from squad.live import FLOAT_STAT_COLUMNS, INT_STAT_COLUMNS

    return {
        "gw": gw, "element_id": eid, "position": position, "team": "Team A", "is_promoted_club": False,
        **{col: 0 for col in INT_STAT_COLUMNS},
        **{col: 0.0 for col in FLOAT_STAT_COLUMNS},
        **{"minutes": 90, "total_points": total_points},
    }


def _squad_state() -> SquadState:
    players = [
        SquadPlayer(element_id=eid, purchase_price=50, selling_price=50, squad_position=eid,
                    multiplier=(2 if eid == 1 else (1 if eid <= 11 else 0)), is_captain=eid == 1, is_vice_captain=eid == 2)
        for eid in range(1, 16)
    ]
    return SquadState(as_of=AS_OF, players=tuple(players), bank=0)


def _real_freeze(gw: int, n: int = 20) -> dict:
    return {"projections": {str(gw): {str(eid): float(eid % 5) + 0.5 for eid in range(1, n + 1)}}}


def _degenerate_freeze(gw: int, n: int = 20) -> dict:
    return {"projections": {str(gw): {str(eid): 0.8 for eid in range(1, n + 1)}}}


def test_the_empty_state_is_honest_not_an_error(tmp_path):
    """Today's real state: no freezes, no actuals. Every field degrades to
    zero/empty/"insufficient data" rather than the build failing."""
    file = build_papertrade(
        real_points_by_gw={}, freezes_dir=tmp_path, actuals=pl.DataFrame(schema=_SCHEMA), price_eval=_ZERO_PRICE_EVAL,
    )

    assert file.player_level == []
    assert file.squad_level.n_gameweeks == 0
    assert file.squad_level.per_gw == []
    assert file.launch_gate.ready_to_launch is False
    assert file.launch_gate.gameweeks_evaluated == 0
    for criterion in file.launch_gate.criteria.values():
        assert criterion.status in ("insufficient data", "not tracked")
    assert file.header.source_gameweek is None
    assert file.header.rows == 0


def test_one_real_gameweek_populates_both_levels(tmp_path):
    write_freeze(3, {**_real_freeze(3), "shadow_state_after": squad_state_to_dict(_squad_state())}, freezes_dir=tmp_path)
    actuals = pl.DataFrame(
        [_actual_row(3, eid, "MID", 4 if eid != 1 else 10) for eid in range(1, 21)], schema=_SCHEMA
    )

    file = build_papertrade(
        real_points_by_gw={3: 50}, freezes_dir=tmp_path, actuals=actuals, price_eval=_ZERO_PRICE_EVAL,
    )

    assert [row.gw for row in file.player_level] == [3]
    assert file.player_level[0].n == 20
    assert file.squad_level.n_gameweeks == 1
    assert file.squad_level.per_gw[0].real_points == 50
    assert file.header.source_gameweek == 3
    assert file.header.rows == 1


def test_a_degenerate_freeze_is_excluded_from_both_levels(tmp_path):
    state = squad_state_to_dict(_squad_state())
    write_freeze(2, {**_degenerate_freeze(2), "shadow_state_after": state}, freezes_dir=tmp_path)
    write_freeze(3, {**_real_freeze(3), "shadow_state_after": state}, freezes_dir=tmp_path)
    actuals = pl.DataFrame(
        [_actual_row(gw, eid, "MID", 4 if eid != 1 else 10) for gw in (2, 3) for eid in range(1, 21)],
        schema=_SCHEMA,
    )

    file = build_papertrade(
        real_points_by_gw={2: 55, 3: 55}, freezes_dir=tmp_path, actuals=actuals, price_eval=_ZERO_PRICE_EVAL,
    )

    assert [row.gw for row in file.player_level] == [3]
    assert [note.gw for note in file.player_level_excluded] == [2]
    assert [row.gw for row in file.squad_level.per_gw] == [3]
    assert [note.gw for note in file.squad_level.excluded_gameweeks] == [2]


def test_the_two_shas_are_kept_distinct(tmp_path):
    """`header.model_git_sha` describes the export run; a gameweek's own
    frozen sha lives only inside `freeze_provenance`. They must not be
    forced into agreement."""
    write_freeze(
        3, {**_real_freeze(3), "model_git_sha": "deadbeef01", "leakage_check": {"ran": True, "passed": True, "n_features": 20}},
        freezes_dir=tmp_path,
    )
    actuals = pl.DataFrame([_actual_row(3, eid, "MID", 4) for eid in range(1, 21)], schema=_SCHEMA)

    file = build_papertrade(
        real_points_by_gw={}, freezes_dir=tmp_path, actuals=actuals, price_eval=_ZERO_PRICE_EVAL,
    )

    provenance = file.launch_gate.freeze_provenance[0]
    assert provenance.gw == 3
    assert provenance.model_git_sha == "deadbeef01"
    assert provenance.leakage_check is not None
    assert provenance.leakage_check.deadline is None, "this fixture's leakage_check omits deadline on purpose"
    # The header's own sha describes this export run, not gw3's freeze -- it
    # is whatever the real repo's git state is, and must not be asserted
    # equal to the frozen gameweek's sha above.
    assert file.header.model_git_sha != "deadbeef01"


def test_the_file_round_trips_through_the_strict_contract(tmp_path):
    write_freeze(3, {**_real_freeze(3), "shadow_state_after": squad_state_to_dict(_squad_state())}, freezes_dir=tmp_path)
    actuals = pl.DataFrame([_actual_row(3, eid, "MID", 4) for eid in range(1, 21)], schema=_SCHEMA)

    file = build_papertrade(
        real_points_by_gw={3: 44}, freezes_dir=tmp_path, actuals=actuals, price_eval=_ZERO_PRICE_EVAL,
    )

    reloaded = PaperTradeFile.model_validate_json(file.model_dump_json())
    assert reloaded == file


def test_a_nan_mae_serializes_as_null_not_a_nan_token(tmp_path):
    """Predictions and actuals both exist for gw3, but for disjoint
    element_ids, so the inner join in `evaluate_gw_player_level` is
    non-empty-inputs-but-zero-rows: `mae`/`spearman_mean` come back NaN,
    not an exception. `json_safe` must turn that into `null` on the wire,
    never a literal NaN token (which `JSON.parse` rejects outright)."""
    write_freeze(3, _real_freeze(3, n=20), freezes_dir=tmp_path)  # projections for element_ids 1-20
    actuals = pl.DataFrame(
        [_actual_row(3, eid, "MID", 4) for eid in range(101, 121)], schema=_SCHEMA  # actuals for 101-120
    )

    file = build_papertrade(
        real_points_by_gw={}, freezes_dir=tmp_path, actuals=actuals, price_eval=_ZERO_PRICE_EVAL,
    )

    # A zero-row join is not a degenerate freeze (the projections themselves
    # have real spread) and it is not an exception (both inputs existed) --
    # it is included, honestly, at n=0 with null metrics rather than 0.0.
    assert [row.gw for row in file.player_level] == [3]
    assert file.player_level[0].n == 0
    assert file.player_level[0].mae is None
    assert file.player_level[0].spearman_mean is None
    assert "NaN" not in file.model_dump_json()
