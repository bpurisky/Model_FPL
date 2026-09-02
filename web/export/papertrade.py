"""`papertrade.json` — §6.3-6.5 made legible (§5.16 deviation D14).

`fpl-trends-frontend-superprompt-v2.md` §5.1.3/§5.13 keep Phase 3 and 4
surfaces stubbed "throughout" Phase 5. Paper Trade is different from the
Squad Optimizer in exactly the way that matters for that boundary: it
reports what an already-frozen shadow team already scored. It decides
nothing — no transfer, no captain, no squad — so exporting it real stays
inside §5.0.2's Job 1/Job 2 boundary the same way Model Board's D1 did for
a different Phase 3/4 boundary claim.

**Two shas, not one.** `PaperTradeFile.header.model_git_sha` (via
`build_header`) is the sha of the code that ran *this export* — the live
model today. Each `launch_gate.freeze_provenance[].model_git_sha` is that
gameweek's own frozen sha, from when that freeze was written. A surface
showing both must label them differently; they usually disagree.

**The empty state is the real state right now.** `papertrade/freezes/` is
empty on disk and `data/current_season/` does not exist yet, so every
field below is expected to degrade to an honest zero/empty/"insufficient
data" rather than fail to build. `evaluate_squad_level` and
`evaluate_player_level` already treat "no freezes yet" as a normal input,
not a missing build artifact — unlike `board.py`'s `panel.parquet`, there
is no local file this function requires to exist, so there is no
`FileNotFoundError` path to raise here.

**Assembly mirrors `papertrade/__main__.py:cmd_evaluate` exactly** — same
`evaluated_gws` -> `candidate_gws` -> `evaluate_player_level` ->
`evaluate_squad_level` -> `run_price_model_for_gate` ->
`collect_freeze_provenance` -> `launch_gate_report` sequence — so the
CLI's own `evaluate` report and this export can never silently disagree
about what counts.
"""

from __future__ import annotations

import logging
from pathlib import Path

import polars as pl

from analytics.price_model import PriceModelEvaluation
from papertrade.actuals import load_actuals
from papertrade.evaluate import (
    collect_freeze_provenance,
    evaluate_player_level,
    evaluate_squad_level,
    launch_gate_report,
    run_price_model_for_gate,
)
from papertrade.freeze import FREEZES_DIR, latest_frozen_gw
from web.export.contract import (
    PaperTradeFile,
    PaperTradeFreezeProvenance,
    PaperTradeGateCriterion,
    PaperTradeGwNote,
    PaperTradeLaunchGate,
    PaperTradeLeakageCheck,
    PaperTradePlayerLevelGw,
    PaperTradePriceEval,
    PaperTradeSquadLevel,
    PaperTradeSquadLevelGw,
    build_header,
    json_safe,
)

logger = logging.getLogger("web.export.papertrade")

NORMALIZATION_BASIS = "raw_points_not_normalized"


def _gw_notes(rows: list[dict]) -> list[PaperTradeGwNote]:
    return [PaperTradeGwNote(gw=row["gw"], reason=row["reason"]) for row in rows]


def _leakage_check(raw: dict | None) -> PaperTradeLeakageCheck | None:
    if raw is None:
        return None
    return PaperTradeLeakageCheck(
        ran=raw["ran"],
        passed=raw["passed"],
        n_features=raw["n_features"],
        latest_feature_available_at=raw.get("latest_feature_available_at"),
        deadline=raw.get("deadline"),
    )


def _freeze_provenance(rows: list[dict]) -> list[PaperTradeFreezeProvenance]:
    return [
        PaperTradeFreezeProvenance(
            gw=row["gw"],
            leakage_check=_leakage_check(row["leakage_check"]),
            leakage_verified=row["leakage_verified"],
            manual_correction=row["manual_correction"],
            records_manual_correction_field=row["records_manual_correction_field"],
            model_git_sha=row["model_git_sha"],
        )
        for row in rows
    ]


def _criteria(raw: dict[str, dict]) -> dict[str, PaperTradeGateCriterion]:
    return {key: PaperTradeGateCriterion(status=value["status"], detail=value["detail"]) for key, value in raw.items()}


def build_papertrade(
    *,
    real_points_by_gw: dict[int, int],
    freezes_dir: Path = FREEZES_DIR,
    actuals: pl.DataFrame | None = None,
    price_eval: PriceModelEvaluation | None = None,
) -> PaperTradeFile:
    """§6.3-6.5's live evaluation, exported. See module docstring for the
    empty-state and two-sha guarantees."""
    actuals = load_actuals() if actuals is None else actuals
    evaluated_gws = sorted(set(actuals["gw"].to_list())) if actuals.height else []
    latest_gw = latest_frozen_gw(freezes_dir)
    candidate_gws = [gw for gw in evaluated_gws if gw <= (latest_gw or 0)]

    player_level = evaluate_player_level(candidate_gws, freezes_dir=freezes_dir, actuals=actuals)
    squad_eval = evaluate_squad_level(real_points_by_gw, actuals=actuals, freezes_dir=freezes_dir)
    price_eval = price_eval or run_price_model_for_gate()
    freeze_provenance = collect_freeze_provenance(candidate_gws, freezes_dir)
    gate = launch_gate_report(
        player_level["included"],
        squad_eval,
        price_eval=price_eval,
        excluded_gws=player_level["excluded"],
        freeze_provenance=freeze_provenance,
    )

    player_level_rows = [
        PaperTradePlayerLevelGw(
            gw=gw,
            n=evaluation["n"],
            mae=json_safe(evaluation["mae"]),
            spearman_mean=json_safe(evaluation["spearman_within_position"].get("mean")),
        )
        for gw, evaluation in sorted(player_level["included"].items())
    ]

    logger.info(
        "papertrade: player-level %d gw(s), squad-level %d gw(s), gate %s",
        len(player_level_rows), squad_eval["n_gameweeks"],
        "READY" if gate["ready_to_launch"] else "NOT READY",
    )

    return PaperTradeFile(
        header=build_header(
            rows=len(player_level_rows),
            source_gameweek=latest_gw,
            normalization_basis=NORMALIZATION_BASIS,
        ),
        player_level=player_level_rows,
        player_level_excluded=_gw_notes(player_level["excluded"]),
        player_level_skipped=_gw_notes(player_level["skipped"]),
        squad_level=PaperTradeSquadLevel(
            n_gameweeks=squad_eval["n_gameweeks"],
            per_gw=[PaperTradeSquadLevelGw(**row) for row in squad_eval["per_gw"]],
            excluded_gameweeks=_gw_notes(squad_eval["excluded_gameweeks"]),
            cumulative_real_points=squad_eval["cumulative_real_points"],
            cumulative_shadow_points=squad_eval["cumulative_shadow_points"],
            shadow_minus_real=squad_eval["shadow_minus_real"],
            warning=squad_eval["warning"],
        ),
        launch_gate=PaperTradeLaunchGate(
            ready_to_launch=gate["ready_to_launch"],
            gameweeks_evaluated=gate["gameweeks_evaluated"],
            gameweeks_excluded=_gw_notes(gate["gameweeks_excluded"]),
            freeze_provenance=_freeze_provenance(gate["freeze_provenance"]),
            criteria=_criteria(gate["criteria"]),
        ),
        price_eval=PaperTradePriceEval(
            n=price_eval.n,
            n_moves_predicted=price_eval.n_moves_predicted,
            hit_rate=json_safe(price_eval.hit_rate),
            ci_low=json_safe(price_eval.ci_low),
            ci_high=json_safe(price_eval.ci_high),
        ),
    )
