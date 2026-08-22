"""§6.3-6.5: honest forward evaluation of the frozen predictions and the
shadow team, against real results as they accumulate.

Two levels, deliberately kept separate — §6.3's own framing:

**Player-level** (papertrade/actuals.py's real per-gw results vs
papertrade/freeze.py's frozen projections): ~500 players x however many
gameweeks are evaluated is ample for MAE, rank correlation, calibration.
Trust these numbers.

**Squad-level** (the real team's actual gw points, from
`/entry/{id}/history/`, vs the shadow team's realized points,
`squad/shadow.py:realized_points`): 13 observations at most, dominated by
variance. A genuinely good model can sit at a bad rank after 13 gameweeks
because one captain blanked twice. Log it, report it, and do **not**
overhaul the model on the basis of it — that warning belongs in the report
output itself (§6.3's explicit instruction), not just this docstring; see
`evaluate_squad_level`'s `warning` field.

§6.5's launch gate is reported honestly rather than forced: with 0-1
gameweeks of live data so far, most of its five criteria are correctly
"not yet measurable," not a fabricated pass. Two of the five (baseline
comparison on live data) are still flagged as not wired up yet — building
the opponent-difficulty/kickoff-time-decorated live equivalent of
`backtest/baselines.py`'s inputs is real, separate plumbing beyond this
module's current scope. The fifth (§6.4's price-change hit rate) now calls
`analytics/price_model.py` directly — built alongside this module rather
than left as a permanent gap — but its own module docstring is explicit
that a hit rate this early is near-meaningless (unfitted thresholds, ~a
day of distilled data, no price has necessarily moved yet): PASS still
requires an actual computed hit rate to clear a bar, not just a number to
exist.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import polars as pl

from analytics.price_model import PriceModelEvaluation, run_price_model_evaluation
from backtest.report import calibration_curve, mae, spearman_within_position
from collector.client import FPLClient
from collector.config import CollectorConfig
from collector.schemas import parse_entry_history
from papertrade.actuals import ACTUALS_PATH, load_actuals
from papertrade.freeze import FREEZES_DIR, load_freeze
from squad.reconstruct import squad_state_from_dict
from squad.shadow import realized_points

DISTILLED_DIR = Path("data/distilled")
PRICE_MODEL_HORIZON_HOURS = 24  # FPL price changes land roughly once a day

logger = logging.getLogger("papertrade.evaluate")

SQUAD_LEVEL_VARIANCE_WARNING = (
    "Squad-level comparison is at most 13 observations, dominated by variance "
    "(§6.3): a genuinely good model can sit at a bad rank after 13 gameweeks "
    "because one captain blanked twice. Treat this as a log, not a verdict -- "
    "do not overhaul the model on the basis of these numbers alone."
)


def evaluate_gw_player_level(gw: int, freezes_dir: Path = FREEZES_DIR, actuals: pl.DataFrame | None = None) -> dict[str, Any]:
    """MAE, within-position Spearman, and calibration for gw's frozen
    projections against its real results. Requires both a freeze file for
    `gw` and gw's actuals to already be recorded (`papertrade/actuals.py`).
    """
    freeze = load_freeze(gw, freezes_dir)
    predictions = freeze["projections"][str(gw)]  # {element_id (str) -> predicted points}
    actuals = load_actuals() if actuals is None else actuals
    gw_actuals = actuals.filter(pl.col("gw") == gw)
    if gw_actuals.height == 0:
        raise ValueError(f"no actuals recorded for gw{gw} — run papertrade.actuals.fetch_gw_actuals first")

    pred_df = pl.DataFrame({"element_id": [int(k) for k in predictions], "prediction": list(predictions.values())})
    results = gw_actuals.join(pred_df, on="element_id", how="inner").with_columns(
        (pl.col("prediction") - pl.col("total_points")).alias("error")
    )

    return {
        "gw": gw,
        "n": results.height,
        "mae": mae(results),
        "spearman_within_position": spearman_within_position(results),
        "calibration": calibration_curve(results),
    }


async def fetch_real_gw_points(cfg: CollectorConfig, entry_id: int) -> dict[int, int]:
    """gw -> the real entry's actual points that gameweek, from
    `/entry/{id}/history/`'s `current` list."""
    async with FPLClient(**cfg.api.client_kwargs()) as client:
        entry_history_raw = await client.get_json(f"/entry/{entry_id}/history/")
    entry_history = parse_entry_history(entry_history_raw, logger)
    return {h.event: h.points for h in entry_history.current}


def evaluate_squad_level(
    real_points_by_gw: dict[int, int], actuals: pl.DataFrame | None = None, freezes_dir: Path = FREEZES_DIR
) -> dict[str, Any]:
    """Real team vs shadow team, gw by gw and cumulative. Only gameweeks
    with both a frozen shadow recommendation *and* recorded actuals are
    included — see module docstring for why this must not be over-read."""
    actuals = load_actuals() if actuals is None else actuals
    per_gw: list[dict[str, Any]] = []

    for gw in sorted(real_points_by_gw):
        try:
            freeze = load_freeze(gw, freezes_dir)
        except FileNotFoundError:
            continue
        gw_actuals = actuals.filter(pl.col("gw") == gw)
        if gw_actuals.height == 0:
            continue
        actual_points_by_id = dict(zip(gw_actuals["element_id"].to_list(), gw_actuals["total_points"].to_list()))
        shadow_state = squad_state_from_dict(freeze["shadow_state_after"])
        shadow_points = realized_points(shadow_state, actual_points_by_id)
        per_gw.append({"gw": gw, "real_points": real_points_by_gw[gw], "shadow_points": shadow_points})

    cumulative_real = sum(row["real_points"] for row in per_gw)
    cumulative_shadow = sum(row["shadow_points"] for row in per_gw)

    return {
        "n_gameweeks": len(per_gw),
        "per_gw": per_gw,
        "cumulative_real_points": cumulative_real,
        "cumulative_shadow_points": cumulative_shadow,
        "shadow_minus_real": cumulative_shadow - cumulative_real,
        "warning": SQUAD_LEVEL_VARIANCE_WARNING,
    }


def run_price_model_for_gate(
    distilled_dir: Path = DISTILLED_DIR, now: datetime | None = None, horizon_hours: int = PRICE_MODEL_HORIZON_HOURS
) -> PriceModelEvaluation:
    """Predicts from pressure `horizon_hours` ago, checks the real outcome
    now — the launch-gate's live invocation of `analytics/price_model.py`.
    """
    now = now or datetime.now(timezone.utc)
    return run_price_model_evaluation(distilled_dir, now - timedelta(hours=horizon_hours), now)


def launch_gate_report(
    player_eval_by_gw: dict[int, dict], squad_eval: dict[str, Any], price_eval: PriceModelEvaluation | None = None
) -> dict[str, Any]:
    """§6.5's five criteria, reported honestly rather than forced — see
    module docstring for exactly what's and isn't wired up yet."""
    n_gws = len(player_eval_by_gw)
    sufficient = n_gws >= 13
    price_eval = price_eval or PriceModelEvaluation(n=0, n_moves_predicted=0, hit_rate=None, ci_low=None, ci_high=None)

    if price_eval.hit_rate is None:
        price_status = "insufficient data"
        price_detail = (
            f"analytics/price_model.py ran against {price_eval.n} player(s) but predicted zero actual moves "
            "(rise/fall) to score — either too little distilled history has accumulated yet, or pressure "
            "never crossed the model's (unfitted) thresholds in this window."
        )
    else:
        price_status = "PASS" if price_eval.hit_rate > 0.5 else "FAIL"
        price_detail = (
            f"hit_rate={price_eval.hit_rate:.2f} (95% CI [{price_eval.ci_low:.2f}, {price_eval.ci_high:.2f}]) "
            f"over {price_eval.n_moves_predicted} predicted move(s) of {price_eval.n} evaluated -- see "
            "analytics/price_model.py's module docstring: thresholds are unfitted and this early a hit rate "
            "carries little weight regardless of which side of 0.5 it lands on."
        )

    criteria = {
        "beats_fixture_adjusted_trailing_mean_mae": {
            "status": "insufficient data" if not sufficient else "not wired to live baselines yet",
            "detail": f"{n_gws}/13 gameweeks evaluated; live-data baseline comparison (backtest/baselines.py against papertrade/actuals.py) is not built yet regardless of gameweek count.",
        },
        "beats_baselines_on_rank_correlation": {
            "status": "insufficient data" if not sufficient else "not wired to live baselines yet",
            "detail": f"{n_gws}/13 gameweeks evaluated; same live-baseline gap as above.",
        },
        "no_leakage_assertion_fired": {
            "status": "not tracked",
            "detail": "Leakage assertions (backtest/leakage.py) exist for the historical walk-forward harness only; nothing runs them against the live pipeline yet.",
        },
        "squad_reconstruction_ran_13_consecutive_gws_without_manual_correction": {
            "status": "not tracked",
            "detail": f"{squad_eval['n_gameweeks']}/13 gameweeks have both a freeze and recorded actuals; no automated log of manual corrections exists.",
        },
        "price_change_model_reports_hit_rate_with_ci": {"status": price_status, "detail": price_detail},
    }
    all_pass = all(c["status"] == "PASS" for c in criteria.values())
    return {"ready_to_launch": all_pass, "gameweeks_evaluated": n_gws, "criteria": criteria}
