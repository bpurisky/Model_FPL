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

**Degenerate freezes are excluded, not corrected.** A freeze whose
projections are identical for every player predicted nothing, so metrics
computed against it measure nothing — see `DEGENERATE_FREEZE_POLICY` and
`projection_degeneracy`. This is not hypothetical: `papertrade/freezes/gw2.json`
holds exactly 0.8 for all 600 players across all three horizon gameweeks,
because `collector/schemas.py:fixture_is_played` gated on the raw
`finished` flag at the time and every player fell through to the pooled
prior. That was caught by hand; `projection_degeneracy` is what catches
the next one automatically, at both player and squad level, and reports
the exclusion in the gate rather than quietly shrinking the denominator.

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
import statistics
from collections import Counter
from collections.abc import Mapping
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

# A freeze whose projections carry no signal at all: every player assigned
# the identical number. This is not a hypothetical -- papertrade/freezes/gw2.json
# projects exactly 0.8 for all 600 players across all three horizon
# gameweeks, because `fixture_is_played` gated on the raw `finished` flag
# at the time and every player fell through to the pooled prior.
#
# The floor is an epsilon rather than exact zero only to absorb float
# noise; the real case is variance == 0.0 exactly.
DEGENERACY_VARIANCE_FLOOR = 1e-9

# A softer signal: most players share one identical projection, but not
# quite all. This is the same pooled-prior fallthrough happening at 97%
# instead of 100%, and it is *reported* rather than used to exclude --
# there is no evidence behind any particular cutoff here, and silently
# dropping a gameweek that carries partial signal is a worse error than
# making a human look at it. Exclusion is reserved for the unambiguous
# case above.
NEAR_DEGENERACY_MODAL_SHARE = 0.95

DEGENERATE_FREEZE_POLICY = (
    "A freeze with zero projection variance is a null observation, not a bad "
    "prediction: every player received the same number, so MAE and rank "
    "correlation against it measure nothing about the model. Such gameweeks are "
    "EXCLUDED from the evaluation and from §6.5's 13-gameweek count -- never "
    "corrected, re-frozen or back-filled, because the freeze is immutable by "
    "design (§6.1) and a repaired freeze is no longer a record of what was "
    "actually predicted before the deadline."
)


def projection_degeneracy(projections: Mapping[str, float]) -> dict[str, Any]:
    """Whether a freeze's projections for one gameweek distinguish players
    at all, plus the diagnostics needed to say why they don't.

    `is_degenerate` is the automated version of the judgement that had to
    be made by hand about gw2. It keys on variance because that is the
    quantity that is exactly zero when every player has fallen through to
    the same pooled prior, and because it does not depend on how many
    players happen to be in the pool.
    """
    values = [float(v) for v in projections.values()]
    n = len(values)
    if n == 0:
        # Missing is deliberately NOT degenerate. "The freeze recorded no
        # projections" and "the freeze recorded projections that carry no
        # signal" are different facts, and only the second one licenses
        # throwing the gameweek away: exclusion is a strong action, and
        # applying it to data we simply cannot see would silently discard
        # squad-level observations that may be perfectly real.
        return {
            "n": 0, "n_distinct": 0, "variance": None, "modal_value": None,
            "modal_share": None, "is_degenerate": False, "is_near_degenerate": False,
            "is_missing": True,
            "reason": "the freeze records no projections for this gameweek",
        }

    counts = Counter(values)
    modal_value, modal_count = counts.most_common(1)[0]
    modal_share = modal_count / n
    variance = statistics.pvariance(values) if n > 1 else 0.0
    is_degenerate = variance <= DEGENERACY_VARIANCE_FLOOR
    is_near_degenerate = not is_degenerate and modal_share >= NEAR_DEGENERACY_MODAL_SHARE

    if is_degenerate:
        reason = (
            f"all {n} projections are effectively identical "
            f"(variance={variance:.3g}, {len(counts)} distinct value(s), modal={modal_value:g})"
        )
    elif is_near_degenerate:
        reason = (
            f"{modal_share:.1%} of {n} projections share the identical value {modal_value:g} -- "
            "most of the pool has fallen through to the pooled prior"
        )
    else:
        reason = ""

    return {
        "n": n,
        "n_distinct": len(counts),
        "variance": variance,
        "modal_value": modal_value,
        "modal_share": modal_share,
        "is_degenerate": is_degenerate,
        "is_near_degenerate": is_near_degenerate,
        "is_missing": False,
        "reason": reason,
    }


def freeze_degeneracy(gw: int, freezes_dir: Path = FREEZES_DIR, freeze: dict[str, Any] | None = None) -> dict[str, Any]:
    """`projection_degeneracy` for gw's own projections inside gw's freeze.

    A freeze covers a horizon (gw, gw+1, gw+2); only the entry for `gw`
    itself is the prediction being evaluated here.
    """
    freeze = load_freeze(gw, freezes_dir) if freeze is None else freeze
    return projection_degeneracy(freeze.get("projections", {}).get(str(gw), {}))


def evaluate_gw_player_level(gw: int, freezes_dir: Path = FREEZES_DIR, actuals: pl.DataFrame | None = None) -> dict[str, Any]:
    """MAE, within-position Spearman, and calibration for gw's frozen
    projections against its real results. Requires both a freeze file for
    `gw` and gw's actuals to already be recorded (`papertrade/actuals.py`).
    """
    freeze = load_freeze(gw, freezes_dir)
    # ValueError, not the bare KeyError this used to raise: `evaluate_player_level`
    # catches ValueError to record a gameweek as skipped, and a freeze written
    # without projections should land there rather than aborting the whole run.
    predictions = freeze.get("projections", {}).get(str(gw))  # {element_id (str) -> predicted points}
    if not predictions:
        raise ValueError(f"freeze for gw{gw} records no projections for gw{gw} — nothing to evaluate against")
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
        # Computed and returned even when degenerate, rather than raising:
        # the metrics for a null observation are still worth being able to
        # look at, and it is `evaluate_player_level` that decides what to
        # do about them. Raising here would make the failure invisible.
        "degeneracy": freeze_degeneracy(gw, freezes_dir, freeze=freeze),
    }


def evaluate_player_level(
    candidate_gws: list[int], freezes_dir: Path = FREEZES_DIR, actuals: pl.DataFrame | None = None
) -> dict[str, Any]:
    """Player-level evaluation across every candidate gameweek, split into
    the ones that count and the ones that do not.

    The split exists because of gw2 (see DEGENERATE_FREEZE_POLICY). A
    gameweek whose freeze assigned every player the same projection is not
    a gameweek the model got wrong -- it is a gameweek the model did not
    predict. Averaging its MAE in alongside real gameweeks, or counting it
    toward §6.5's 13, would both overstate how much live evidence exists.

    `skipped` is a different thing from `excluded`: skipped means there
    was nothing to evaluate (no freeze, no actuals), excluded means there
    was a freeze and it carried no signal.
    """
    actuals = load_actuals() if actuals is None else actuals
    included: dict[int, dict[str, Any]] = {}
    excluded: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []

    for gw in sorted(candidate_gws):
        try:
            evaluation = evaluate_gw_player_level(gw, freezes_dir=freezes_dir, actuals=actuals)
        except (FileNotFoundError, ValueError) as exc:
            skipped.append({"gw": gw, "reason": str(exc)})
            logger.info("skipping gw%d player-level eval: %s", gw, exc)
            continue

        degeneracy = evaluation["degeneracy"]
        if degeneracy["is_degenerate"]:
            excluded.append({"gw": gw, "reason": degeneracy["reason"], "degeneracy": degeneracy})
            logger.warning(
                "gw%d EXCLUDED from evaluation: %s. %s",
                gw, degeneracy["reason"], DEGENERATE_FREEZE_POLICY,
            )
            continue

        if degeneracy["is_near_degenerate"]:
            logger.warning(
                "gw%d is included but looks degenerate: %s. Check the freeze before trusting its metrics.",
                gw, degeneracy["reason"],
            )
        included[gw] = evaluation

    return {"included": included, "excluded": excluded, "skipped": skipped}


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
    included — see module docstring for why this must not be over-read.

    A gameweek whose freeze was degenerate is excluded here too, for the
    same reason it is excluded at player level: the shadow XI and captain
    for that gameweek were chosen by optimizing against a projection that
    was identical for every player, so the resulting points measure the
    optimizer's tie-breaking, not the model. Note that the exclusion is of
    the *observation* only — `shadow_state_after` still chains forward into
    the next gameweek's starting squad, because that is what actually
    happened and the shadow team cannot be un-picked retroactively.
    """
    actuals = load_actuals() if actuals is None else actuals
    per_gw: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []

    for gw in sorted(real_points_by_gw):
        try:
            freeze = load_freeze(gw, freezes_dir)
        except FileNotFoundError:
            continue
        gw_actuals = actuals.filter(pl.col("gw") == gw)
        if gw_actuals.height == 0:
            continue
        degeneracy = freeze_degeneracy(gw, freezes_dir, freeze=freeze)
        if degeneracy["is_degenerate"]:
            excluded.append({"gw": gw, "reason": degeneracy["reason"]})
            logger.warning("gw%d excluded from squad-level comparison: %s", gw, degeneracy["reason"])
            continue
        if degeneracy["is_missing"]:
            # Included, because we cannot show the observation is null --
            # but said out loud, because we also cannot show it isn't.
            logger.warning(
                "gw%d has a shadow squad but no recorded projections; including it in the "
                "squad-level comparison, but its picks cannot be checked for degeneracy",
                gw,
            )
        actual_points_by_id = dict(zip(gw_actuals["element_id"].to_list(), gw_actuals["total_points"].to_list()))
        shadow_state = squad_state_from_dict(freeze["shadow_state_after"])
        shadow_points = realized_points(shadow_state, actual_points_by_id)
        per_gw.append({"gw": gw, "real_points": real_points_by_gw[gw], "shadow_points": shadow_points})

    cumulative_real = sum(row["real_points"] for row in per_gw)
    cumulative_shadow = sum(row["shadow_points"] for row in per_gw)

    return {
        "n_gameweeks": len(per_gw),
        "per_gw": per_gw,
        "excluded_gameweeks": excluded,
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


def collect_freeze_provenance(gws: list[int], freezes_dir: Path = FREEZES_DIR) -> list[dict[str, Any]]:
    """Per gameweek: what its freeze recorded about how it was produced —
    the leakage assertion, any declared manual correction, and the model
    sha. Freezes written before those fields existed report `None`, which
    is a different and weaker claim than `False`; §6.5's criteria 3 and 4
    are about what was verified *during* the live run, so a freeze with no
    record cannot retroactively be counted as having passed.
    """
    provenance = []
    for gw in sorted(gws):
        try:
            freeze = load_freeze(gw, freezes_dir)
        except FileNotFoundError:
            continue
        leakage = freeze.get("leakage_check")
        provenance.append({
            "gw": gw,
            "leakage_check": leakage,
            "leakage_verified": bool(leakage and leakage.get("ran") and leakage.get("passed")),
            "manual_correction": freeze.get("manual_correction"),
            "records_manual_correction_field": "manual_correction" in freeze,
            "model_git_sha": freeze.get("model_git_sha"),
        })
    return provenance


def _leakage_criterion(provenance: list[dict[str, Any]], n_gws: int, sufficient: bool) -> dict[str, str]:
    """§6.5 criterion 3, read off the freezes rather than asserted here.

    The assertion runs at freeze time (papertrade/freeze.py) because it is
    a claim about what was available before a deadline that has since
    passed. This function only reports what the freezes recorded.
    """
    if not provenance:
        return {"status": "insufficient data", "detail": "no freezes on disk to report a leakage check for."}

    unverified = [p["gw"] for p in provenance if not p["leakage_verified"]]
    if unverified:
        return {
            "status": "not tracked",
            "detail": (
                f"{len(provenance) - len(unverified)}/{len(provenance)} freeze(s) record a passing leakage check; "
                f"gw{', gw'.join(str(g) for g in unverified)} predate the check being wired into the live freeze path "
                "and cannot be verified retroactively — the assertion is about what was available before a deadline "
                "that has already passed."
            ),
        }
    detail = f"all {len(provenance)} freeze(s) record a leakage assertion that ran and passed at freeze time."
    return {"status": "PASS" if sufficient else "insufficient data", "detail": f"{n_gws}/13 gameweeks evaluated; {detail}"}


def _manual_correction_criterion(
    provenance: list[dict[str, Any]], squad_eval: dict[str, Any], sufficient: bool
) -> dict[str, str]:
    """§6.5 criterion 4. `manual_correction` is human-declared at freeze
    time; a freeze that predates the field says nothing either way and is
    reported as untracked rather than assumed clean."""
    n = squad_eval["n_gameweeks"]
    if not provenance:
        return {"status": "insufficient data", "detail": f"{n}/13 gameweeks have both a freeze and recorded actuals; no freezes to inspect."}

    untracked = [p["gw"] for p in provenance if not p["records_manual_correction_field"]]
    corrected = [(p["gw"], p["manual_correction"]) for p in provenance if p["manual_correction"]]

    if corrected:
        listed = "; ".join(f"gw{gw}: {reason}" for gw, reason in corrected)
        return {
            "status": "FAIL",
            "detail": f"{len(corrected)} gameweek(s) record a manual correction, breaking the consecutive run — {listed}",
        }
    if untracked:
        return {
            "status": "not tracked",
            "detail": (
                f"{n}/13 gameweeks have both a freeze and recorded actuals; "
                f"gw{', gw'.join(str(g) for g in untracked)} predate the manual-correction field and make no claim "
                "either way. The field cannot be added retroactively — a freeze is immutable (§6.1)."
            ),
        }
    return {
        "status": "PASS" if sufficient else "insufficient data",
        "detail": f"{n}/13 gameweeks; all {len(provenance)} freeze(s) declare no manual correction.",
    }


def launch_gate_report(
    player_eval_by_gw: dict[int, dict],
    squad_eval: dict[str, Any],
    price_eval: PriceModelEvaluation | None = None,
    excluded_gws: list[dict[str, Any]] | None = None,
    freeze_provenance: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """§6.5's five criteria, reported honestly rather than forced — see
    module docstring for exactly what's and isn't wired up yet.

    `excluded_gws` is reported rather than merely subtracted. A gate that
    quietly said "1/13 gameweeks" after gw2 was dropped would look
    identical to one where gw2 had never been frozen at all, and the
    difference between those two is the whole reason the guard exists.
    """
    n_gws = len(player_eval_by_gw)
    sufficient = n_gws >= 13
    excluded_gws = excluded_gws or []
    freeze_provenance = freeze_provenance if freeze_provenance is not None else []
    exclusion_note = ""
    if excluded_gws:
        listed = ", ".join(f"gw{e['gw']} ({e['reason']})" for e in excluded_gws)
        exclusion_note = f" Excluded as null observations: {listed}. {DEGENERATE_FREEZE_POLICY}"
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
            "detail": f"{n_gws}/13 gameweeks evaluated; live-data baseline comparison (backtest/baselines.py against papertrade/actuals.py) is not built yet regardless of gameweek count.{exclusion_note}",
        },
        "beats_baselines_on_rank_correlation": {
            "status": "insufficient data" if not sufficient else "not wired to live baselines yet",
            "detail": f"{n_gws}/13 gameweeks evaluated; same live-baseline gap as above.{exclusion_note}",
        },
        "no_leakage_assertion_fired": _leakage_criterion(freeze_provenance, n_gws, sufficient),
        "squad_reconstruction_ran_13_consecutive_gws_without_manual_correction": _manual_correction_criterion(
            freeze_provenance, squad_eval, sufficient
        ),
        "price_change_model_reports_hit_rate_with_ci": {"status": price_status, "detail": price_detail},
    }
    all_pass = all(c["status"] == "PASS" for c in criteria.values())
    return {
        "ready_to_launch": all_pass,
        "gameweeks_evaluated": n_gws,
        "gameweeks_excluded": excluded_gws,
        "freeze_provenance": freeze_provenance,
        "criteria": criteria,
    }
