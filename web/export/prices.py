"""`prices.json` — Price Watch: the rise-only price model's call for the
*current* daily cycle, and its own track record.

`analytics/price_model.py` is the whole model; this file only reads it. The
cycle is built by `current_cycle` (the same staleness, 24h-window and
deadline rules `daily_cycles` applies to history), the scores and 72h
eligibility by `cycle_observations`, and the call by `predict_price_changes`
— nothing here re-derives the rule, so a refit moves this file and the
launch gate together.

**A recorded deviation from §5.0.2 (surfaces make no calls).** Like D15 for
the Squad Optimizer, this is a surface that states a model call. The
justification: it names no squad, captain or transfer and has no decision
to make for the reader; it reports one narrow, mechanical rule (a player
whose 24h net transfers sit at or above a fitted cut and whose price has
not moved in 72h is "rise"), and it ships that rule's *own* out-of-sample
record beside every call — hit rate with a 95% CI and n, or "not enough
scored calls yet" — so the claim and the evidence for it cannot be read
apart. It is rise-only: it never calls falls, because the fall signal was
too weak to be worth a claim (see the model's docstring).

**Cadence.** `web.yml` runs `all` at :20 every 6 hours, so the 18:20 UTC run
lands *before* the 19:00 boundary: that file is an open cycle, a running
total that can still move. The next run (00:20) reads a locked cycle whose
overnight change is pending or just done. Neither is stale — the file says
which it is through `locked`.

**Track record.** The headline is the gate's out-of-sample scoring
(`run_price_model_for_gate`: cycles on or after `fit_cutoff`, rise calls
only, "insufficient data" until `MIN_GATE_PREDICTED_MOVES`). The yaml's fit
statistics are attached but labelled in-sample — the threshold was chosen
to maximise them. `recent_cycles` lists the last `RECENT_CYCLES` scored
cycles, flagged out-of-sample or not, so a reader can see how many are
genuinely held out (right after a refit: none).
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path

import polars as pl
import yaml

from analytics.price_model import (
    PRICE_MODEL_CONFIG,
    RECENT_MOVE_WINDOW,
    PriceModelEvaluation,
    PriceThresholds,
    _state_at,
    current_cycle,
    cycle_observations,
    daily_cycles,
    evaluate_observations,
    load_distilled_history,
    load_thresholds,
    predict_price_changes,
)
from papertrade.evaluate import DISTILLED_DIR, EVENTS_PATH, MIN_GATE_PREDICTED_MOVES, price_gate_status, run_price_model_for_gate
from web.export.contract import (
    PriceCycleOutcome,
    PriceInSampleFit,
    PricesFile,
    PriceTrackRecord,
    PriceWatchPlayer,
    build_header,
)
from web.export.timeseries import POSITION_BY_ELEMENT_TYPE

logger = logging.getLogger("web.export.prices")

REFERENCE_DIR = Path("data/reference")
NORMALIZATION_BASIS = "raw_net_transfers_24h"
RECENT_CYCLES = 14


def _identity(reference_dir: Path) -> pl.DataFrame:
    """element_id, name, team, position from the reference tier."""
    players = pl.read_parquet(reference_dir / "players.parquet")
    teams = pl.read_parquet(reference_dir / "teams.parquet").select(pl.col("id").alias("team"), pl.col("name").alias("team_name"))
    return players.join(teams, on="team", how="left").select(
        pl.col("id").alias("element_id"),
        pl.col("web_name").alias("name"),
        pl.col("team_name").alias("team"),
        pl.col("element_type").replace_strict(POSITION_BY_ELEMENT_TYPE, default=None).alias("position"),
    )


def _in_sample(path: Path) -> PriceInSampleFit | None:
    if not path.exists():
        return None
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))

    def text(key: str) -> str | None:  # yaml parses the dates; the contract carries them as text
        return None if raw.get(key) is None else str(raw[key])

    return PriceInSampleFit(
        fit_cutoff=text("fit_cutoff"),
        fit_span=text("fit_span"),
        fit_cycles=raw.get("fit_cycles"),
        rise_f1=raw.get("fit_rise_f1"),
        rise_n_called=raw.get("fit_rise_n_called"),
        hit_rate=raw.get("fit_in_sample_hit_rate"),
        fitted_at=text("fitted_at"),
    )


def _recent_cycles(history: pl.DataFrame, deadlines: list[datetime], thresholds: PriceThresholds) -> list[PriceCycleOutcome]:
    """Calls and hits of the last `RECENT_CYCLES` scored cycles, each scored
    by `evaluate_observations` (the gate's own scorer) on its own rows."""
    cycles = daily_cycles(history["snapshot_ts"].unique().to_list(), deadlines)[-RECENT_CYCLES:]
    out = []
    for cycle in cycles:
        evaluation = evaluate_observations(cycle_observations(history, [cycle]), thresholds.rise)
        hits = 0 if evaluation.hit_rate is None else round(evaluation.hit_rate * evaluation.n_moves_predicted)
        cutoff = thresholds.fit_cutoff
        out.append(PriceCycleOutcome(
            date=str(cycle.prediction_ts.date()),
            calls=evaluation.n_moves_predicted,
            hits=hits,
            out_of_sample=cutoff is not None and cycle.prediction_ts.date() >= cutoff,
        ))
    return out


def build_prices(
    *,
    distilled_dir: Path = DISTILLED_DIR,
    events_path: Path = EVENTS_PATH,
    reference_dir: Path = REFERENCE_DIR,
    config_path: Path = PRICE_MODEL_CONFIG,
    now: datetime | None = None,
    price_eval: PriceModelEvaluation | None = None,
) -> PricesFile:
    """See the module docstring. `now` is injectable so staleness is
    testable; `price_eval` so a test need not run the gate's backtest."""
    now = now or datetime.now(timezone.utc)
    thresholds = load_thresholds(config_path)
    history = load_distilled_history(distilled_dir)
    deadlines = pl.read_parquet(events_path)["deadline_time"].drop_nulls().to_list() if events_path.exists() else []
    snapshots = history["snapshot_ts"].unique().to_list() if history.height else []

    price_eval = price_eval or run_price_model_for_gate(distilled_dir, events_path, thresholds)
    track_record = PriceTrackRecord(
        status=price_gate_status(price_eval),
        n_calls=price_eval.n_moves_predicted,
        n_cycles=price_eval.n_cycles or 0,
        min_calls=MIN_GATE_PREDICTED_MOVES,
        hit_rate=price_eval.hit_rate,
        ci_low=price_eval.ci_low,
        ci_high=price_eval.ci_high,
        span=price_eval.span,
        in_sample=_in_sample(config_path),
        recent_cycles=_recent_cycles(history, deadlines, thresholds) if history.height else [],
    )

    current = current_cycle(snapshots, deadlines, now)
    players: list[PriceWatchPlayer] = []
    if current.cycle is not None:
        cycle = current.cycle
        observations = cycle_observations(history, [cycle]).select("element_id", "net_transfers", "eligible")
        calls = predict_price_changes(observations, thresholds.rise).select("element_id", "predicted_direction")
        scored = observations.join(calls, on="element_id", how="left")
        # In the snapshot but without one ~24h earlier (new to the dataset):
        # no flow can be computed, which is not the same as no flow.
        state = _state_at(history, cycle.prediction_ts).select("element_id", "now_cost")
        scored = state.join(scored, on="element_id", how="left")
        rows = scored.join(_identity(reference_dir), on="element_id", how="left").to_dicts()
        for row in rows:
            net = row["net_transfers"]
            if net is None:
                reason, eligible = "insufficient_history", False
            elif not row["eligible"]:
                reason, eligible = "moved_within_72h", False
            else:
                reason, eligible = None, True
            players.append(PriceWatchPlayer(
                element_id=row["element_id"],
                name=row["name"] or f"Player {row['element_id']}",
                team=row["team"] or "",
                position=row["position"] or "",
                now_cost=int(row["now_cost"]),
                net_transfers_24h=net,
                progress=None if net is None else net / thresholds.rise,
                eligible=eligible,
                ineligible_reason=reason,
                call=row["predicted_direction"] or "stable",
            ))
        players.sort(key=lambda p: (p.progress is None, -(p.progress or 0.0), p.element_id))

    logger.info(
        "prices: %s, %d players (%d called rise), track record %s over %d call(s)",
        current.status, len(players), sum(p.call == "rise" for p in players), track_record.status, track_record.n_calls,
    )
    return PricesFile(
        header=build_header(rows=len(players), source_gameweek=None, normalization_basis=NORMALIZATION_BASIS),
        status=current.status,
        status_detail=current.detail,
        snapshot_ts=current.cycle.prediction_ts if current.cycle else None,
        boundary_ts=current.boundary_ts,
        next_boundary_ts=current.next_boundary_ts,
        locked=current.locked if current.cycle else None,
        threshold=thresholds.rise,
        recent_move_hours=int(RECENT_MOVE_WINDOW.total_seconds() // 3600),
        players=players,
        track_record=track_record,
    )
