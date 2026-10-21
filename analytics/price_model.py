"""§4.1/§6.4/§6.5's price-change model: a RISE-ONLY predictor that flags
which players are likely to go up in price tonight from their raw 24h net
transfers, and evaluates that prediction's hit rate against real observed
price changes — with an explicit, wide confidence interval (§6.4: market
signals have no historical analogue and gw13 is their *first* evaluation
window, not a settled one; never report a bare point estimate).

No scipy in the locked stack (§1.1), so the confidence interval is a
standard normal approximation for a binomial proportion — matching
`backtest/report.py`'s own precedent of computing Spearman without scipy.

**Why rise-only (investigation 2026-10-21, 49 overnight cycles).** The first
model scored `net_transfers / (ownership + 1)` and called both rises and
falls. Out of sample it hit 0.09 (158 calls, 148 of them falls, 14 right).
Falls are not predictable from this data: fall AUC was only 0.75-0.80 for
every feature tried and fall precision never rose above ~0.25, so any fall
call is mostly noise and drags the pooled hit rate down with it. Rises rank
very well (AUC 0.97-0.99), and are best ranked by *raw* 24h net transfers —
dividing by ownership hurts rises. So the score here is raw 24h net
transfers, and nothing is ever called a fall.

**Recently-moved players are ineligible.** A player whose price moved in the
previous `RECENT_MOVE_WINDOW` (72h) almost never moves again soon
(P(move | moved last night) 0.2% vs a 1.5% base rate) yet keeps the heavy
net inflow that moved him, so he is a systematic false positive. Excluding
them lifted rise precision@k from ~0.35 to ~0.46. Eligibility is read off
the history's own `now_cost` snapshots, so it needs no extra data.

**Thresholds are fitted once, by hand, and then scored only
out-of-sample.** `fit_rise_threshold` grid-searches the rise cut-off over
the daily price-change cycles before a cut-off date and writes it, with
that date, to `config/price_model.yaml`
(`uv run python -m analytics fit-price-thresholds`). The launch gate
(`papertrade/evaluate.py:run_price_model_for_gate`) scores only the cycles
*after* `fit_cutoff`, so the hit rate it reports was never seen by the
fit. Refitting moves the cut-off forward and shrinks that out-of-sample
span back to nothing — do it deliberately, not on a schedule. The
2026-10-21 refit is such a deliberate one: the rise-only design above (raw
net transfers, the 72h eligibility rule) was chosen after looking at the
2026-10-02..2026-10-20 span, so that span is no longer out-of-sample and the
fit now takes every cycle before 2026-10-21. Only cycles from that date on
are genuinely held out. Without the config file the unfitted placeholder
applies, and there is no out-of-sample span at all (the placeholder was
never meaningful: 24h net transfers run into the thousands).

**Why daily cycles, not a fixed lookback from "now".** FPL moves prices
once a night (observed 2026-27: always between ~22:30 and ~02:00 UTC,
never between 08:00 and 19:00). The collector's snapshots land ~4-5 a day
in practice (GitHub throttles cron), not hourly, so a short pressure window
usually compared a snapshot with itself and predicted nothing. A cycle runs
from one `CYCLE_BOUNDARY_HOUR_UTC` to the next: the score is the 24h
net-transfer flow up to the last snapshot before the boundary, and the
outcome is the price change between that snapshot and the last one before
the *next* boundary — which brackets the overnight change whatever minute
it lands.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path

import duckdb
import polars as pl
import yaml

from analytics.deltas import NUMERIC_DELTA_COLUMNS
from analytics.trending import price_change_pressure

Z_95 = 1.959963985  # standard normal critical value for a 95% CI

PRICE_MODEL_CONFIG = Path("config/price_model.yaml")

UNFITTED_RISE_THRESHOLD = 0.5

# Any hour inside the observed 08:00-19:00 UTC quiet span works; 19:00
# keeps the pressure window as close to the overnight change as it allows.
CYCLE_BOUNDARY_HOUR_UTC = 19
PRESSURE_WINDOW = timedelta(hours=24)
# A player whose now_cost changed at any snapshot this far back from the
# prediction is ineligible to be called a riser (see module docstring).
RECENT_MOVE_WINDOW = timedelta(hours=72)
# A cycle whose bracketing snapshot sits further than this from where it
# should be (a collector outage) is dropped rather than scored on stale data.
MAX_SNAPSHOT_STALENESS = timedelta(hours=8)
# A threshold that calls fewer rises than this over the fit span is
# fitting a handful of lucky players, not a signal.
MIN_FIT_PREDICTED_MOVES = 20


@dataclass(frozen=True)
class PriceThresholds:
    rise: float
    fit_cutoff: date | None  # None: unfitted placeholder, nothing is out-of-sample


def load_thresholds(path: Path = PRICE_MODEL_CONFIG) -> PriceThresholds:
    if not path.exists():
        return PriceThresholds(rise=UNFITTED_RISE_THRESHOLD, fit_cutoff=None)
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    return PriceThresholds(rise=float(raw["rise_threshold"]), fit_cutoff=raw["fit_cutoff"])


_CONFIGURED = load_thresholds()
DEFAULT_RISE_THRESHOLD = _CONFIGURED.rise


@dataclass(frozen=True)
class PriceModelEvaluation:
    n: int  # players with both a prediction and a known actual outcome
    n_moves_predicted: int  # of those, how many were called as rises (the model never calls falls)
    hit_rate: float | None  # None, never a fabricated 0.0, when n_moves_predicted == 0
    ci_low: float | None
    ci_high: float | None
    # Set by the daily-cycle gate path only: how many overnight cycles were
    # pooled and the span they cover, so the gate can say what it scored.
    n_cycles: int | None = None
    span: str | None = None


def predict_price_changes(scores: pl.DataFrame, rise_threshold: float = DEFAULT_RISE_THRESHOLD) -> pl.DataFrame:
    """`scores`: element_id, net_transfers (24h, raw), eligible. Returns
    element_id, net_transfers, predicted_direction ("rise"/"stable" only —
    falls are not predictable from this data, see module docstring). A
    player is called a riser iff eligible and at or above `rise_threshold`."""
    return scores.select(
        "element_id",
        "net_transfers",
        pl.when(pl.col("eligible") & (pl.col("net_transfers") >= rise_threshold))
        .then(pl.lit("rise"))
        .otherwise(pl.lit("stable"))
        .alias("predicted_direction"),
    )


def _normal_approx_ci(hits: int, n: int, z: float = Z_95) -> tuple[float, float]:
    """Fine for the sample sizes this model will realistically see (dozens
    to low hundreds of rise calls per week) — wide and honest at small n
    rather than pretending precision it doesn't have."""
    if n == 0:
        return (0.0, 1.0)
    p = hits / n
    margin = z * ((p * (1 - p)) / n) ** 0.5
    return (max(0.0, p - margin), min(1.0, p + margin))


def evaluate_price_model(predictions: pl.DataFrame, actuals: pl.DataFrame) -> PriceModelEvaluation:
    """§6.5's "reports its hit rate with a stated confidence interval": of
    every player called to *move* (here only ever a rise — a "stable"
    prediction is trivially right most of the time and isn't the claim
    being tested), how often did the real direction match. A call is a hit
    only if `actual_direction` equals the predicted one. Never fabricates a
    rate when zero predictions called for a move.
    """
    joined = predictions.join(actuals, on="element_id", how="inner")
    moved = joined.filter(pl.col("predicted_direction") != "stable")
    n_moves = moved.height
    if n_moves == 0:
        return PriceModelEvaluation(n=joined.height, n_moves_predicted=0, hit_rate=None, ci_low=None, ci_high=None)
    hits = moved.filter(pl.col("predicted_direction") == pl.col("actual_direction")).height
    ci_low, ci_high = _normal_approx_ci(hits, n_moves)
    return PriceModelEvaluation(n=joined.height, n_moves_predicted=n_moves, hit_rate=hits / n_moves, ci_low=ci_low, ci_high=ci_high)


# --------------------------------------------------------------------------
# daily-cycle backtest and threshold fitting (see module docstring)
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class PriceCycle:
    pressure_from: datetime  # snapshot ~24h before prediction_ts
    prediction_ts: datetime  # last snapshot at or before the cycle's opening boundary
    # Last snapshot at or before the next boundary; None for the *current*
    # cycle, whose outcome has not happened yet (see `current_cycle`).
    evaluation_ts: datetime | None


def load_distilled_history(distilled_dir: Path) -> pl.DataFrame:
    """Every distilled row, read once. The backtest needs player state at
    hundreds of timestamps, and `state_as_of` re-scans every shard per call."""
    pattern = str(distilled_dir / "gw*" / "*.parquet").replace("\\", "/")
    columns = ["element_id", "snapshot_ts", *NUMERIC_DELTA_COLUMNS]
    if not distilled_dir.exists() or not any(distilled_dir.glob("gw*/*.parquet")):
        return pl.DataFrame(schema={c: (pl.Datetime("us", "UTC") if c == "snapshot_ts" else pl.Float64) for c in columns})
    con = duckdb.connect()
    try:
        con.execute("SET TimeZone='UTC'")
        return con.execute(f"SELECT {', '.join(columns)} FROM read_parquet('{pattern}')").pl().sort("snapshot_ts")
    finally:
        con.close()


def _state_at(history: pl.DataFrame, ts: datetime) -> pl.DataFrame:
    """`analytics.deltas.state_as_of`'s query, over an in-memory history."""
    return history.filter(pl.col("snapshot_ts") <= ts).group_by("element_id").last()


def _last_at_or_before(snapshots: list[datetime], ts: datetime) -> datetime | None:
    earlier = [s for s in snapshots if s <= ts]
    return earlier[-1] if earlier else None


def daily_cycles(snapshots: list[datetime], deadlines: list[datetime]) -> list[PriceCycle]:
    """One `PriceCycle` per boundary-to-boundary day the snapshots cover.

    Dropped rather than scored: a cycle with any bracketing snapshot more
    than `MAX_SNAPSHOT_STALENESS` from where it should be, and any cycle
    whose pressure window contains a gameweek deadline —
    `transfers_in_event`/`transfers_out_event` reset to zero at a deadline,
    so a 24h difference taken across one is not a transfer flow at all.
    """
    snapshots = sorted(snapshots)
    if not snapshots:
        return []
    cycles: list[PriceCycle] = []
    day, last_day = snapshots[0].date(), snapshots[-1].date()
    while day < last_day:
        open_boundary = datetime.combine(day, time(CYCLE_BOUNDARY_HOUR_UTC), tzinfo=timezone.utc)
        close_boundary = open_boundary + timedelta(days=1)
        day += timedelta(days=1)

        prediction_ts = _last_at_or_before(snapshots, open_boundary)
        evaluation_ts = _last_at_or_before(snapshots, close_boundary)
        if prediction_ts is None or evaluation_ts is None or evaluation_ts <= prediction_ts:
            continue
        if open_boundary - prediction_ts > MAX_SNAPSHOT_STALENESS or close_boundary - evaluation_ts > MAX_SNAPSHOT_STALENESS:
            continue
        pressure_target = prediction_ts - PRESSURE_WINDOW
        pressure_from = _last_at_or_before(snapshots, pressure_target)
        if pressure_from is None or pressure_target - pressure_from > MAX_SNAPSHOT_STALENESS:
            continue
        if any(pressure_from < deadline <= prediction_ts for deadline in deadlines):
            continue
        cycles.append(PriceCycle(pressure_from=pressure_from, prediction_ts=prediction_ts, evaluation_ts=evaluation_ts))
    return cycles


@dataclass(frozen=True)
class CurrentCycle:
    """Where the latest snapshot sits in the daily cycle, and whether it can
    be predicted from. `cycle` is None exactly when `status != "ok"`."""

    status: str  # "ok" | "no_data" | "stale" | "insufficient_history" | "deadline_in_window"
    detail: str | None
    cycle: PriceCycle | None
    boundary_ts: datetime | None  # the 19:00 UTC the cycle's prediction is (or was) taken at
    next_boundary_ts: datetime | None  # the following 19:00, where the next cycle opens
    locked: bool  # True once `boundary_ts` has passed: the call can no longer change


def current_cycle(snapshots: list[datetime], deadlines: list[datetime], now: datetime) -> CurrentCycle:
    """The cycle the latest snapshot belongs to, built from the same rules
    `daily_cycles` applies to history — a stale or gappy cycle is reported,
    never predicted from — but with no outcome, because it has not happened.

    Before today's boundary the cycle is *open*: its prediction snapshot is
    the latest one, so the score is a live running total that can still
    move before the 19:00 cut. After it the cycle is *locked*: the
    prediction snapshot is the last one at or before the boundary, exactly
    what a scored cycle would use, and the overnight price change is
    pending.

    Staleness is measured from `now`: a latest snapshot more than
    `MAX_SNAPSHOT_STALENESS` old (a collector outage) is not a current
    reading of anything. The pressure window needs a snapshot ~24h before
    the prediction and no gameweek deadline inside it, as for any cycle.
    """
    snapshots = sorted(snapshots)
    if not snapshots:
        return CurrentCycle("no_data", "no collector snapshots on disk", None, None, None, False)
    latest = snapshots[-1]
    boundary = datetime.combine(latest.date(), time(CYCLE_BOUNDARY_HOUR_UTC), tzinfo=timezone.utc)
    next_boundary = boundary + timedelta(days=1)
    locked = latest >= boundary

    def unavailable(status: str, detail: str) -> CurrentCycle:
        return CurrentCycle(status, detail, None, boundary, next_boundary, locked)

    if now - latest > MAX_SNAPSHOT_STALENESS:
        return unavailable("stale", f"latest snapshot {latest:%Y-%m-%d %H:%M} UTC is more than {MAX_SNAPSHOT_STALENESS} old")
    prediction_ts = _last_at_or_before(snapshots, boundary) if locked else latest
    if locked and (prediction_ts is None or boundary - prediction_ts > MAX_SNAPSHOT_STALENESS):
        return unavailable("stale", f"no snapshot within {MAX_SNAPSHOT_STALENESS} before the {boundary:%Y-%m-%d %H:%M} UTC boundary")
    pressure_target = prediction_ts - PRESSURE_WINDOW
    pressure_from = _last_at_or_before(snapshots, pressure_target)
    if pressure_from is None or pressure_target - pressure_from > MAX_SNAPSHOT_STALENESS:
        return unavailable("insufficient_history", "no snapshot near 24h before the prediction snapshot")
    if any(pressure_from < deadline <= prediction_ts for deadline in deadlines):
        return unavailable("deadline_in_window", "a gameweek deadline falls inside the 24h window, which resets transfer counts")
    cycle = PriceCycle(pressure_from=pressure_from, prediction_ts=prediction_ts, evaluation_ts=None)
    return CurrentCycle("ok", None, cycle, boundary, next_boundary, locked)


def cycle_observations(history: pl.DataFrame, cycles: list[PriceCycle]) -> pl.DataFrame:
    """One row per (cycle, player): `analytics.trending.price_change_pressure`
    over the cycle's 24h window (its `net_transfers` is the rise score;
    `price_change_pressure` is kept only as a diagnostic), whether the
    player is `eligible` (no `now_cost` change at any snapshot in the
    `RECENT_MOVE_WINDOW` up to the prediction), and the real direction his
    price moved across the overnight change."""
    frames = []
    for cycle in cycles:
        now = _state_at(history, cycle.prediction_ts)
        past = _state_at(history, cycle.pressure_from).select(
            "element_id", pl.col("transfers_in_event").alias("in_past"), pl.col("transfers_out_event").alias("out_past")
        )
        deltas = now.join(past, on="element_id", how="inner").select(
            "element_id",
            "selected_by_percent",
            (pl.col("transfers_in_event") - pl.col("in_past")).alias("transfers_in_event_delta_24h"),
            (pl.col("transfers_out_event") - pl.col("out_past")).alias("transfers_out_event_delta_24h"),
        )
        outcome = None
        if cycle.evaluation_ts is not None:
            after = _state_at(history, cycle.evaluation_ts).select("element_id", pl.col("now_cost").alias("cost_after"))
            outcome = now.select("element_id", "now_cost").join(after, on="element_id", how="inner").select(
                "element_id",
                pl.when(pl.col("cost_after") > pl.col("now_cost")).then(pl.lit("rise"))
                .when(pl.col("cost_after") < pl.col("now_cost")).then(pl.lit("fall"))
                .otherwise(pl.lit("stable"))
                .alias("actual_direction"),
            )
        # One filter + one group_by per cycle: a player moved recently iff his
        # now_cost is not constant across the window's snapshots.
        recent = (
            history.filter((pl.col("snapshot_ts") >= cycle.prediction_ts - RECENT_MOVE_WINDOW) & (pl.col("snapshot_ts") <= cycle.prediction_ts))
            .group_by("element_id")
            .agg((pl.col("now_cost").min() != pl.col("now_cost").max()).alias("moved_recently"))
        )
        pressure = price_change_pressure(deltas, window="24h")
        if outcome is not None:
            pressure = pressure.join(outcome, on="element_id", how="inner")
        frames.append(
            pressure
            .join(recent, on="element_id", how="left")
            .with_columns(
                (~pl.col("moved_recently").fill_null(False)).alias("eligible"),
                pl.lit(cycle.prediction_ts).alias("prediction_ts"),
            )
            .drop("moved_recently")
        )
    if not frames:
        return pl.DataFrame(schema={
            "element_id": pl.Int64, "net_transfers": pl.Float64, "price_change_pressure": pl.Float64,
            "actual_direction": pl.Utf8, "eligible": pl.Boolean, "prediction_ts": pl.Datetime("us", "UTC"),
        })
    return pl.concat(frames)


def evaluate_observations(observations: pl.DataFrame, rise_threshold: float) -> PriceModelEvaluation:
    """Pooled hit rate of rise calls over every (cycle, player) row:
    `predict_price_changes` and `evaluate_price_model`, keyed per cycle so
    one player's rows from different nights never join to each other."""
    keyed = observations.with_columns(
        (pl.col("prediction_ts").dt.epoch("s") * 10_000 + pl.col("element_id")).alias("key")
    )
    predictions = predict_price_changes(keyed.with_columns(pl.col("key").alias("element_id")), rise_threshold)
    actuals = keyed.select(pl.col("key").alias("element_id"), "actual_direction")
    return evaluate_price_model(predictions, actuals)


def _rise_f1(observations: pl.DataFrame, threshold: float) -> tuple[float, int]:
    """(F1 of calling a rise at `threshold`, number of such calls) over
    `observations`, which must already be restricted to eligible rows."""
    called = observations["net_transfers"] >= threshold
    actual = observations["actual_direction"] == "rise"
    tp, n_called, n_actual = int((called & actual).sum()), int(called.sum()), int(actual.sum())
    if tp == 0:
        return 0.0, n_called
    precision, recall = tp / n_called, tp / n_actual
    return 2 * precision * recall / (precision + recall), n_called


def fit_rise_threshold(observations: pl.DataFrame) -> dict[str, float | int]:
    """The single rise threshold, maximizing F1 of rise calls over a
    quantile grid of eligible players' positive 24h net transfers.

    Fitted on eligible rows only: an ineligible player can never be called,
    so letting him into the recall denominator (or the grid) would fit a
    threshold to players the model is forbidden to flag.

    F1 rather than raw hit rate: hit rate alone is maximized by calling
    one or two extreme players and being right, which says nothing.
    `MIN_FIT_PREDICTED_MOVES` rules out the same degenerate corner from
    the other side.
    """
    eligible = observations.filter(pl.col("eligible"))
    net = eligible["net_transfers"]
    side = net.filter(net > 0)
    grid = sorted({float(side.quantile(q / 400)) for q in range(1, 400)}) if side.len() else []
    scored = [(threshold, *_rise_f1(eligible, threshold)) for threshold in grid]
    scored = [s for s in scored if s[2] >= MIN_FIT_PREDICTED_MOVES]
    if not scored:
        raise ValueError(f"no rise threshold calls at least {MIN_FIT_PREDICTED_MOVES} moves over the fit span")
    threshold, f1, n_called = max(scored, key=lambda s: s[1])
    return {"rise_threshold": threshold, "rise_f1": f1, "rise_n_called": n_called}
