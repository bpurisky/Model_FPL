"""§4.1/§6.4/§6.5's price-change model: predicts which players are likely
to rise or fall in price from `analytics/trending.py`'s pressure signal,
and evaluates that prediction's hit rate against real observed price
changes — with an explicit, wide confidence interval (§6.4: market signals
have no historical analogue and gw13 is their *first* evaluation window,
not a settled one; never report a bare point estimate).

No scipy in the locked stack (§1.1), so the confidence interval is a
standard normal approximation for a binomial proportion — matching
`backtest/report.py`'s own precedent of computing Spearman without scipy.

**Thresholds are fitted once, by hand, and then scored only
out-of-sample.** `fit_thresholds` grid-searches the rise and fall cut-offs
over the daily price-change cycles before a cut-off date and writes them,
with that date, to `config/price_model.yaml`
(`uv run python -m analytics fit-price-thresholds`). The launch gate
(`papertrade/evaluate.py:run_price_model_for_gate`) scores only the cycles
*after* `fit_cutoff`, so the hit rate it reports was never seen by the
fit. Refitting moves the cut-off forward and shrinks that out-of-sample
span back to nothing — do it deliberately, not on a schedule. Without the
config file the old unfitted ±0.5 placeholders apply (and they were never
meaningful: 24h pressure runs into the thousands, so ±0.5 flags nearly
every player with any net transfer at all).

**Why daily cycles, not a fixed lookback from "now".** FPL moves prices
once a night (observed 2026-27: always between ~22:30 and ~02:00 UTC,
never between 08:00 and 19:00). The collector's snapshots land every ~4h
in practice, not hourly, so the original 1h pressure window usually
compared a snapshot with itself and predicted nothing. A cycle runs from
one `CYCLE_BOUNDARY_HOUR_UTC` to the next: pressure is the 24h net-transfer
flow up to the last snapshot before the boundary, and the outcome is the
price change between that snapshot and the last one before the *next*
boundary — which brackets the overnight change whatever minute it lands.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path

import duckdb
import polars as pl
import yaml

from analytics.deltas import NUMERIC_DELTA_COLUMNS, compute_deltas, reference_timestamps, state_as_of
from analytics.trending import price_change_pressure

Z_95 = 1.959963985  # standard normal critical value for a 95% CI

PRICE_MODEL_CONFIG = Path("config/price_model.yaml")

UNFITTED_RISE_THRESHOLD = 0.5
UNFITTED_FALL_THRESHOLD = -0.5

# Any hour inside the observed 08:00-19:00 UTC quiet span works; 19:00
# keeps the pressure window as close to the overnight change as it allows.
CYCLE_BOUNDARY_HOUR_UTC = 19
PRESSURE_WINDOW = timedelta(hours=24)
# A cycle whose bracketing snapshot sits further than this from where it
# should be (a collector outage) is dropped rather than scored on stale data.
MAX_SNAPSHOT_STALENESS = timedelta(hours=8)
# A threshold that calls fewer moves than this over the fit span is
# fitting a handful of lucky players, not a signal.
MIN_FIT_PREDICTED_MOVES = 20


@dataclass(frozen=True)
class PriceThresholds:
    rise: float
    fall: float
    fit_cutoff: date | None  # None: unfitted placeholders, nothing is out-of-sample


def load_thresholds(path: Path = PRICE_MODEL_CONFIG) -> PriceThresholds:
    if not path.exists():
        return PriceThresholds(rise=UNFITTED_RISE_THRESHOLD, fall=UNFITTED_FALL_THRESHOLD, fit_cutoff=None)
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    return PriceThresholds(rise=float(raw["rise_threshold"]), fall=float(raw["fall_threshold"]), fit_cutoff=raw["fit_cutoff"])


_CONFIGURED = load_thresholds()
DEFAULT_RISE_THRESHOLD = _CONFIGURED.rise
DEFAULT_FALL_THRESHOLD = _CONFIGURED.fall


@dataclass(frozen=True)
class PriceModelEvaluation:
    n: int  # players with both a prediction and a known actual outcome
    n_moves_predicted: int  # of those, how many were predicted to actually move (rise or fall)
    hit_rate: float | None  # None, never a fabricated 0.0, when n_moves_predicted == 0
    ci_low: float | None
    ci_high: float | None
    # Set by the daily-cycle gate path only: how many overnight cycles were
    # pooled and the span they cover, so the gate can say what it scored.
    n_cycles: int | None = None
    span: str | None = None


def predict_price_changes(
    pressure: pl.DataFrame, rise_threshold: float = DEFAULT_RISE_THRESHOLD, fall_threshold: float = DEFAULT_FALL_THRESHOLD
) -> pl.DataFrame:
    """`pressure`: `analytics.trending.price_change_pressure`'s output
    (element_id, net_transfers, price_change_pressure). Returns element_id,
    price_change_pressure, predicted_direction ("rise"/"fall"/"stable")."""
    return pressure.select(
        "element_id",
        "price_change_pressure",
        pl.when(pl.col("price_change_pressure") >= rise_threshold)
        .then(pl.lit("rise"))
        .when(pl.col("price_change_pressure") <= fall_threshold)
        .then(pl.lit("fall"))
        .otherwise(pl.lit("stable"))
        .alias("predicted_direction"),
    )


def actual_price_direction(distilled_dir: Path, before: datetime, after: datetime) -> pl.DataFrame:
    """element_id, actual_direction ("rise"/"fall"/"stable") from real
    `now_cost` movement between two points in time — only players with a
    known state at both ends are included (inner join), never a guessed
    baseline."""
    before_state = state_as_of(distilled_dir, before).select("element_id", pl.col("now_cost").alias("now_cost_before"))
    after_state = state_as_of(distilled_dir, after).select("element_id", pl.col("now_cost").alias("now_cost_after"))
    joined = before_state.join(after_state, on="element_id", how="inner")
    return joined.select(
        "element_id",
        pl.when(pl.col("now_cost_after") > pl.col("now_cost_before"))
        .then(pl.lit("rise"))
        .when(pl.col("now_cost_after") < pl.col("now_cost_before"))
        .then(pl.lit("fall"))
        .otherwise(pl.lit("stable"))
        .alias("actual_direction"),
    )


def _normal_approx_ci(hits: int, n: int, z: float = Z_95) -> tuple[float, float]:
    """Fine for the sample sizes this model will realistically see (dozens
    to low hundreds of "moved" predictions per week) — wide and honest at
    small n rather than pretending precision it doesn't have."""
    if n == 0:
        return (0.0, 1.0)
    p = hits / n
    margin = z * ((p * (1 - p)) / n) ** 0.5
    return (max(0.0, p - margin), min(1.0, p + margin))


def evaluate_price_model(predictions: pl.DataFrame, actuals: pl.DataFrame) -> PriceModelEvaluation:
    """§6.5's "reports its hit rate with a stated confidence interval": of
    every player predicted to actually *move* (rise or fall — a "stable"
    prediction is trivially right most of the time and isn't the claim
    being tested), how often did the real direction match. Never
    fabricates a rate when zero predictions called for a move.
    """
    joined = predictions.join(actuals, on="element_id", how="inner")
    moved = joined.filter(pl.col("predicted_direction") != "stable")
    n_moves = moved.height
    if n_moves == 0:
        return PriceModelEvaluation(n=joined.height, n_moves_predicted=0, hit_rate=None, ci_low=None, ci_high=None)
    hits = moved.filter(pl.col("predicted_direction") == pl.col("actual_direction")).height
    ci_low, ci_high = _normal_approx_ci(hits, n_moves)
    return PriceModelEvaluation(n=joined.height, n_moves_predicted=n_moves, hit_rate=hits / n_moves, ci_low=ci_low, ci_high=ci_high)


def run_price_model_evaluation(
    distilled_dir: Path,
    prediction_ts: datetime,
    evaluation_ts: datetime,
    pressure_window: str = "1h",
    rise_threshold: float = DEFAULT_RISE_THRESHOLD,
    fall_threshold: float = DEFAULT_FALL_THRESHOLD,
) -> PriceModelEvaluation:
    """The live-data path: predict from pressure as of `prediction_ts`,
    then check the real outcome as of `evaluation_ts` (typically 24h later
    — FPL price changes land roughly once a day). Ties together
    analytics/deltas.py, analytics/trending.py, and this module's own
    predict/evaluate pair.
    """
    refs = reference_timestamps(prediction_ts)
    if pressure_window not in refs:
        refs[pressure_window] = prediction_ts  # caller-supplied window not one of the standard three
    deltas = compute_deltas(distilled_dir, prediction_ts, refs)
    if deltas.height == 0:
        return PriceModelEvaluation(n=0, n_moves_predicted=0, hit_rate=None, ci_low=None, ci_high=None)

    pressure = price_change_pressure(deltas, window=pressure_window)
    predictions = predict_price_changes(pressure, rise_threshold, fall_threshold)
    actuals = actual_price_direction(distilled_dir, prediction_ts, evaluation_ts)
    return evaluate_price_model(predictions, actuals)


# --------------------------------------------------------------------------
# daily-cycle backtest and threshold fitting (see module docstring)
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class PriceCycle:
    pressure_from: datetime  # snapshot ~24h before prediction_ts
    prediction_ts: datetime  # last snapshot at or before the cycle's opening boundary
    evaluation_ts: datetime  # last snapshot at or before the next boundary


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


def cycle_observations(history: pl.DataFrame, cycles: list[PriceCycle]) -> pl.DataFrame:
    """One row per (cycle, player): `analytics.trending.price_change_pressure`
    over the cycle's 24h window, and the real direction its price moved
    across the overnight change."""
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
        after = _state_at(history, cycle.evaluation_ts).select("element_id", pl.col("now_cost").alias("cost_after"))
        outcome = now.select("element_id", "now_cost").join(after, on="element_id", how="inner").select(
            "element_id",
            pl.when(pl.col("cost_after") > pl.col("now_cost")).then(pl.lit("rise"))
            .when(pl.col("cost_after") < pl.col("now_cost")).then(pl.lit("fall"))
            .otherwise(pl.lit("stable"))
            .alias("actual_direction"),
        )
        frames.append(
            price_change_pressure(deltas, window="24h")
            .join(outcome, on="element_id", how="inner")
            .with_columns(pl.lit(cycle.prediction_ts).alias("prediction_ts"))
        )
    if not frames:
        return pl.DataFrame(schema={
            "element_id": pl.Int64, "net_transfers": pl.Float64, "price_change_pressure": pl.Float64,
            "actual_direction": pl.Utf8, "prediction_ts": pl.Datetime("us", "UTC"),
        })
    return pl.concat(frames)


def evaluate_observations(observations: pl.DataFrame, rise_threshold: float, fall_threshold: float) -> PriceModelEvaluation:
    """Pooled hit rate over every (cycle, player) row: `predict_price_changes`
    and `evaluate_price_model`, keyed per cycle so one player's rows from
    different nights never join to each other."""
    keyed = observations.with_columns(
        (pl.col("prediction_ts").dt.epoch("s") * 10_000 + pl.col("element_id")).alias("key")
    )
    predictions = predict_price_changes(keyed.with_columns(pl.col("key").alias("element_id")), rise_threshold, fall_threshold)
    actuals = keyed.select(pl.col("key").alias("element_id"), "actual_direction")
    return evaluate_price_model(predictions, actuals)


def _f1(observations: pl.DataFrame, direction: str, threshold: float) -> tuple[float, int]:
    """(F1 of calling `direction` at `threshold`, number of such calls)."""
    pressure = observations["price_change_pressure"]
    called = pressure >= threshold if direction == "rise" else pressure <= threshold
    actual = observations["actual_direction"] == direction
    tp, n_called, n_actual = int((called & actual).sum()), int(called.sum()), int(actual.sum())
    if tp == 0:
        return 0.0, n_called
    precision, recall = tp / n_called, tp / n_actual
    return 2 * precision * recall / (precision + recall), n_called


def fit_thresholds(observations: pl.DataFrame) -> dict[str, float | int]:
    """Rise and fall thresholds fitted independently, each maximizing F1 of
    its own direction's calls over a quantile grid of that side's pressure.

    F1 rather than raw hit rate: hit rate alone is maximized by calling
    one or two extreme players and being right, which says nothing.
    `MIN_FIT_PREDICTED_MOVES` rules out the same degenerate corner from
    the other side.
    """
    fitted: dict[str, float | int] = {}
    for direction in ("rise", "fall"):
        pressure = observations["price_change_pressure"]
        side = pressure.filter(pressure > 0) if direction == "rise" else pressure.filter(pressure < 0)
        grid = sorted({float(side.quantile(q / 400)) for q in range(1, 400)}) if side.len() else []
        scored = [(threshold, *_f1(observations, direction, threshold)) for threshold in grid]
        scored = [s for s in scored if s[2] >= MIN_FIT_PREDICTED_MOVES]
        if not scored:
            raise ValueError(f"no {direction} threshold calls at least {MIN_FIT_PREDICTED_MOVES} moves over the fit span")
        threshold, f1, n_called = max(scored, key=lambda s: s[1])
        fitted[f"{direction}_threshold"] = threshold
        fitted[f"{direction}_f1"] = f1
        fitted[f"{direction}_n_called"] = n_called
    return fitted
