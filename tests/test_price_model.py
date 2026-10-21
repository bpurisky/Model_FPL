"""§4.1/§6.4/§6.5 price-change model: prediction, real-outcome extraction,
and the hit-rate-with-CI evaluator."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import polars as pl
import pytest

from analytics.deltas import _STATE_SCHEMA
from analytics.price_model import (
    PriceModelEvaluation,
    PriceThresholds,
    actual_price_direction,
    cycle_observations,
    daily_cycles,
    evaluate_observations,
    evaluate_price_model,
    fit_thresholds,
    load_distilled_history,
    load_thresholds,
    predict_price_changes,
    run_price_model_evaluation,
)
from papertrade.evaluate import run_price_model_for_gate

BASE = datetime(2026, 8, 21, 12, 0, tzinfo=timezone.utc)


def test_predict_price_changes_thresholds():
    pressure = pl.DataFrame([
        {"element_id": 1, "net_transfers": 0, "price_change_pressure": 2.0},   # >= 0.5 -> rise
        {"element_id": 2, "net_transfers": 0, "price_change_pressure": -2.0},  # <= -0.5 -> fall
        {"element_id": 3, "net_transfers": 0, "price_change_pressure": 0.1},   # between -> stable
    ])
    result = predict_price_changes(pressure, rise_threshold=0.5, fall_threshold=-0.5)
    by_id = {row["element_id"]: row["predicted_direction"] for row in result.to_dicts()}
    assert by_id == {1: "rise", 2: "fall", 3: "stable"}


def _row(ts, eid, now_cost) -> dict:
    return {
        "snapshot_ts": ts, "element_id": eid, "now_cost": now_cost, "selected_by_percent": 10.0,
        "transfers_in_event": 0, "transfers_out_event": 0, "form": 5.0, "status": "a",
        "chance_of_playing_next_round": None, "news_added": None, "ep_next": 4.0,
    }


def _write_shard(gw_dir, ts, rows) -> None:
    gw_dir.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(rows, schema=_STATE_SCHEMA).write_parquet(gw_dir / f"{ts.strftime('%Y%m%dT%H%M%SZ')}.parquet")


def test_actual_price_direction_classifies_rise_fall_stable(tmp_path):
    gw1 = tmp_path / "gw1"
    _write_shard(gw1, BASE, [_row(BASE, 1, 50), _row(BASE, 2, 50), _row(BASE, 3, 50)])
    after = BASE + timedelta(hours=24)
    _write_shard(gw1, after, [_row(after, 1, 51), _row(after, 2, 49), _row(after, 3, 50)])

    result = actual_price_direction(tmp_path, BASE, after)
    by_id = {row["element_id"]: row["actual_direction"] for row in result.to_dicts()}
    assert by_id == {1: "rise", 2: "fall", 3: "stable"}


def test_evaluate_price_model_computes_hit_rate_and_ci():
    predictions = pl.DataFrame([
        {"element_id": 1, "predicted_direction": "rise"},
        {"element_id": 2, "predicted_direction": "rise"},
        {"element_id": 3, "predicted_direction": "fall"},
        {"element_id": 4, "predicted_direction": "stable"},  # excluded from the hit-rate denominator
    ])
    actuals = pl.DataFrame([
        {"element_id": 1, "actual_direction": "rise"},   # hit
        {"element_id": 2, "actual_direction": "stable"},  # miss
        {"element_id": 3, "actual_direction": "fall"},    # hit
        {"element_id": 4, "actual_direction": "stable"},
    ])

    result = evaluate_price_model(predictions, actuals)

    assert result.n == 4
    assert result.n_moves_predicted == 3
    assert result.hit_rate == pytest.approx(2 / 3)
    assert result.ci_low is not None and result.ci_high is not None
    assert result.ci_low <= result.hit_rate <= result.ci_high


def test_evaluate_price_model_never_fabricates_a_rate_with_zero_moves():
    predictions = pl.DataFrame([{"element_id": 1, "predicted_direction": "stable"}])
    actuals = pl.DataFrame([{"element_id": 1, "actual_direction": "stable"}])

    result = evaluate_price_model(predictions, actuals)

    assert result.n_moves_predicted == 0
    assert result.hit_rate is None
    assert result.ci_low is None and result.ci_high is None


def test_evaluate_price_model_wider_ci_at_smaller_n():
    small = evaluate_price_model(
        pl.DataFrame([{"element_id": i, "predicted_direction": "rise"} for i in range(4)]),
        pl.DataFrame([{"element_id": i, "actual_direction": "rise" if i < 3 else "stable"} for i in range(4)]),
    )
    large = evaluate_price_model(
        pl.DataFrame([{"element_id": i, "predicted_direction": "rise"} for i in range(400)]),
        pl.DataFrame([{"element_id": i, "actual_direction": "rise" if i < 300 else "stable"} for i in range(400)]),
    )
    # same ~75% hit rate, but far more observations -> a tighter interval
    assert (small.ci_high - small.ci_low) > (large.ci_high - large.ci_low)


def test_run_price_model_evaluation_end_to_end_with_synthetic_shards(tmp_path):
    gw1 = tmp_path / "gw1"
    predict_ts = BASE
    eval_ts = BASE + timedelta(hours=24)
    baseline_ts = BASE - timedelta(hours=1)

    _write_shard(gw1, baseline_ts, [_row(baseline_ts, 1, 50)])
    # heavy net-in transfer pressure at prediction time -> should predict "rise"
    row_at_predict = _row(predict_ts, 1, 50)
    row_at_predict["transfers_in_event"] = 5000
    row_at_predict["selected_by_percent"] = 5.0
    _write_shard(gw1, predict_ts, [row_at_predict])
    _write_shard(gw1, eval_ts, [_row(eval_ts, 1, 51)])  # price actually rose

    result = run_price_model_evaluation(tmp_path, predict_ts, eval_ts, pressure_window="1h", rise_threshold=0.5, fall_threshold=-0.5)

    assert isinstance(result, PriceModelEvaluation)
    assert result.n_moves_predicted == 1
    assert result.hit_rate == pytest.approx(1.0)


def test_run_price_model_evaluation_returns_empty_when_no_data(tmp_path):
    result = run_price_model_evaluation(tmp_path, BASE, BASE + timedelta(hours=24))
    assert result.n == 0
    assert result.hit_rate is None


# --------------------------------------------------------------------------
# daily cycles, fitting, and the out-of-sample gate path
# --------------------------------------------------------------------------


def _at(day: int, hour: int) -> datetime:
    return datetime(2026, 9, day, hour, tzinfo=timezone.utc)


def test_daily_cycles_bracket_the_overnight_change():
    snapshots = [_at(1, 18), _at(2, 18), _at(3, 18)]
    cycles = daily_cycles(snapshots, deadlines=[])
    # cycle opening 2 Sep 19:00 needs pressure from ~1 Sep 18:00 and closes 3 Sep 19:00
    assert [(c.pressure_from, c.prediction_ts, c.evaluation_ts) for c in cycles] == [(_at(1, 18), _at(2, 18), _at(3, 18))]


def test_a_cycle_whose_pressure_window_spans_a_deadline_is_dropped():
    """transfers_*_event reset at a deadline -- a 24h difference across one
    is not a transfer flow."""
    snapshots = [_at(1, 18), _at(2, 18), _at(3, 18)]
    assert daily_cycles(snapshots, deadlines=[_at(2, 10)]) == []


def test_a_cycle_with_a_stale_snapshot_is_dropped():
    # nothing between 1 Sep 18:00 and 3 Sep 18:00: 2 Sep's boundary is 25h stale
    assert daily_cycles([_at(1, 18), _at(3, 18), _at(4, 18)], deadlines=[]) == []


def _cycle_shards(tmp_path, days: range, pressure_by_eid: dict[int, int], moves: dict[int, int]) -> None:
    """One shard per day at 18:00. transfers_in_event grows by
    pressure_by_eid[eid] a day; now_cost steps by moves[eid] each night."""
    for i, day in enumerate(days):
        ts = _at(day, 18)
        rows = []
        for eid, daily_in in pressure_by_eid.items():
            row = _row(ts, eid, 50 + i * moves.get(eid, 0))
            row["transfers_in_event"] = daily_in * i
            row["selected_by_percent"] = 1.0
            rows.append(row)
        _write_shard(tmp_path / "gw1", ts, rows)


def test_cycle_observations_and_pooled_evaluation(tmp_path):
    # player 1: heavy inflow and rises nightly; player 2: no flow, stable
    _cycle_shards(tmp_path, range(1, 6), {1: 2000, 2: 0}, {1: 1})
    history = load_distilled_history(tmp_path)
    cycles = daily_cycles(history["snapshot_ts"].unique().to_list(), deadlines=[])

    observations = cycle_observations(history, cycles)

    assert len(cycles) == 3
    assert observations.height == 6  # 2 players x 3 cycles
    p1 = observations.filter(pl.col("element_id") == 1)
    assert p1["price_change_pressure"].to_list() == [1000.0] * 3  # 2000 / (1% + 1)
    assert p1["actual_direction"].to_list() == ["rise"] * 3

    result = evaluate_observations(observations, rise_threshold=500, fall_threshold=-500)
    # one player's three nights stay three separate observations
    assert result.n == 6
    assert result.n_moves_predicted == 3
    assert result.hit_rate == pytest.approx(1.0)


def test_fit_thresholds_refuses_a_fit_with_too_few_calls():
    observations = pl.DataFrame({
        "element_id": [1, 2], "price_change_pressure": [10.0, -10.0], "actual_direction": ["rise", "fall"],
    })
    with pytest.raises(ValueError, match="at least"):
        fit_thresholds(observations)


def test_missing_config_means_unfitted_placeholders(tmp_path):
    thresholds = load_thresholds(tmp_path / "absent.yaml")
    assert (thresholds.rise, thresholds.fall, thresholds.fit_cutoff) == (0.5, -0.5, None)


def _events(tmp_path):
    path = tmp_path / "events.parquet"
    pl.DataFrame({"id": [1], "deadline_time": [_at(1, 1)]}, schema={"id": pl.Int64, "deadline_time": pl.Datetime("us", "UTC")}).write_parquet(path)
    return path


def test_gate_scores_only_cycles_after_the_fit_cutoff(tmp_path):
    distilled = tmp_path / "distilled"
    _cycle_shards(distilled, range(1, 8), {1: 2000}, {1: 1})
    cutoff = PriceThresholds(rise=500, fall=-500, fit_cutoff=_at(4, 0).date())

    result = run_price_model_for_gate(distilled, _events(tmp_path), thresholds=cutoff)

    # cycles open 2..6 Sep; only 4, 5, 6 Sep are on or after the cutoff
    assert result.n_cycles == 3
    assert result.span == "2026-09-04 to 2026-09-06"
    assert result.hit_rate == pytest.approx(1.0)


def test_gate_has_no_out_of_sample_span_without_a_fit(tmp_path):
    distilled = tmp_path / "distilled"
    _cycle_shards(distilled, range(1, 5), {1: 2000}, {1: 1})
    unfitted = PriceThresholds(rise=0.5, fall=-0.5, fit_cutoff=None)

    result = run_price_model_for_gate(distilled, _events(tmp_path), thresholds=unfitted)

    assert result.n_cycles == 0
    assert result.hit_rate is None
