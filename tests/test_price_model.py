"""§4.1/§6.4/§6.5 price-change model: prediction, real-outcome extraction,
and the hit-rate-with-CI evaluator."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import polars as pl
import pytest

from analytics.deltas import _STATE_SCHEMA
from analytics.price_model import (
    PriceModelEvaluation,
    actual_price_direction,
    evaluate_price_model,
    predict_price_changes,
    run_price_model_evaluation,
)

BASE = datetime(2026, 8, 21, 12, 0, tzinfo=timezone.utc)


def test_predict_price_changes_thresholds():
    pressure = pl.DataFrame([
        {"element_id": 1, "net_transfers": 0, "price_change_pressure": 2.0},   # >= 0.5 -> rise
        {"element_id": 2, "net_transfers": 0, "price_change_pressure": -2.0},  # <= -0.5 -> fall
        {"element_id": 3, "net_transfers": 0, "price_change_pressure": 0.1},   # between -> stable
    ])
    result = predict_price_changes(pressure)
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

    result = run_price_model_evaluation(tmp_path, predict_ts, eval_ts, pressure_window="1h")

    assert isinstance(result, PriceModelEvaluation)
    assert result.n_moves_predicted == 1
    assert result.hit_rate == pytest.approx(1.0)


def test_run_price_model_evaluation_returns_empty_when_no_data(tmp_path):
    result = run_price_model_evaluation(tmp_path, BASE, BASE + timedelta(hours=24))
    assert result.n == 0
    assert result.hit_rate is None
