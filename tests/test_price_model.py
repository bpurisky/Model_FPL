"""§4.1/§6.4/§6.5 price-change model: rise-only prediction, 72h eligibility,
rise-threshold fitting, and the hit-rate-with-CI evaluator."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Callable

import polars as pl
import pytest

from analytics.deltas import _STATE_SCHEMA
from analytics.price_model import (
    MIN_FIT_PREDICTED_MOVES,
    PriceModelEvaluation,
    PriceThresholds,
    cycle_observations,
    daily_cycles,
    evaluate_observations,
    evaluate_price_model,
    fit_rise_threshold,
    load_distilled_history,
    load_thresholds,
    predict_price_changes,
)
from papertrade.evaluate import run_price_model_for_gate

BASE = datetime(2026, 8, 21, 12, 0, tzinfo=timezone.utc)


def test_predict_price_changes_rises_only_and_only_when_eligible():
    scores = pl.DataFrame([
        {"element_id": 1, "net_transfers": 5000.0, "eligible": True},     # >= threshold -> rise
        {"element_id": 2, "net_transfers": -9000.0, "eligible": True},    # heavy net outflow: still never a fall
        {"element_id": 3, "net_transfers": 100.0, "eligible": True},      # below threshold -> stable
        {"element_id": 4, "net_transfers": 5000.0, "eligible": False},    # moved recently -> stable
    ])
    result = predict_price_changes(scores, rise_threshold=500)
    by_id = {row["element_id"]: row["predicted_direction"] for row in result.to_dicts()}
    assert by_id == {1: "rise", 2: "stable", 3: "stable", 4: "stable"}


def _row(ts, eid, now_cost) -> dict:
    return {
        "snapshot_ts": ts, "element_id": eid, "now_cost": now_cost, "selected_by_percent": 10.0,
        "transfers_in_event": 0, "transfers_out_event": 0, "form": 5.0, "status": "a",
        "chance_of_playing_next_round": None, "news_added": None, "ep_next": 4.0,
    }


def _write_shard(gw_dir, ts, rows) -> None:
    gw_dir.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(rows, schema=_STATE_SCHEMA).write_parquet(gw_dir / f"{ts.strftime('%Y%m%dT%H%M%SZ')}.parquet")


def test_evaluate_price_model_computes_hit_rate_and_ci():
    predictions = pl.DataFrame([
        {"element_id": 1, "predicted_direction": "rise"},
        {"element_id": 2, "predicted_direction": "rise"},
        {"element_id": 3, "predicted_direction": "rise"},
        {"element_id": 4, "predicted_direction": "stable"},  # excluded from the hit-rate denominator
    ])
    actuals = pl.DataFrame([
        {"element_id": 1, "actual_direction": "rise"},   # hit
        {"element_id": 2, "actual_direction": "stable"},  # miss
        {"element_id": 3, "actual_direction": "fall"},    # a call is a hit only if the price actually rose
        {"element_id": 4, "actual_direction": "stable"},
    ])

    result = evaluate_price_model(predictions, actuals)

    assert result.n == 4
    assert result.n_moves_predicted == 3
    assert result.hit_rate == pytest.approx(1 / 3)
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


def _cycle_shards(tmp_path, days: range, inflow_by_eid: dict[int, int], cost_on_day: dict[int, Callable[[int], int]] | None = None) -> None:
    """One shard per day at 18:00. transfers_in_event grows by
    inflow_by_eid[eid] a day; now_cost is cost_on_day[eid](day), flat 50 by default."""
    cost_on_day = cost_on_day or {}
    for i, day in enumerate(days):
        ts = _at(day, 18)
        rows = []
        for eid, daily_in in inflow_by_eid.items():
            row = _row(ts, eid, cost_on_day.get(eid, lambda d: 50)(day))
            row["transfers_in_event"] = daily_in * i
            row["selected_by_percent"] = 1.0
            rows.append(row)
        _write_shard(tmp_path / "gw1", ts, rows)


def _rises_on(*days: int) -> Callable[[int], int]:
    """Price steps up by one on each listed day's snapshot."""
    return lambda day: 50 + sum(1 for d in days if day >= d)


def _observations(tmp_path, inflow, cost_on_day, days=range(1, 9)):
    _cycle_shards(tmp_path, days, inflow, cost_on_day)
    history = load_distilled_history(tmp_path)
    cycles = daily_cycles(history["snapshot_ts"].unique().to_list(), deadlines=[])
    return cycle_observations(history, cycles)


def test_cycle_observations_score_is_raw_net_transfers(tmp_path):
    # player 1: heavy inflow and rises on 5 Sep; player 2: no flow, stable
    observations = _observations(tmp_path, {1: 2000, 2: 0}, {1: _rises_on(5)})

    assert observations.height == 12  # 2 players x 6 cycles (opening 2..7 Sep)
    p1 = observations.filter(pl.col("element_id") == 1).sort("prediction_ts")
    assert p1["net_transfers"].to_list() == [2000.0] * 6  # raw, not divided by ownership
    assert p1["price_change_pressure"].to_list() == [1000.0] * 6  # kept as a diagnostic: 2000 / (1% + 1)
    # the 5 Sep step lands in the cycle that predicts at 4 Sep 18:00 and evaluates at 5 Sep 18:00
    assert p1["actual_direction"].to_list() == ["stable", "stable", "rise", "stable", "stable", "stable"]


def test_eligibility_excludes_a_price_move_within_72h_and_keeps_one_73h_plus_ago(tmp_path):
    # The cycle predicting at 7 Sep 18:00 sees snapshots 4 Sep 18:00 .. 7 Sep 18:00.
    #   player 1: cost steps at the 5 Sep snapshot -> inside the window, ineligible
    #   player 2: cost steps at the 3 Sep snapshot -> 96h back, outside it, eligible
    #   player 3: never moves, eligible
    observations = _observations(tmp_path, {1: 2000, 2: 2000, 3: 2000}, {1: _rises_on(5), 2: _rises_on(3)})
    last = observations.filter(pl.col("prediction_ts") == _at(7, 18))
    eligible = {row["element_id"]: row["eligible"] for row in last.to_dicts()}
    assert eligible == {1: False, 2: True, 3: True}

    # and the ineligible one is never called, however heavy its inflow
    predictions = predict_price_changes(last, rise_threshold=500)
    called = {row["element_id"] for row in predictions.filter(pl.col("predicted_direction") == "rise").to_dicts()}
    assert called == {2, 3}


def test_no_fall_prediction_is_ever_emitted(tmp_path):
    # player 1: heavy net outflow and falls; player 2: heavy inflow and rises
    observations = _observations(tmp_path, {1: -2000, 2: 2000}, {1: lambda d: 50 - (d >= 5), 2: _rises_on(5)})
    assert "fall" in set(observations["actual_direction"])

    for threshold in (-1e9, -500, 0, 500):
        predictions = predict_price_changes(observations, rise_threshold=threshold)
        assert set(predictions["predicted_direction"]) <= {"rise", "stable"}


def test_evaluate_observations_scores_rise_calls_only(tmp_path):
    observations = _observations(tmp_path, {1: 2000, 2: 2000}, {1: _rises_on(5)})
    result = evaluate_observations(observations, rise_threshold=500)
    # player 1 is eligible on 2, 3, 4 Sep only (5, 6, 7 Sep sit inside the 72h
    # window of his 5 Sep step); player 2 is called on all six nights
    assert result.n == 12
    assert result.n_moves_predicted == 3 + 6
    assert result.hit_rate == pytest.approx(1 / 9)  # only player 1's 4 Sep call actually rose


def _fit_observations(n_hits: int, n_misses: int, n_ineligible_hits: int = 0) -> pl.DataFrame:
    """Big net transfers: n_hits that rose, n_misses that did not, plus
    ineligible players that rose far above them; noise below."""
    rows = [
        {"element_id": i, "net_transfers": 10_000.0 + i, "actual_direction": "rise", "eligible": True}
        for i in range(n_hits)
    ] + [
        {"element_id": 1_000 + i, "net_transfers": 10_000.0 + i, "actual_direction": "stable", "eligible": True}
        for i in range(n_misses)
    ] + [
        {"element_id": 2_000 + i, "net_transfers": 90_000.0 + i, "actual_direction": "rise", "eligible": False}
        for i in range(n_ineligible_hits)
    ] + [
        {"element_id": 3_000 + i, "net_transfers": float(i), "actual_direction": "stable", "eligible": True}
        for i in range(500)
    ]
    return pl.DataFrame(rows)


def test_fit_rise_threshold_returns_a_rise_threshold_only():
    fitted = fit_rise_threshold(_fit_observations(n_hits=60, n_misses=10))
    assert set(fitted) == {"rise_threshold", "rise_f1", "rise_n_called"}
    assert fitted["rise_n_called"] >= MIN_FIT_PREDICTED_MOVES
    assert 0 < fitted["rise_f1"] <= 1


def test_fit_rise_threshold_ignores_ineligible_rows():
    # ineligible risers sit far above every eligible player; had they counted,
    # the best threshold would sit up among them
    fitted = fit_rise_threshold(_fit_observations(n_hits=60, n_misses=10, n_ineligible_hits=200))
    assert fitted["rise_threshold"] < 90_000


def test_fit_rise_threshold_refuses_a_fit_with_too_few_calls():
    observations = pl.DataFrame({
        "element_id": [1, 2], "net_transfers": [10.0, 20.0], "actual_direction": ["rise", "stable"], "eligible": [True, True],
    })
    with pytest.raises(ValueError, match="at least"):
        fit_rise_threshold(observations)


def test_missing_config_means_unfitted_placeholder(tmp_path):
    thresholds = load_thresholds(tmp_path / "absent.yaml")
    assert (thresholds.rise, thresholds.fit_cutoff) == (0.5, None)


def _events(tmp_path):
    path = tmp_path / "events.parquet"
    pl.DataFrame({"id": [1], "deadline_time": [_at(1, 1)]}, schema={"id": pl.Int64, "deadline_time": pl.Datetime("us", "UTC")}).write_parquet(path)
    return path


def test_gate_scores_only_cycles_after_the_fit_cutoff_and_only_rise_calls(tmp_path):
    distilled = tmp_path / "distilled"
    # player 1 rises on 5 and 10 Sep; player 2 never moves. Both have the same inflow.
    _cycle_shards(distilled, range(1, 13), {1: 2000, 2: 2000}, {1: _rises_on(5, 10)})
    cutoff = PriceThresholds(rise=500, fit_cutoff=_at(4, 0).date())

    result = run_price_model_for_gate(distilled, _events(tmp_path), thresholds=cutoff)

    # cycles open 2..11 Sep; 4..11 Sep (8 cycles) are on or after the cutoff
    assert result.n_cycles == 8
    assert result.span == "2026-09-04 to 2026-09-11"
    # player 1 is eligible and called on 4, 8, 9 Sep (5-7 and 10-11 Sep sit inside a 72h window);
    # his 4 and 9 Sep calls rose. Player 2 is called all 8 nights and never rises.
    assert result.n_moves_predicted == 3 + 8
    assert result.hit_rate == pytest.approx(2 / 11)


def test_gate_has_no_out_of_sample_span_without_a_fit(tmp_path):
    distilled = tmp_path / "distilled"
    _cycle_shards(distilled, range(1, 5), {1: 2000}, {1: _rises_on(3)})
    unfitted = PriceThresholds(rise=0.5, fit_cutoff=None)

    result = run_price_model_for_gate(distilled, _events(tmp_path), thresholds=unfitted)

    assert result.n_cycles == 0
    assert result.hit_rate is None
