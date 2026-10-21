"""Price Watch (`prices.json`) over synthetic snapshots: the progress math,
72h eligibility, the stale-snapshot path, and the embedded track record."""

from __future__ import annotations

from datetime import datetime, timezone

import polars as pl
import pytest

from analytics.price_model import PriceModelEvaluation
from tests.test_price_model import _at, _cycle_shards, _events, _rises_on
from web.export.contract import PricesFile
from web.export.prices import build_prices

THRESHOLD = 1000.0
_NO_GATE_CALLS = PriceModelEvaluation(n=0, n_moves_predicted=0, hit_rate=None, ci_low=None, ci_high=None, n_cycles=0)


def _reference(tmp_path, eids):
    ref = tmp_path / "reference"
    ref.mkdir()
    pl.DataFrame({
        "id": eids, "code": eids, "web_name": [f"P{e}" for e in eids], "team": [1] * len(eids), "element_type": [3] * len(eids),
    }).write_parquet(ref / "players.parquet")
    pl.DataFrame({"id": [1], "name": ["Team A"], "short_name": ["TMA"], "strength": [None]}, schema={
        "id": pl.Int64, "name": pl.Utf8, "short_name": pl.Utf8, "strength": pl.Int64}).write_parquet(ref / "teams.parquet")
    return ref


def _config(tmp_path, cutoff="2026-09-03", extra=""):
    path = tmp_path / "price_model.yaml"
    path.write_text(
        f"rise_threshold: {THRESHOLD}\nfit_cutoff: {cutoff}\nfit_span: 2026-08-22 to 2026-09-02\n"
        f"fit_cycles: 5\nfit_rise_f1: 0.4\nfit_rise_n_called: 58\nfit_in_sample_hit_rate: 0.431\n{extra}",
        encoding="utf-8",
    )
    return path


def _build(tmp_path, inflow, cost_on_day, now=_at(8, 19), days=range(1, 9), price_eval=_NO_GATE_CALLS):
    distilled = tmp_path / "distilled"
    _cycle_shards(distilled, days, inflow, cost_on_day)
    return build_prices(
        distilled_dir=distilled, events_path=_events(tmp_path), reference_dir=_reference(tmp_path, list(inflow)),
        config_path=_config(tmp_path), now=now, price_eval=price_eval,
    )


def test_progress_is_net_transfers_over_the_threshold_and_can_be_negative_or_above_one(tmp_path):
    file = _build(tmp_path, {1: 2000, 2: 500, 3: -1000}, {})
    by_id = {p.element_id: p for p in file.players}

    assert file.status == "ok" and file.threshold == THRESHOLD
    assert by_id[1].net_transfers_24h == 2000 and by_id[1].progress == pytest.approx(2.0)
    assert by_id[2].progress == pytest.approx(0.5)
    assert by_id[3].progress == pytest.approx(-1.0)
    assert [p.element_id for p in file.players] == [1, 2, 3]  # sorted by progress
    assert (by_id[1].call, by_id[2].call, by_id[3].call) == ("rise", "stable", "stable")  # never a fall


def test_a_player_who_moved_within_72h_is_ineligible_and_never_called(tmp_path):
    # player 1 steps up at the 7 Sep snapshot (inside the 72h window), player 2 at 3 Sep (outside it)
    file = _build(tmp_path, {1: 2000, 2: 2000, 3: 2000}, {1: _rises_on(7), 2: _rises_on(3)})
    by_id = {p.element_id: p for p in file.players}

    assert (by_id[1].eligible, by_id[1].ineligible_reason, by_id[1].call) == (False, "moved_within_72h", "stable")
    assert by_id[1].progress == pytest.approx(2.0)  # the progress is still reported
    assert (by_id[2].eligible, by_id[2].ineligible_reason, by_id[2].call) == (True, None, "rise")
    assert by_id[3].eligible and by_id[3].call == "rise"
    assert by_id[2].now_cost == 51 and by_id[2].position == "MID" and by_id[2].team == "Team A"


def test_a_player_without_a_snapshot_24h_ago_has_no_progress(tmp_path):
    distilled = tmp_path / "distilled"
    _cycle_shards(distilled, range(1, 9), {1: 2000}, {})
    _cycle_shards(distilled, range(8, 9), {2: 2000}, {})  # player 2 first appears in the latest snapshot only
    file = build_prices(
        distilled_dir=distilled, events_path=_events(tmp_path), reference_dir=_reference(tmp_path, [1, 2]),
        config_path=_config(tmp_path), now=_at(8, 19), price_eval=_NO_GATE_CALLS,
    )
    new = next(p for p in file.players if p.element_id == 2)

    assert (new.net_transfers_24h, new.progress) == (None, None)
    assert (new.eligible, new.ineligible_reason, new.call) == (False, "insufficient_history", "stable")
    assert file.players[-1].element_id == 2  # unknowns sort last


def test_the_cycle_is_open_before_the_boundary_and_locked_after_it(tmp_path):
    file = _build(tmp_path, {1: 2000}, {})
    assert file.locked is False
    assert file.snapshot_ts == _at(8, 18)
    assert (file.boundary_ts, file.next_boundary_ts) == (_at(8, 19), _at(9, 19))


def test_a_stale_snapshot_is_reported_not_predicted(tmp_path):
    file = _build(tmp_path, {1: 2000}, {}, now=_at(9, 6))  # latest snapshot is 12h old

    assert file.status == "stale" and file.status_detail
    assert file.players == [] and file.snapshot_ts is None and file.locked is None
    assert file.track_record.status == "insufficient data"  # the record is still reported


def test_track_record_reports_the_gate_and_labels_the_fit_in_sample(tmp_path):
    gate = PriceModelEvaluation(n=900, n_moves_predicted=25, hit_rate=0.6, ci_low=0.41, ci_high=0.79, n_cycles=9, span="2026-10-21 to 2026-10-29")
    record = _build(tmp_path, {1: 2000}, {}, price_eval=gate).track_record

    assert (record.status, record.n_calls, record.n_cycles, record.min_calls) == ("PASS", 25, 9, 20)
    assert (record.hit_rate, record.ci_low, record.ci_high, record.span) == (0.6, 0.41, 0.79, "2026-10-21 to 2026-10-29")
    assert record.in_sample is not None
    assert (record.in_sample.fit_cutoff, record.in_sample.fit_cycles, record.in_sample.hit_rate) == ("2026-09-03", 5, 0.431)


def test_too_few_calls_is_insufficient_data_not_a_rate(tmp_path):
    gate = PriceModelEvaluation(n=100, n_moves_predicted=3, hit_rate=1.0, ci_low=1.0, ci_high=1.0, n_cycles=2, span="x")
    assert _build(tmp_path, {1: 2000}, {}, price_eval=gate).track_record.status == "insufficient data"


def test_recent_cycles_list_calls_and_hits_flagged_out_of_sample(tmp_path):
    # player 1 is called every night and rises on 5 Sep; cutoff 3 Sep
    record = _build(tmp_path, {1: 2000}, {1: _rises_on(5)}).track_record
    by_date = {c.date: c for c in record.recent_cycles}

    assert by_date["2026-09-02"].out_of_sample is False and by_date["2026-09-03"].out_of_sample is True
    assert (by_date["2026-09-04"].calls, by_date["2026-09-04"].hits) == (1, 1)  # evaluates across the 5 Sep step
    assert by_date["2026-09-03"].hits == 0


def test_the_file_round_trips_through_the_contract(tmp_path):
    file = _build(tmp_path, {1: 2000, 2: 10}, {})
    assert PricesFile.model_validate_json(file.model_dump_json()) == file
