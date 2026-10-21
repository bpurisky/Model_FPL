"""Elite Ownership (`ownership.json`): the EO / top / captain arithmetic over
a hand-built picks file, the overall-ownership delta, chips, and no_data."""

from __future__ import annotations

from datetime import datetime, timezone

import polars as pl
import pytest

from web.export.contract import OwnershipFile
from web.export.ownership import build_ownership

FETCHED = datetime(2026, 9, 1, 12, 0, tzinfo=timezone.utc)


def _reference(tmp_path):
    ref = tmp_path / "reference"
    ref.mkdir(exist_ok=True)
    pl.DataFrame({
        "id": [1, 2, 3], "code": [1, 2, 3], "web_name": ["Alpha", "Beta", "Gamma"],
        "team": [1, 1, 1], "element_type": [3, 4, 2],
    }).write_parquet(ref / "players.parquet")
    pl.DataFrame({"id": [1], "name": ["Team A"], "short_name": ["TMA"], "strength": [None]}, schema={
        "id": pl.Int64, "name": pl.Utf8, "short_name": pl.Utf8, "strength": pl.Int64}).write_parquet(ref / "teams.parquet")
    return ref


def _distilled(tmp_path, gw=5, with_state=True):
    d = tmp_path / "distilled"
    (d / f"gw{gw}").mkdir(parents=True)
    # 4 managers. Alpha: all four (captained by 1, triple-captained by 1... see below), Beta: two, Gamma: bench for one.
    def pick(entry, element, mult, cap=False, chip=None):
        return {"rank": entry, "entry_id": entry, "gw": gw, "element": element, "position": 1, "multiplier": mult,
                "is_captain": cap, "is_vice_captain": False, "active_chip": chip, "fetched_at": FETCHED}
    rows = [
        pick(1, 1, 3, cap=True, chip="3xc"), pick(2, 1, 2, cap=True), pick(3, 1, 1), pick(4, 1, 1),
        pick(1, 2, 1), pick(2, 2, 1),
        pick(3, 3, 0, chip="bboost"),
    ]
    pl.DataFrame(rows, schema_overrides={"fetched_at": pl.Datetime("us", "UTC")}).write_parquet(d / f"gw{gw}" / "top_picks.parquet")
    if with_state:
        pl.DataFrame({
            "snapshot_ts": [datetime(2026, 9, 1, 6, 0, tzinfo=timezone.utc)] * 2,
            "element_id": [1, 3], "now_cost": [100, 45], "selected_by_percent": [40.0, 10.0],
            "transfers_in_event": [0, 0], "transfers_out_event": [0, 0], "form": [1.0, 1.0],
            "status": ["a", "a"], "chance_of_playing_next_round": [None, None],
            "news_added": [None, None], "ep_next": [1.0, 1.0],
        }, schema_overrides={"snapshot_ts": pl.Datetime("us", "UTC"), "news_added": pl.Datetime("us", "UTC"),
                             "chance_of_playing_next_round": pl.Int64}).write_parquet(d / f"gw{gw}" / "20260901T060000Z.parquet")
    return d


def test_percentages_and_eo(tmp_path):
    file = build_ownership(distilled_dir=_distilled(tmp_path), reference_dir=_reference(tmp_path))
    by = {p.element_id: p for p in file.players}

    assert file.status == "ok" and file.gw == 5 and file.n_managers == 4
    assert file.fetched_at == FETCHED
    a = by[1]
    assert a.top_pct == pytest.approx(100.0) and a.top_start_pct == pytest.approx(100.0)
    assert a.captain_pct == pytest.approx(50.0)
    assert a.eo_pct == pytest.approx((3 + 2 + 1 + 1) / 4 * 100)  # 175: a captain counts 2, triple captain 3
    assert a.overall_pct == 40.0 and a.delta_pct == pytest.approx(60.0) and a.now_cost == 100
    b = by[2]
    assert b.top_pct == pytest.approx(50.0) and b.eo_pct == pytest.approx(50.0)
    assert b.overall_pct is None and b.delta_pct is None  # no snapshot for him
    g = by[3]  # benched: held but not started, zero EO
    assert g.top_pct == pytest.approx(25.0) and g.top_start_pct == 0 and g.eo_pct == 0
    assert g.delta_pct == pytest.approx(25.0 - 10.0)
    assert [p.element_id for p in file.players] == [1, 2, 3]  # sorted by EO


def test_chips_count_managers_once(tmp_path):
    file = build_ownership(distilled_dir=_distilled(tmp_path), reference_dir=_reference(tmp_path))
    assert {c.chip: c.count for c in file.chips} == {"3xc": 1, "bboost": 1}


def test_top_picks_beside_delta_shards_does_not_break_state(tmp_path):
    # The shard readers must skip top_picks.parquet or this would raise on schema.
    file = build_ownership(distilled_dir=_distilled(tmp_path), reference_dir=_reference(tmp_path))
    assert file.players


def test_latest_gameweek_wins(tmp_path):
    d = _distilled(tmp_path, gw=5)
    (d / "gw7").mkdir()
    pl.read_parquet(d / "gw5" / "top_picks.parquet").with_columns(pl.lit(7).alias("gw")).write_parquet(d / "gw7" / "top_picks.parquet")
    assert build_ownership(distilled_dir=d, reference_dir=_reference(tmp_path)).gw == 7


def test_no_data_is_a_valid_empty_file(tmp_path):
    (tmp_path / "distilled" / "gw1").mkdir(parents=True)
    file = build_ownership(distilled_dir=tmp_path / "distilled", reference_dir=_reference(tmp_path))
    assert (file.status, file.gw, file.n_managers, file.fetched_at, file.players, file.chips) == ("no_data", None, 0, None, [], [])
    OwnershipFile.model_validate(file.model_dump())
    assert build_ownership(distilled_dir=tmp_path / "missing", reference_dir=_reference(tmp_path)).status == "no_data"
