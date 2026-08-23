"""§5.11.1 over `timeseries.parquet`: that the delta-only shape survives
the export, and that the two projection series stay distinguishable."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import polars as pl
import pytest

from web.export.timeseries import (
    DISTILLED_COLUMNS,
    build_timeseries,
    load_distilled,
    model_projections,
    player_identity,
    shard_gameweek,
)

REFERENCE = Path("data/reference")
BASE = datetime(2026, 8, 21, 12, 0, tzinfo=timezone.utc)


def _shard(rows: list[dict]) -> pl.DataFrame:
    return pl.DataFrame(
        {c: [r.get(c) for r in rows] for c in DISTILLED_COLUMNS},
        schema={
            "snapshot_ts": pl.Datetime(time_unit="us", time_zone="UTC"),
            "element_id": pl.Int64, "now_cost": pl.Int64,
            "selected_by_percent": pl.Float64, "transfers_in_event": pl.Int64,
            "transfers_out_event": pl.Int64, "form": pl.Float64, "status": pl.Utf8,
            "chance_of_playing_next_round": pl.Int64, "news_added": pl.Utf8,
            "ep_next": pl.Float64,
        },
    )


def _row(element_id: int, hours: int, cost: int) -> dict:
    return {
        "snapshot_ts": BASE + timedelta(hours=hours), "element_id": element_id,
        "now_cost": cost, "selected_by_percent": 12.5, "transfers_in_event": 0,
        "transfers_out_event": 0, "form": 3.0, "status": "a",
        "chance_of_playing_next_round": 100, "news_added": None, "ep_next": 4.2,
    }


# --- gameweek attribution ---------------------------------------------------


def test_the_gameweek_comes_from_the_shard_directory():
    """The collector partitions by gameweek, so it has already answered
    which one a snapshot belongs to. Re-deriving it from deadlines would
    be a second opinion that can disagree with the first."""
    assert shard_gameweek(Path("data/distilled/gw7/20260821T154505Z.parquet")) == 7
    assert shard_gameweek(Path("data/distilled/gw13/x.parquet")) == 13
    assert shard_gameweek(Path("data/distilled/loose.parquet")) is None


def test_a_shard_outside_a_gameweek_directory_is_skipped_loudly(tmp_path, caplog):
    (tmp_path / "gw1").mkdir()
    _shard([_row(1, 0, 50)]).write_parquet(tmp_path / "gw1" / "a.parquet")
    _shard([_row(2, 0, 50)]).write_parquet(tmp_path / "stray.parquet")

    loaded = load_distilled(tmp_path)

    assert loaded["element_id"].to_list() == [1]
    assert "outside a gw directory" in caplog.text


# --- delta-only is the point ------------------------------------------------


def test_the_export_preserves_one_row_per_recorded_change(tmp_path):
    """§2.3's delta-only tier is a step function sampled irregularly, not
    a regular series. Forward-filling here would manufacture observations
    nobody made, and a player whose price never moved having two rows in a
    month is a fact about his price rather than a gap."""
    (tmp_path / "gw1").mkdir()
    _shard([_row(1, 0, 50), _row(1, 72, 51)]).write_parquet(tmp_path / "gw1" / "a.parquet")

    loaded = load_distilled(tmp_path)

    assert loaded.height == 2, "no interpolation between the two observations"
    assert loaded["now_cost"].to_list() == [50, 51]


def test_rows_are_ordered_by_player_then_time():
    """A series read in file order should already be a series."""
    df = build_timeseries()
    head = df.head(500).select("element_id", "snapshot_ts")

    assert head.equals(head.sort(["element_id", "snapshot_ts"]))


# --- the two projections are different claims -------------------------------


def test_model_projection_is_null_when_no_freeze_exists(tmp_path):
    """Null means "the model had not spoken yet", which is not a low
    projection — the UI has to be able to tell those apart."""
    assert model_projections(tmp_path / "absent").height == 0

    df = build_timeseries(freezes_dir=tmp_path / "absent")

    assert df["model_projection"].null_count() == df.height
    assert df["ep_next"].null_count() < df.height, "FPL's own projection is present"


def test_only_the_freezes_own_gameweek_is_used(tmp_path):
    """A freeze projects a three-gameweek horizon. The later two are
    forecasts made before the intervening football happened; putting them
    on the same series as a deadline-day projection would present a
    three-week-old guess as current."""
    tmp_path.mkdir(exist_ok=True)
    (tmp_path / "gw4.json").write_text(
        json.dumps(
            {
                "gameweek": 4,
                "projections": {
                    "4": {"11": 5.5, "12": 2.0},
                    "5": {"11": 9.9},  # horizon — must not appear
                    "6": {"11": 9.9},
                },
            }
        ),
        encoding="utf-8",
    )

    projections = model_projections(tmp_path)

    assert projections["gw"].unique().to_list() == [4]
    assert sorted(projections["element_id"].to_list()) == [11, 12]
    assert 9.9 not in projections["model_projection"].to_list()


def test_a_freeze_projection_lands_on_its_own_gameweeks_snapshots(tmp_path):
    distilled, freezes = tmp_path / "d", tmp_path / "f"
    (distilled / "gw1").mkdir(parents=True)
    freezes.mkdir()
    _shard([_row(11, 0, 50), _row(12, 0, 45)]).write_parquet(distilled / "gw1" / "a.parquet")
    (freezes / "gw1.json").write_text(
        json.dumps({"gameweek": 1, "projections": {"1": {"11": 5.5}}}), encoding="utf-8"
    )

    df = build_timeseries(distilled_dir=distilled, reference_dir=REFERENCE, freezes_dir=freezes)
    by_element = dict(zip(df["element_id"].to_list(), df["model_projection"].to_list()))

    assert by_element[11] == pytest.approx(5.5)
    assert by_element[12] is None, "a player the freeze did not project stays null"


# --- identity and shape -----------------------------------------------------


def test_identity_is_joined_so_a_series_can_be_labelled_without_the_panel():
    """`panel.parquet` is not committed (§5.3.4), so the Explorer cannot
    rely on it to turn an element id into a name."""
    identity = player_identity(REFERENCE)

    assert identity.height > 500
    assert set(identity.columns) == {"element_id", "name", "team", "position"}
    assert set(identity["position"].unique().to_list()) <= {"GK", "DEF", "MID", "FWD"}
    assert identity["name"].null_count() == 0


def test_an_absent_collector_history_fails_loudly(tmp_path):
    """§0.1: data not collected is permanently lost, so an empty distilled
    tier is a broken pipeline rather than an empty chart."""
    with pytest.raises(ValueError, match="cannot be reconstructed"):
        build_timeseries(distilled_dir=tmp_path / "absent")


def test_the_real_export_covers_every_collected_snapshot():
    df = build_timeseries()

    assert df.height > 0
    assert df["snapshot_ts"].n_unique() >= 50
    assert df["gw"].null_count() == 0
    assert df["name"].null_count() == 0
    for column in ("now_cost", "selected_by_percent", "ep_next"):
        assert column in df.columns
