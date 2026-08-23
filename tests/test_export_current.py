"""§5.11.1 over the current-season append.

The failure this file exists to prevent does not raise, does not warn, and
produces a panel that looks complete: an unenriched actuals store carries
no `n_fixtures`, `normalize` cumulative-sums it, `eligible_mask` compares
against it, and every 2026-27 player falls out of the reference population
with null z-scores. The season anyone is actually playing would render
with no normalized values at all.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import polars as pl
import pytest

from squad.live import TRAIN_SCHEMA
from web.export.current import (
    enrich,
    fixture_context,
    load_current_actuals,
    load_current_season,
    price_at_deadline,
)
from web.export.panel import build_panel

REFERENCE = Path("data/reference")


def _actuals(rows: list[dict], season: str | None = None) -> pl.DataFrame:
    """A frame in the actuals store's real schema, defaulting every stat
    the caller did not name. Built from `TRAIN_SCHEMA` rather than from
    literals so a column added there cannot silently arrive as null."""
    filled = []
    for row in rows:
        unknown = set(row) - set(TRAIN_SCHEMA) - {"season"}
        assert not unknown, f"unknown key(s): {unknown}"
        base = {
            name: (
                row.get(name)
                if name in row
                else (False if dtype == pl.Boolean else "" if dtype == pl.Utf8 else 0)
            )
            for name, dtype in TRAIN_SCHEMA.items()
        }
        filled.append(base)
    df = pl.DataFrame(filled, schema=TRAIN_SCHEMA)
    return df.with_columns(pl.lit(season).alias("season")) if season else df


def _squad(n: int, *, team: str, gw: int = 1, position: str = "MID") -> list[dict]:
    """A position group big enough to have a spread to normalize against."""
    return [
        {
            "gw": gw, "element_id": i, "position": position, "team": team,
            "minutes": 90, "expected_goals": 0.05 * i, "expected_assists": 0.02 * i,
            "goals_scored": i % 3, "total_points": 2 + (i % 5), "bps": 10 + i,
        }
        for i in range(1, n + 1)
    ]


# --- the silent failure ----------------------------------------------------


def test_an_unenriched_store_would_null_every_current_season_z_score():
    """The bug, stated as a passing test so the fix cannot be undone
    quietly. Feeding the raw actuals schema straight into the panel is
    green, produces rows, and leaves the whole season unnormalized."""
    raw = _actuals(_squad(24, team="Arsenal"), season="2026-27")
    assert "n_fixtures" not in raw.columns

    panel = build_panel(current=raw)
    current = panel.filter(pl.col("season") == "2026-27")

    assert current.height == 24, "the rows do arrive — nothing raises"
    assert current["xg_per90_z_pos"].null_count() == 24, "and every one is unnormalized"


def test_the_enriched_store_normalizes_like_any_other_season():
    """The same rows, enriched, behave as the archive does."""
    raw = _actuals(_squad(24, team="Arsenal"), season="2026-27")

    panel = build_panel(current=enrich(raw))
    current = panel.filter(pl.col("season") == "2026-27")

    assert current.height == 24
    assert current["xg_per90_z_pos"].null_count() == 0
    assert current["xg_per90_n_pos"].unique().to_list() == [24]
    # z-scores over a full population recover to mean 0, sd 1 (§5.11.2)
    z = current["xg_per90_z_pos"]
    assert z.mean() == pytest.approx(0.0, abs=1e-10)
    assert z.std() == pytest.approx(1.0, abs=1e-10)


# --- the derived context ---------------------------------------------------


def test_enrich_populates_n_fixtures_from_the_committed_fixture_list():
    enriched = enrich(_actuals(_squad(3, team="Arsenal"), season="2026-27"))

    assert enriched["n_fixtures"].null_count() == 0
    assert enriched["n_fixtures"].to_list() == [1, 1, 1]


def test_enrich_carries_the_opponent_and_venue():
    enriched = enrich(_actuals(_squad(1, team="Arsenal", gw=1), season="2026-27"))

    assert enriched["opponent_team"][0] == "Coventry City"
    assert enriched["was_home"][0] is True
    assert enriched["kickoff_time"][0] is not None


def test_enrich_recovers_the_price_that_stood_at_the_deadline():
    enriched = enrich(_actuals(_squad(3, team="Arsenal"), season="2026-27"))

    assert enriched["value"].null_count() == 0
    assert all(30 <= v <= 200 for v in enriched["value"].to_list())


def test_selected_is_null_rather_than_a_percentage_in_a_count_column():
    """The panel's `selected` is a squad count (0-9.5M in the archive);
    distilled records `selected_by_percent`. Converting needs
    bootstrap-static's `total_players`, which nothing collects. A
    percentage written into a count column would render confidently and
    wrongly on every chart that touched it."""
    enriched = enrich(_actuals(_squad(3, team="Arsenal"), season="2026-27"))

    assert enriched["selected"].null_count() == 3


def test_a_team_with_no_fixture_row_defaults_to_one_fixture(caplog):
    """A zero would divide the minutes floor away and make the player
    unconditionally eligible; 1 keeps the floor meaningful. Logged, not
    silent, because it means the reference data disagrees with the store."""
    raw = _actuals(_squad(2, team="Not A Real Club"), season="2026-27")

    enriched = enrich(raw)

    assert enriched["n_fixtures"].to_list() == [1, 1]
    assert "no fixture context" in caplog.text


# --- fixture context and prices --------------------------------------------


def test_fixture_context_counts_both_sides_of_every_fixture():
    context = fixture_context(REFERENCE)

    assert context.height == 760, "20 clubs x 38 gameweeks"
    assert context["n_fixtures"].min() >= 1
    assert context.filter(pl.col("gw") == 1).height == 20


def test_a_double_gameweek_counts_two_and_takes_the_first_fixture(tmp_path):
    """Matching what the archive already does — verified against 2023-24
    gw7, where Burnley's doubled rows carry one opponent and the earlier
    kickoff."""
    reference = tmp_path
    kickoff = datetime(2026, 9, 1, 14, 0, tzinfo=timezone.utc)
    pl.DataFrame(
        {
            "id": [1, 2],
            "event": [5, 5],
            "team_h": [1, 2],
            "team_a": [2, 1],
            "kickoff_time": [kickoff + timedelta(days=2), kickoff],
            # Alpha faces 2 at home in the first and 5 away in the second,
            # so the mean is a real average rather than a repeated value.
            "team_h_difficulty": [2, 3],
            "team_a_difficulty": [4, 5],
        },
        schema={
            "id": pl.Int64, "event": pl.Int64, "team_h": pl.Int64,
            "team_a": pl.Int64,
            "kickoff_time": pl.Datetime(time_unit="us", time_zone="UTC"),
            "team_h_difficulty": pl.Int64, "team_a_difficulty": pl.Int64,
        },
    ).write_parquet(reference / "fixtures.parquet")
    pl.DataFrame({"id": [1, 2], "name": ["Alpha", "Beta"]}).write_parquet(
        reference / "teams.parquet"
    )

    context = fixture_context(reference).filter(pl.col("team") == "Alpha")

    assert context["n_fixtures"].to_list() == [2]
    assert context["kickoff_time"][0] == kickoff, "the earlier fixture wins"
    assert context["was_home"][0] is False, "and its venue comes with it"
    # FPL publishes difficulty per side; a double averages the two, matching
    # backtest/backfill.py's own aggregation.
    assert context["opponent_difficulty"][0] == pytest.approx(3.5)


def test_price_uses_the_last_snapshot_at_or_before_the_deadline(tmp_path):
    """Distilled is delta-only, so the price in force at a deadline is the
    most recent snapshot before it, not a row bearing that timestamp."""
    reference, distilled = tmp_path / "ref", tmp_path / "dist"
    reference.mkdir()
    distilled.mkdir()
    deadline = datetime(2026, 8, 21, 17, 30, tzinfo=timezone.utc)
    pl.DataFrame(
        {"id": [1], "deadline_time": [deadline]},
        schema={"id": pl.Int64, "deadline_time": pl.Datetime(time_unit="us", time_zone="UTC")},
    ).write_parquet(reference / "events.parquet")
    pl.DataFrame(
        {
            "snapshot_ts": [
                deadline - timedelta(days=2),
                deadline - timedelta(hours=1),
                deadline + timedelta(hours=1),  # after: must not win
            ],
            "element_id": [7, 7, 7],
            "now_cost": [50, 51, 99],
        },
        schema={
            "snapshot_ts": pl.Datetime(time_unit="us", time_zone="UTC"),
            "element_id": pl.Int64, "now_cost": pl.Int64,
        },
    ).write_parquet(distilled / "gw1.parquet")

    prices = price_at_deadline([1], reference, distilled)

    assert prices["value"].to_list() == [51]


def test_price_is_null_when_no_snapshot_predates_the_deadline(tmp_path):
    """Rather than the earliest price known, which would be a guess about
    a past nobody recorded."""
    reference, distilled = tmp_path / "ref", tmp_path / "dist"
    reference.mkdir()
    distilled.mkdir()
    deadline = datetime(2026, 8, 21, 17, 30, tzinfo=timezone.utc)
    pl.DataFrame(
        {"id": [1], "deadline_time": [deadline]},
        schema={"id": pl.Int64, "deadline_time": pl.Datetime(time_unit="us", time_zone="UTC")},
    ).write_parquet(reference / "events.parquet")
    pl.DataFrame(
        {
            "snapshot_ts": [deadline + timedelta(hours=1)],
            "element_id": [7], "now_cost": [50],
        },
        schema={
            "snapshot_ts": pl.Datetime(time_unit="us", time_zone="UTC"),
            "element_id": pl.Int64, "now_cost": pl.Int64,
        },
    ).write_parquet(distilled / "gw1.parquet")

    assert price_at_deadline([1], reference, distilled).height == 0


# --- loading ----------------------------------------------------------------


def test_no_store_is_none_rather_than_an_empty_frame(tmp_path):
    """"No gameweek recorded yet" and "a gameweek was recorded and had no
    rows" are different facts; only the first is normal right now."""
    assert load_current_actuals(tmp_path / "absent") is None
    (tmp_path / "empty").mkdir()
    assert load_current_actuals(tmp_path / "empty") is None
    assert load_current_season(tmp_path / "absent") is None


def test_the_season_label_comes_from_the_filename(tmp_path):
    """The store is one file per season and names itself; the rows inside
    do not repeat it."""
    _actuals(_squad(2, team="Arsenal")).write_parquet(tmp_path / "2026-27.parquet")

    loaded = load_current_actuals(tmp_path)

    assert loaded["season"].unique().to_list() == ["2026-27"]


def test_the_store_carries_every_per90_source_the_panel_derives_from():
    """The append is only worth wiring if the metrics survive it. This is
    the half that already worked and must keep working."""
    from web.export.columns import PER90_SOURCES

    assert not set(PER90_SOURCES.values()) - set(TRAIN_SCHEMA)
