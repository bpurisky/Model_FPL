"""§5.11.1 over `panel.parquet`: schema conformance, null preservation,
and the availability masking §5.3.3 depends on."""

from __future__ import annotations

import polars as pl
import pytest

from web.export.columns import MATRIX_METRICS, PER90_SOURCES, companion_keys
from web.export.panel import (
    CONTEXT_COLUMNS,
    apply_availability,
    available_seasons,
    build_panel,
    derive_metrics,
)


def _panel() -> pl.DataFrame:
    return build_panel(seasons=["2025-26"])


# --- availability masking (§5.3.3) --------------------------------------


def test_a_discontinued_column_is_nulled_even_though_fpl_sends_zero():
    """The dangerous case, and the reason `available_to_season` exists.
    FPL publishes influence/creativity/threat as literal 0.0 for all 604
    elements from 2026-27, so nothing upstream is null and nothing
    raises. A well-formed column of manufactured zeros would otherwise
    flow into the panel, into the positional means, and onto a heat map
    as though it were measurement."""
    df = pl.DataFrame({"season": ["2026-27", "2025-26"], "creativity_per90": [0.0, 12.5]})

    out = apply_availability(df)

    assert out["creativity_per90"][0] is None
    assert out["creativity_per90"][1] == pytest.approx(12.5)


def test_a_column_that_starts_late_is_nulled_before_its_first_season():
    """Belt and braces over backtest/backfill.py's own null restoration:
    if that ever regresses to emitting zeros, this is the layer that
    stops them reaching a chart."""
    df = pl.DataFrame({"season": ["2024-25", "2025-26"], "tackles_per90": [0.0, 3.0]})

    out = apply_availability(df)

    assert out["tackles_per90"][0] is None
    assert out["tackles_per90"][1] == pytest.approx(3.0)


def test_an_always_available_column_is_untouched():
    df = pl.DataFrame({"season": ["2023-24", "2026-27"], "xg_per90": [0.4, 0.6]})

    out = apply_availability(df)

    assert out["xg_per90"].to_list() == pytest.approx([0.4, 0.6])


def test_masking_a_zero_is_not_the_same_as_leaving_it():
    """§5.3.3 in one assertion: zero is a claim, absent is not the same
    claim, and the export must not collapse them."""
    df = pl.DataFrame({"season": ["2026-27"], "threat_per90": [0.0]})

    assert apply_availability(df)["threat_per90"][0] is not 0.0  # noqa: F632 - identity is the point
    assert apply_availability(df)["threat_per90"][0] is None


# --- derivation ----------------------------------------------------------


def test_derive_metrics_skips_absent_source_columns():
    """A current-season frame will not carry everything the archive does.
    A missing source is a smaller column set, not a crash."""
    df = pl.DataFrame({"minutes": [90], "expected_goals": [0.5]})

    out = derive_metrics(df)

    assert "xg_per90" in out.columns
    assert "tackles_per90" not in out.columns


def test_derived_rates_are_null_for_a_non_appearance():
    df = pl.DataFrame({"minutes": [0], "expected_goals": [0.0]})

    assert derive_metrics(df)["xg_per90"][0] is None


# --- the built panel -----------------------------------------------------


def test_panel_carries_context_and_every_matrix_metric_with_companions():
    df = _panel()

    for column in CONTEXT_COLUMNS:
        assert column in df.columns, f"missing context column {column}"
    for key in MATRIX_METRICS:
        if key not in PER90_SOURCES:
            continue  # model-derived, added when the current-season store exists
        assert key in df.columns
        for companion in companion_keys(key):
            assert companion in df.columns, f"{key} has no {companion}"


def test_panel_is_one_row_per_player_per_gameweek():
    df = _panel()

    assert df.select(["season", "gw", "element_id"]).is_duplicated().sum() == 0


def test_panel_grain_columns_are_never_null():
    """Everything else may legitimately be null; the grain may not."""
    df = _panel()

    for column in ("season", "gw", "element_id", "position"):
        assert df[column].null_count() == 0, f"{column} has nulls"


def test_panel_defensive_actions_are_null_in_the_seasons_before_the_rule():
    df = build_panel(seasons=["2024-25", "2025-26"])

    early = df.filter(pl.col("season") == "2024-25")
    late = df.filter(pl.col("season") == "2025-26")

    assert early["tackles_per90"].null_count() == early.height
    assert late["tackles_per90"].null_count() < late.height


def test_panel_appends_a_current_season_frame():
    """The 2026-27 path, before data/current_season/ exists.

    The current-season store carries fewer columns than the historical
    archive -- papertrade/actuals.py's schema is a strict subset, because
    /event/{gw}/live/ does not serve fixture context. A frame narrower
    than the panel must widen with nulls rather than raise."""
    current = pl.DataFrame(
        {
            "season": ["2026-27"] * 2,
            "gw": [1, 1],
            "element_id": [1, 2],
            "name": ["A", "B"],
            "team": ["X", "Y"],
            "position": ["MID", "MID"],
            "minutes": [90, 90],
            "n_fixtures": [1, 1],
            "total_points": [5, 2],
            "expected_goals": [0.5, 0.1],
        }
    )

    out = build_panel(seasons=["2025-26"], current=current)

    assert out.filter(pl.col("season") == "2026-27").height == 2
    assert out.filter(pl.col("season") == "2026-27")["xg_per90"].null_count() == 0


def test_available_seasons_finds_the_committed_archive():
    seasons = available_seasons()

    assert "2024-25" in seasons
    assert seasons == sorted(seasons)
