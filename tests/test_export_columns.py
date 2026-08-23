"""§5.11.1 registry completeness, plus the §5.15 decisions the registry
encodes.

The spec singles out the completeness check as "the most likely silent
breakage in the whole system": a metric that reaches the panel without a
registry entry renders with no definition, no format and no
`higher_is_better`, and nothing anywhere raises.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from web.export.columns import (
    DEFENSIVE_ACTIONS_FROM,
    ICT_UNTIL,
    MATRIX_METRICS,
    PER90_SOURCES,
    REGISTRY,
    by_key,
    companion_keys,
    normalizable_keys,
    per90_expr,
)

PANEL = "data/historical/2025-26.parquet"


def _panel() -> pl.DataFrame:
    return pl.read_parquet(PANEL)


def test_registry_keys_are_unique():
    keys = [c.key for c in REGISTRY]
    assert len(keys) == len(set(keys))


def test_every_per90_metric_has_a_registry_entry():
    registry = by_key()
    for key in PER90_SOURCES:
        assert key in registry, f"{key} derives a column but has no registry entry"


def test_every_per90_source_exists_in_the_panel():
    """The half of the completeness check that catches a renamed upstream
    column. PER90_SOURCES names raw panel columns; if one drifts, the
    metric silently becomes unbuildable."""
    panel_columns = set(_panel().columns)
    for key, source in PER90_SOURCES.items():
        assert source in panel_columns, f"{key} derives from missing panel column {source!r}"


def test_matrix_metrics_are_all_registered_and_normalizable():
    registry = by_key()
    for key in MATRIX_METRICS:
        assert key in registry, f"{key} is in the matrix set but not the registry"
        assert registry[key].normalizable, f"{key} is compared across players and must be normalizable"


def test_matrix_set_answers_q2_with_sixteen_metrics():
    """§5.15 Q2 asks for ~15-20: too many makes the hero unreadable, too
    few makes it trivial."""
    assert len(MATRIX_METRICS) == 16


def test_normalizable_columns_declare_their_companion_key():
    for column in REGISTRY:
        if column.normalizable:
            assert column.normalized_key == companion_keys(column.key)[0]
        else:
            assert column.normalized_key is None


def test_normalizable_columns_state_a_direction():
    """`higher_is_better` orients the diverging scale. A normalizable
    column without one would render xGC coloured like xG, which tells the
    reader the opposite of the truth."""
    for column in REGISTRY:
        if column.normalizable:
            assert column.higher_is_better is not None, f"{column.key} has no direction"


def test_conceding_metrics_are_lower_is_better():
    registry = by_key()
    assert registry["xgc_per90"].higher_is_better is False
    assert registry["goals_conceded_per90"].higher_is_better is False
    assert registry["xg_per90"].higher_is_better is True


def test_every_column_carries_a_definition():
    """§5.3.5: the registry drives every tooltip definition in the app."""
    for column in REGISTRY:
        assert column.definition.strip(), f"{column.key} has no definition"


def test_position_relevance_covers_all_four_positions():
    for column in REGISTRY:
        assert set(column.position_relevance) == {"GK", "DEF", "MID", "FWD"}, column.key


def test_saves_is_relevant_to_goalkeepers_only():
    """The clearest case of why `none` must dim rather than hide: saves
    per 90 is a real column that simply does not matter for a forward."""
    relevance = by_key()["saves_per90"].position_relevance
    assert relevance["GK"] == "primary"
    assert relevance["DEF"] == relevance["MID"] == relevance["FWD"] == "none"


# --- the §5.15 decisions the registry encodes ---------------------------


def test_shots_in_box_is_absent():
    """§5.4.6 gave it an 0.15 FWD weight; it exists in no source this
    project has, and threat_per90 is not a substitute (see module
    docstring in columns.py)."""
    assert "shots_in_box_per90" not in by_key()


def test_ict_is_registered_but_ends_in_2025_26():
    """FPL published influence/creativity/threat through 2025-26 and
    reports literal 0.0 for all 604 elements from 2026-27. Registering
    them with an end season keeps three seasons of real data usable while
    stopping manufactured zeros rendering as measurement (§5.3.3)."""
    registry = by_key()
    for key in ("influence_per90", "creativity_per90", "threat_per90"):
        assert registry[key].available_to_season == ICT_UNTIL
        assert registry[key].applies_to_season("2025-26") is True
        assert registry[key].applies_to_season("2026-27") is False


def test_ict_is_excluded_from_the_matrix_set():
    for key in ("influence_per90", "creativity_per90", "threat_per90"):
        assert key not in MATRIX_METRICS


@pytest.mark.parametrize(
    "key", ["defensive_contribution_per90", "tackles_per90", "recoveries_per90", "cbi_per90"]
)
def test_defensive_action_metrics_start_in_2025_26(key):
    """Verified against the committed panel: these four are entirely null
    for 2023-24 and 2024-25 and fully populated from 2025-26. §5.3.3
    requires those rows render as not-applicable, never as zero."""
    column = by_key()[key]
    assert column.available_from_season == DEFENSIVE_ACTIONS_FROM
    assert column.applies_to_season("2024-25") is False
    assert column.applies_to_season("2025-26") is True


def test_declared_availability_matches_the_committed_panel():
    """The registry's claim about when a column starts, checked against
    the data rather than against the FPL rule history. This is the test
    that fails if a future backfill populates an era it should not."""
    registry = by_key()
    for season, applies in (("2024-25", False), ("2025-26", True)):
        df = pl.read_parquet(f"data/historical/{season}.parquet")
        for key in ("defensive_contribution_per90", "tackles_per90"):
            source = PER90_SOURCES[key]
            all_null = df[source].null_count() == df.height
            assert registry[key].applies_to_season(season) is applies
            assert all_null is not applies, f"{source} in {season}: panel and registry disagree"


# --- per-90 derivation ---------------------------------------------------


def test_per90_is_null_for_a_player_who_did_not_appear():
    """Not zero. A player who did not play has no rate, and averaging a
    manufactured 0.0 into the positional mean would drag it toward
    whoever sat on the bench."""
    df = pl.DataFrame({"minutes": [90, 0, 45], "expected_goals": [0.5, 0.0, 0.2]})

    out = df.with_columns(per90_expr("expected_goals", "xg_per90"))

    assert out["xg_per90"][0] == pytest.approx(0.5)
    assert out["xg_per90"][1] is None
    assert out["xg_per90"][2] == pytest.approx(0.4)


def test_normalizable_keys_excludes_identity_columns():
    keys = set(normalizable_keys())
    for identity in ("element_id", "team", "position", "season", "gw", "minutes"):
        assert identity not in keys


# --- the committed artifact ---------------------------------------------


def test_committed_columns_json_matches_the_registry():
    """§5.3.4 commits columns.json so a fresh clone renders without a
    pipeline run. That only holds if the committed file is current --
    otherwise the app ships one registry and the code holds another, and
    nothing raises. Compares the entries only: the header carries a
    timestamp and a git sha that move every run."""
    import json

    path = Path("data/web/v1/columns.json")
    if not path.exists():  # pragma: no cover
        pytest.skip("columns.json not generated yet -- run `python -m web.export columns`")

    committed = json.loads(path.read_text(encoding="utf-8"))["columns"]
    current = [json.loads(c.model_dump_json()) for c in REGISTRY]

    assert committed == current, "data/web/v1/columns.json is stale -- regenerate it"
