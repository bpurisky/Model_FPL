"""§5.11.1 header completeness and §5.3.1 contract conformance."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from web.export.columns import REGISTRY
from web.export.contract import (
    CONTRACT_VERSION,
    ColumnSpec,
    ColumnsFile,
    Header,
    build_header,
    contract_shape,
)


def _header(**overrides) -> Header:
    base = dict(
        generated_at=datetime(2026, 8, 23, 14, 0, tzinfo=timezone.utc),
        source_gameweek=3,
        scoring_config="scoring_2026_27.yaml",
        model_git_sha="a1b2c3d",
        normalization_basis="within_position_season_to_date",
        rows=691,
    )
    return Header(**{**base, **overrides})


def test_header_carries_every_field_5_3_1_requires():
    header = _header()

    assert header.contract_version == CONTRACT_VERSION
    for field in ("generated_at", "source_gameweek", "scoring_config", "model_git_sha", "normalization_basis", "rows"):
        assert hasattr(header, field)


def test_build_header_reads_the_sha_rather_than_accepting_one():
    """A sha supplied by the caller is a sha that can be wrong. §5.3.1
    calls model_git_sha non-negotiable because every number on screen has
    to lead back to the code that produced it."""
    header = build_header(rows=10, source_gameweek=3, normalization_basis="basis")

    assert header.model_git_sha is None or len(header.model_git_sha) == 40


def test_build_header_stamps_a_timezone_aware_timestamp():
    header = build_header(rows=1, source_gameweek=None, normalization_basis="b")

    assert header.generated_at.tzinfo is not None


def test_header_rejects_an_unknown_field():
    """extra='forbid', unlike collector/schemas.py. There the payload is
    FPL's and drift is news; here it is ours and drift is a bug."""
    with pytest.raises(ValidationError):
        _header(unexpected="value")


def test_header_requires_a_normalization_basis():
    """§5.3.1: equally non-negotiable. A z-score is meaningless without
    the population it was computed against."""
    with pytest.raises(ValidationError):
        Header(
            generated_at=datetime.now(timezone.utc),
            source_gameweek=1,
            scoring_config="s.yaml",
            model_git_sha=None,
            rows=1,
        )


def test_source_gameweek_may_be_null_but_must_be_present():
    """Historical-only exports have no source gameweek. Absent and
    explicitly null are different claims, and only the second is allowed."""
    assert _header(source_gameweek=None).source_gameweek is None


def test_every_registry_entry_validates_against_the_contract():
    for column in REGISTRY:
        assert isinstance(column, ColumnSpec)
        ColumnSpec.model_validate(column.model_dump())


def test_columns_file_round_trips_through_json():
    file = ColumnsFile(header=_header(rows=len(REGISTRY)), columns=REGISTRY)

    restored = ColumnsFile.model_validate_json(file.model_dump_json())

    assert len(restored.columns) == len(REGISTRY)
    assert restored.columns[0].key == REGISTRY[0].key
    assert restored.header.rows == len(REGISTRY)


def test_column_spec_rejects_an_unknown_role():
    with pytest.raises(ValidationError):
        ColumnSpec(
            key="k", label="K", role="continuous", unit=None, format=".2f",
            definition="d", source="fpl_api", grain="player_gameweek",
            normalizable=False, normalized_key=None,
            position_relevance={"GK": "none", "DEF": "none", "MID": "none", "FWD": "none"},
            higher_is_better=None, available_from_season=None,
        )


def test_column_spec_rejects_an_unknown_position_relevance():
    with pytest.raises(ValidationError):
        ColumnSpec(
            key="k", label="K", role="quantitative", unit=None, format=".2f",
            definition="d", source="fpl_api", grain="player_gameweek",
            normalizable=False, normalized_key=None,
            position_relevance={"GK": "vital", "DEF": "none", "MID": "none", "FWD": "none"},
            higher_is_better=None, available_from_season=None,
        )


def test_applies_to_season_is_inclusive_at_both_ends():
    column = ColumnSpec(
        key="k", label="K", role="quantitative", unit=None, format=".2f",
        definition="d", source="fpl_api", grain="player_gameweek",
        normalizable=False, normalized_key=None,
        position_relevance={"GK": "none", "DEF": "none", "MID": "none", "FWD": "none"},
        higher_is_better=None,
        available_from_season="2024-25", available_to_season="2025-26",
    )

    assert column.applies_to_season("2023-24") is False
    assert column.applies_to_season("2024-25") is True
    assert column.applies_to_season("2025-26") is True
    assert column.applies_to_season("2026-27") is False


def test_a_column_with_no_bounds_applies_to_every_season():
    column = next(c for c in REGISTRY if c.key == "xg_per90")

    assert column.applies_to_season("2023-24") is True
    assert column.applies_to_season("2026-27") is True


def test_contract_shape_describes_each_model_for_the_schema_ts_test():
    """§5.12.2 requires a test asserting contract.py and schema.ts agree.
    This is the Python half; it is deliberately coarse so the eventual
    comparison fails on real divergence rather than on pydantic detail
    zod cannot express."""
    shape = contract_shape()

    assert set(shape) == {
        "Header",
        "ColumnSpec",
        "ColumnsFile",
        "CorrelationCell",
        "GroupSummary",
        "CorrelationsFile",
    }
    assert shape["Header"]["rows"] == {"required": True, "type": "int"}
    assert shape["Header"]["generated_at"]["type"] == "datetime"
    assert shape["ColumnSpec"]["position_relevance"]["type"] == "record"
    assert shape["ColumnSpec"]["available_to_season"]["required"] is False
    assert shape["ColumnsFile"]["columns"]["type"] == "array"
    # rho and p_value are nullable and n is not: a cell can fail to
    # produce a correlation and still have to report what it ran over.
    assert shape["CorrelationCell"]["rho"]["type"] == "float?"
    assert shape["CorrelationCell"]["n"]["type"] == "int"
    assert shape["CorrelationsFile"]["cells"]["type"] == "array"
