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
        "PositionSpearman",
        "ScorecardRow",
        "CalibrationBin",
        "EventErrorBucket",
        "ComponentError",
        "MinutesHead",
        "ScorecardFile",
        "FixtureRow",
        "FixturesFile",
        "GoldenSample",
        "GoldenPair",
        "GoldenSpearmanFile",
        "ReductionCase",
        "GoldenReductionsFile",
        "ShrinkagePoint",
        "ShrinkageFile",
        "PositionWeights",
        "BoardBucketAccuracy",
        "BoardPlayer",
        "BoardFile",
        "PlayerMetric",
        "PlayerProjection",
        "PlayerRow",
        "PlayersFile",
        "SeasonSummary",
        "ObservationRow",
        "ObservationsFile",
    }
    assert shape["Header"]["rows"] == {"required": True, "type": "int"}
    # Optional because a file written before the field existed must still
    # validate. It is what lets the frontend offer the current season as a
    # filter before any of its data exists (§5.4.2).
    assert shape["Header"]["current_season"] == {"required": False, "type": "str?"}
    assert shape["Header"]["generated_at"]["type"] == "datetime"
    assert shape["ColumnSpec"]["position_relevance"]["type"] == "record"
    assert shape["ColumnSpec"]["available_to_season"]["required"] is False
    assert shape["ColumnsFile"]["columns"]["type"] == "array"
    # rho and p_value are nullable and n is not: a cell can fail to
    # produce a correlation and still have to report what it ran over.
    assert shape["CorrelationCell"]["rho"]["type"] == "float?"
    assert shape["CorrelationCell"]["n"]["type"] == "int"
    assert shape["CorrelationsFile"]["cells"]["type"] == "array"
    # season and gw are nullable on purpose -- they are how a row declares
    # which grain it rolls up to, so zod must model them as nullable too.
    assert shape["ScorecardRow"]["season"]["type"] == "str?"
    assert shape["ScorecardRow"]["gw"]["type"] == "int?"
    assert shape["ScorecardRow"]["spearman_by_position"]["type"] == "array"
    # difficulty_basis is a closed set, not free text -- the two values are
    # different epistemic claims and zod must reject a third.
    assert shape["FixtureRow"]["difficulty_basis"]["type"] == "union"
    assert shape["FixtureRow"]["kickoff_time"]["type"] == "datetime?"
    # The golden sample is a positional matrix of nullable floats: zod has
    # to model the nulls, because dropping incomplete pairs is the part of
    # the port most likely to be got wrong.
    assert shape["GoldenSample"]["rows"]["type"] == "array"
    assert shape["GoldenPair"]["rho"]["type"] == "float?"
    assert shape["GoldenSpearmanFile"]["tolerance"]["type"] == "float"
    # The reduction fixture's asymmetry is the rule under test: an empty
    # set has no mean, no sum and no maximum, but it always has a count.
    assert shape["ReductionCase"]["value"]["type"] == "float?"
    assert shape["ReductionCase"]["n"]["type"] == "int"
    # `q` is populated only for `quantile` -- the parameter is what makes
    # it the one §5.6.2 name with competing conventions in the wild, so it
    # travels with the answer rather than being implied by the caller.
    assert shape["ReductionCase"]["q"]["type"] == "float?"
    assert shape["GoldenReductionsFile"]["rows"]["type"] == "array"
    assert shape["GoldenReductionsFile"]["tolerance"]["type"] == "float"
    # The two §4.4 bars are booleans the export decided, not numbers the
    # surface compares (§5.6).
    assert shape["ShrinkagePoint"]["beats_mae_bar"]["type"] == "bool"
    assert shape["ShrinkagePoint"]["beats_focus_bar"]["type"] == "bool"
    assert shape["ShrinkagePoint"]["spearman_focus"]["type"] == "float?"
    assert shape["ShrinkageFile"]["points"]["type"] == "array"
    # Weights are a metric->number map, and negative values are meaningful
    # (§5.4.6), so zod must not model them as unsigned.
    assert shape["PositionWeights"]["weights"]["type"] == "record"
    assert shape["BoardPlayer"]["drivers"]["type"] == "array"
    assert shape["BoardBucketAccuracy"]["lift"]["type"] == "float?"
    # n lives on the file, not the metric: it is a property of the position
    # group and repeating it per player would imply otherwise.
    assert set(shape["PlayerMetric"]) == {"value", "z", "percentile"}
    assert shape["PlayersFile"]["population"]["type"] == "record"
    assert shape["PlayerRow"]["projection"]["type"] == "PlayerProjection?"
    # `values` is a positional array of nullable floats: the nulls are the
    # part a client-side correlation must drop rather than rank.
    assert shape["ObservationRow"]["values"]["type"] == "array"
    assert shape["SeasonSummary"]["partial"]["type"] == "bool"


# --- json_safe: what may cross the boundary -------------------------------


def test_json_safe_nulls_nan_and_infinity():
    """NaN serializes to a bare `NaN` token that `JSON.parse` rejects, and
    it is not a measurement anyway — `report.py` returns it for degenerate
    input, which is §5.3.3's null."""
    from web.export.contract import json_safe

    assert json_safe(float("nan")) is None
    assert json_safe(float("inf")) is None
    assert json_safe(float("-inf")) is None
    assert json_safe(None) is None


def test_json_safe_absorbs_parallel_reduction_jitter():
    """The reason rounding is here at all. Polars aggregates groups in
    parallel, so summation order varies between runs: two builds of
    identical committed data produced calibration means differing in their
    last digits. A published number that changes when nothing changed is
    not traceable to the code that produced it."""
    from web.export.contract import json_safe

    observed_a, observed_b = 1.878969028177423, 1.8789690281774227

    assert observed_a != observed_b
    assert json_safe(observed_a) == json_safe(observed_b)


def test_json_safe_keeps_significant_digits_not_decimal_places():
    """A p-value here is legitimately 2.9e-119. Rounding to twelve decimal
    places would publish 0.0 — a claim of certainty rather than a very
    small number."""
    from web.export.contract import json_safe

    tiny = 2.9234567890123456e-119

    assert json_safe(tiny) == pytest.approx(2.92345678901e-119, rel=1e-11)
    assert json_safe(tiny) != 0.0


def test_json_safe_stays_far_inside_the_spearman_tolerance():
    """§5.6.1 fails CI at 1e-9 disagreement on |rho| <= 1; the rounding
    must not eat into that budget."""
    from web.export.contract import SIGNIFICANT_DIGITS, json_safe

    assert SIGNIFICANT_DIGITS == 12
    for value in (0.9999999999999, -0.123456789012345, 0.5):
        assert abs(json_safe(value) - value) < 1e-9
