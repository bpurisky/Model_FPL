"""`golden_reductions.json` — the fixture that polices §5.6.2 (§5.11.2).

The TypeScript side of this bargain is `reduce.golden.test.ts`, which runs
`src/query/reduce.ts` over the embedded rows and checks it against the
answers here. These tests cover the other side: that the fixture says what
it claims to, and that the conventions it pins down are the ones the
browser was told to implement.

The load-bearing case is the empty set. §5.6.2 permits the browser to
reduce on the grounds that these seven operations "cannot silently
disagree with Python because there is nothing to disagree about" — and
the one place they genuinely can is when there is nothing to reduce.
Python's own `sum([])` is 0; this export says it is null, because a bar
with no rows behind it has no height and drawing it at zero would put it
on the axis beside genuine zeroes (§5.3.3).
"""

from __future__ import annotations

import math

import polars as pl
import pytest

from web.export.reductions import (
    QUANTILES,
    SAMPLE_COLUMNS,
    SAMPLE_ROWS,
    SAMPLE_SEASONS,
    TOLERANCE,
    UNPARAMETERISED,
    apply,
    build_golden_reductions,
    present,
)


@pytest.fixture(scope="module")
def fixture_file():
    return build_golden_reductions()


def test_every_permitted_reduction_is_covered(fixture_file):
    """§5.6.2 names seven. A fixture covering six would leave one of them
    shipping to the browser with no test behind it."""
    names = {case.fn for case in fixture_file.cases}
    assert names == {"count", "sum", "mean", "median", "min", "max", "quantile"}


def test_quantile_carries_its_parameter_and_the_others_do_not(fixture_file):
    for case in fixture_file.cases:
        if case.fn == "quantile":
            assert case.q in QUANTILES, f"{case.q} is not one of the sampled quantiles"
        else:
            assert case.q is None, f"{case.fn} does not take a parameter"


def test_the_sample_predates_the_2025_26_metrics(fixture_file):
    """The empty-set rule has to be tested against real absence.

    `cbi_per90` does not exist before 2025-26, and the sample is drawn
    from 2023-24 and 2024-25, so it is null in every row rather than
    merely sparse. A constructed blank column would test the same code
    path while asserting nothing about the export.
    """
    assert "2025-26" not in SAMPLE_SEASONS
    index = fixture_file.columns.index("cbi_per90")
    assert all(row[index] is None for row in fixture_file.rows)


def test_an_empty_set_reduces_to_null_except_for_count(fixture_file):
    empty = [case for case in fixture_file.cases if case.n == 0]
    assert empty, "nothing in the sample is entirely absent"

    for case in empty:
        if case.fn == "count":
            assert case.value == 0, "how many rows are there always has an answer"
        else:
            assert case.value is None, (
                f"{case.column} {case.fn} reduced an empty set to {case.value}; "
                "§5.3.3 says absent is not zero"
            )


def test_n_counts_present_values_not_group_size(fixture_file):
    """`n` is the count of rows carrying a value, which is what a surface
    means when it prints one beside a reduced number (§5.6.3)."""
    for name in fixture_file.columns:
        index = fixture_file.columns.index(name)
        values = [row[index] for row in fixture_file.rows]
        expected = len(present(values))
        for case in fixture_file.cases:
            if case.column == name:
                assert case.n == expected


def test_answers_are_computed_from_the_rounded_values(fixture_file):
    """The subtle one, and getting it backwards makes CI unpassable.

    The TypeScript reads six decimal places out of this file. A golden
    derived from full-precision doubles would disagree in the twelfth
    decimal and fail the 1e-12 check for a reason having nothing to do
    with the reductions.
    """
    for name in fixture_file.columns:
        index = fixture_file.columns.index(name)
        values = [row[index] for row in fixture_file.rows]
        for case in fixture_file.cases:
            if case.column != name:
                continue
            recomputed = apply(values, case.fn, case.q)
            if case.value is None:
                assert recomputed is None
            else:
                assert recomputed == pytest.approx(case.value, abs=TOLERANCE)


def test_nulls_are_dropped_rather_than_read_as_zero():
    values = [1.0, None, 3.0, None]
    assert apply(values, "count") == 2
    assert apply(values, "mean") == 2.0
    assert apply(values, "sum") == 4.0
    # Averaging the nulls in as zero would give 1.0, which is a number no
    # observation supports.
    assert apply(values, "mean") != 1.0


def test_reductions_are_order_independent():
    """The property §5.6.2's whole argument rests on."""
    values = [3.5, 1.25, 9.0, 2.75, 8.5, 0.5]
    for fn in UNPARAMETERISED:
        assert apply(values, fn) == apply(list(reversed(values)), fn)
        assert apply(values, fn) == apply(sorted(values), fn)


def test_median_is_quantile_at_one_half():
    """Delegating rather than implementing twice is what keeps the two
    from acquiring different tie conventions."""
    for values in ([1.0, 2.0, 3.0], [1.0, 2.0, 3.0, 4.0], [5.0]):
        assert apply(values, "median") == apply(values, "quantile", 0.5)


def test_quantile_uses_linear_interpolation():
    """Named and pinned because it is the one §5.6.2 operation with
    competing conventions in the wild -- R alone ships nine."""
    values = [1.0, 2.0, 3.0, 4.0]
    assert apply(values, "quantile", 0.25) == pytest.approx(1.75)
    assert apply(values, "quantile", 0.5) == pytest.approx(2.5)
    assert apply(values, "quantile", 0.75) == pytest.approx(3.25)
    # The extremes are the extremes, with no extrapolation past them.
    assert apply(values, "quantile", 0.0) == 1.0
    assert apply(values, "quantile", 1.0) == 4.0


def test_sum_survives_an_input_where_order_genuinely_matters():
    """Floating-point addition is not associative, so a reduction that
    claims order-independence has to impose one.

    These three values are the demonstration: added left to right they
    give 0.0, added right to left they give 1.0, and a `sum` that simply
    took the caller's order would return whichever the query engine
    happened to emit. Sorting first is what makes the answer a property
    of the set rather than of the row order.
    """
    values = [1.0, 1e100, 1.0, -1e100]

    # Sequential addition, which is what `apply` does once it has imposed
    # an order. Not `sum()`: CPython 3.12 gives that Neumaier compensated
    # summation, so the builtin hides the very effect under test.
    def sequential(items: list[float]) -> float:
        total = 0.0
        for item in items:
            total += item
        return total

    assert sequential(values) != sequential(list(reversed(values))), (
        "the input is not actually order-sensitive, so this proves nothing"
    )

    answer = apply(values, "sum")
    assert answer == apply(list(reversed(values)), "sum")
    assert answer == apply(sorted(values), "sum")


def test_the_fixture_is_self_contained(fixture_file):
    """§5.14.8: a fresh clone works with no pipeline run, and this test
    cannot read `panel.parquet` because §5.3.4 does not commit it."""
    assert fixture_file.columns
    assert len(fixture_file.rows) == SAMPLE_ROWS
    assert all(len(row) == len(fixture_file.columns) for row in fixture_file.rows)
    assert set(fixture_file.columns) == set(SAMPLE_COLUMNS)
    assert fixture_file.tolerance == TOLERANCE


def test_the_fixture_is_deterministic():
    """It is committed. A fixture that changed every build would produce a
    diff on every run and a CI failure on none."""
    first = build_golden_reductions()
    second = build_golden_reductions()
    assert first.rows == second.rows
    assert [(c.column, c.fn, c.q, c.value, c.n) for c in first.cases] == [
        (c.column, c.fn, c.q, c.value, c.n) for c in second.cases
    ]


def test_refuses_a_sample_that_would_leave_the_empty_rule_untested():
    """The guard that keeps this fixture honest as the archive grows.

    Once 2025-26 has a second season behind it, someone will widen
    SAMPLE_SEASONS and every column will populate. The fixture must fail
    loudly then rather than quietly stop testing the case it exists for.
    """
    # Inside the sampled seasons, so the rows survive the filter -- and
    # every column populated, which is the state the guard exists to
    # refuse.
    panel = pl.DataFrame(
        {
            "season": [SAMPLE_SEASONS[0]] * 8,
            "element_id": list(range(8)),
            "gw": list(range(1, 9)),
            **{name: [1.0] * 8 for name in SAMPLE_COLUMNS},
        }
    )
    with pytest.raises(ValueError, match="entirely absent"):
        build_golden_reductions(panel=panel)


def test_refuses_a_panel_missing_a_sample_column():
    panel = pl.DataFrame(
        {
            "season": ["2024-25"] * 4,
            "element_id": list(range(4)),
            "gw": list(range(1, 5)),
            "total_points": [1.0] * 4,
        }
    )
    with pytest.raises(ValueError, match="missing sample columns"):
        build_golden_reductions(panel=panel)


def test_present_drops_non_finite_values():
    """A NaN is not a measurement, and it would poison every reduction it
    reached — `min` and `max` silently, `mean` visibly."""
    assert present([1.0, math.nan, 2.0, math.inf]) == [1.0, 2.0]
