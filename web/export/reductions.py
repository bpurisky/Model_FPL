"""`golden_reductions.json` — the fixtures that police §5.6.2's reductions.

§5.6.2 widened the browser's licence from "renders and slices" to
"renders, slices, and reduces", and justified it in one sentence:

    "These are exact, deterministic, order-independent reductions with no
    estimator choice, no tie-handling convention, and no distributional
    assumption. They cannot silently disagree with Python because there is
    nothing to disagree about."

That claim is only true if two things hold, and neither is free. There
must be exactly **one** implementation on the TypeScript side — hence
`src/query/reduce.ts`, with DuckDB doing the reading, filtering and
grouping and the reduction itself in plain TS — and a golden test must
hold it against Python, which §5.11.2 requires: "exact equality for
integer reductions and 1e-12 for floats".

This file is that fixture, and it is built the same way `golden.py`
builds the Spearman one, for the same reasons:

**Self-contained.** The TypeScript cannot read `panel.parquet` — §5.3.4
does not commit it and §5.14.8 requires a fresh clone to work with no
pipeline run — so the inputs travel with the answers.

**Answers computed from the rounded values.** The TS side reads six
decimal places out of this JSON. A golden derived from full-precision
doubles would disagree in the twelfth decimal and fail the check for a
reason having nothing to do with the reductions. The rounding happens
first and the golden is the truth about *these* numbers.

**The sample deliberately predates 2025-26.** Rows are drawn from
2023-24 and 2024-25 only, so `cbi_per90` and its three siblings are
absent from every row rather than merely sparse. That is what puts the
empty-set rule under test with real absence instead of a constructed
blank: a set with nothing in it has no mean, no sum and no maximum, but
it always has a count, and it is zero. Getting that backwards is how a
bar chart ends up drawing a nothing at the axis beside genuine zeroes
(§5.3.3).

Two conventions are pinned here rather than left to the reader, because
they are the only places these seven names admit a choice at all:

**`quantile` uses linear interpolation** — polars' `interpolation="linear"`,
which is also numpy's default method and therefore what `median` means
here too. R alone ships nine quantile definitions; naming one and testing
it is what keeps `quantile` inside "no estimator choice".

**`sum` adds in ascending order.** Floating-point addition is not
associative, so a reduction claiming order-independence has to impose an
order. Both sides sort first, which makes the two bit-identical rather
than merely close.

Deviation from §5.2.1's module list, recorded per §5.16: that layout
names no module for this file, exactly as it names none for `golden.py`.
It lives beside its sibling for the same reason — publishing inputs and
answers to check an implementation against is a different job from
publishing numbers to render.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path

import polars as pl

from web.export.contract import (
    GoldenReductionsFile,
    ReductionCase,
    build_header,
    json_safe,
)
from web.export.correlations import PANEL_PATH, correlation_basis
from web.export.golden import PRECISION, stride_sample
from web.export.normalize import load_frontend_config

logger = logging.getLogger("web.export.reductions")

# §5.11.2: "exact equality for integer reductions and 1e-12 for floats".
# Carried in the file so the TypeScript test reads the number it must meet
# from the fixture rather than keeping a second copy that can drift.
TOLERANCE = 1e-12

# The six §5.6.2 names that take no parameter, in the order a reader of
# the spec meets them.
UNPARAMETERISED = ("count", "sum", "mean", "median", "min", "max")

# `quantile` is the seventh, and it needs one. These four are enough to
# catch an off-by-one in the interpolation at both ends and in the middle
# without turning the fixture into a table of near-duplicates.
QUANTILES = (0.1, 0.25, 0.75, 0.9)

# Seasons the sample is drawn from -- see the module docstring. Both
# predate the four defensive metrics added for 2025-26.
SAMPLE_SEASONS = ("2023-24", "2024-25")

# Rows in the embedded sample. Larger than `golden.py`'s 32 because a
# reduction is cheap to check and an even count exercises the two-sided
# median path that an odd one never reaches.
SAMPLE_ROWS = 64

# What the sample carries. Chosen to span the three shapes the panel
# actually contains rather than to be tidy:
#
#   dense integers   -- total_points, minutes, value
#   dense floats     -- clean_sheet_prob
#   sparse floats    -- xg_per90 and friends, null below the minutes floor
#   entirely absent  -- cbi_per90, which does not exist before 2025-26
SAMPLE_COLUMNS = (
    "total_points",
    "minutes",
    "value",
    "clean_sheet_prob",
    "xg_per90",
    "xgi_per90",
    "bps_per90",
    "cbi_per90",
)


def present(values: list[float | None]) -> list[float]:
    """Non-null, finite values, ascending.

    The mirror of `present()` in `src/query/reduce.ts`, and the reason
    every case below agrees on what "present" means. Nulls are dropped,
    never coerced to zero (§5.3.3) -- a null here is "this player was
    below the minutes floor that gameweek", and averaging it in as zero
    would drag the mean toward a value no player recorded.
    """
    kept = [float(v) for v in values if v is not None and math.isfinite(v)]
    kept.sort()
    return kept


def apply(values: list[float | None], fn: str, q: float | None = None) -> float | None:
    """One reduction, matching `reduce.ts` exactly.

    `polars` does the quantile work rather than a hand-rolled
    interpolation, so the golden is an independent answer rather than a
    transcription of the TypeScript. The sum is hand-rolled precisely
    because it must *not* be independent: a vectorised pairwise summation
    would differ from a sequential add in the last bits, and the point of
    sorting first is to make the two sides bit-identical.
    """
    sorted_values = present(values)

    if fn == "count":
        return float(len(sorted_values))
    if len(sorted_values) == 0:
        return None

    if fn == "sum":
        total = 0.0
        for value in sorted_values:
            total += value
        return total
    if fn == "mean":
        total = 0.0
        for value in sorted_values:
            total += value
        return total / len(sorted_values)
    if fn == "median":
        return float(pl.Series(sorted_values, dtype=pl.Float64).quantile(0.5, interpolation="linear"))
    if fn == "min":
        return sorted_values[0]
    if fn == "max":
        return sorted_values[-1]
    if fn == "quantile":
        assert q is not None, "quantile needs a q"
        return float(pl.Series(sorted_values, dtype=pl.Float64).quantile(q, interpolation="linear"))

    raise ValueError(f"{fn} is not one of §5.6.2's seven reductions")


def build_golden_reductions(
    panel: pl.DataFrame | None = None,
    panel_path: Path = PANEL_PATH,
    config: dict | None = None,
) -> GoldenReductionsFile:
    """The fixture: embedded inputs, and the answer for every case."""
    if panel is None:
        if not panel_path.exists():
            raise FileNotFoundError(
                f"{panel_path} not found — run `python -m web.export panel` first. "
                "It is a build artifact and §5.3.4 does not commit it."
            )
        panel = pl.read_parquet(panel_path)

    config = config or load_frontend_config()
    per_fixture = config["normalization"]["minutes_per_fixture_floor"]

    columns = [c for c in SAMPLE_COLUMNS if c in panel.columns]
    missing = set(SAMPLE_COLUMNS) - set(columns)
    if missing:
        # A silently shorter fixture is a fixture that stops covering the
        # case it was built for.
        raise ValueError(f"panel is missing sample columns: {sorted(missing)}")

    scoped = panel.filter(pl.col("season").is_in(list(SAMPLE_SEASONS)))
    sampled = stride_sample(
        scoped.sort(["season", "element_id", "gw"]),
        SAMPLE_ROWS,
    )

    rows: list[list[float | None]] = [
        [None if value is None else round(float(value), PRECISION) for value in row]
        for row in sampled.select(columns).iter_rows()
    ]

    by_column = {
        name: [row[index] for row in rows] for index, name in enumerate(columns)
    }

    cases: list[ReductionCase] = []
    for name in columns:
        values = by_column[name]
        n = int(len(present(values)))
        for fn in UNPARAMETERISED:
            cases.append(
                ReductionCase(
                    column=name,
                    fn=fn,
                    q=None,
                    value=json_safe(apply(values, fn)),
                    n=n,
                )
            )
        for q in QUANTILES:
            cases.append(
                ReductionCase(
                    column=name,
                    fn="quantile",
                    q=q,
                    value=json_safe(apply(values, "quantile", q)),
                    n=n,
                )
            )

    empty = [c for c in cases if c.n == 0]
    if not empty:
        # The empty-set rule is the one this fixture exists to pin down;
        # a sample that happened to populate every column would leave it
        # untested while still looking green.
        raise ValueError(
            "no sampled column is entirely absent — the empty-set rule "
            "would ship untested. Check SAMPLE_SEASONS still predates the "
            "2025-26 defensive metrics."
        )

    logger.info(
        "%d reduction cases over %d columns x %d rows (%d columns entirely absent)",
        len(cases), len(columns), len(rows), len(empty) // (len(UNPARAMETERISED) + len(QUANTILES)),
    )

    return GoldenReductionsFile(
        header=build_header(
            rows=len(cases),
            source_gameweek=None,  # fixtures describe a method, not a gameweek
            normalization_basis=correlation_basis(per_fixture),
        ),
        method=(
            "nulls dropped, values sorted ascending; sum added sequentially in that "
            "order; quantile and median by linear interpolation (polars, method=linear). "
            "Empty set reduces to null for every name except count, which is 0. "
            "See web/export/reductions.py and src/query/reduce.ts"
        ),
        tolerance=TOLERANCE,
        precision=PRECISION,
        columns=columns,
        rows=rows,
        cases=cases,
    )
