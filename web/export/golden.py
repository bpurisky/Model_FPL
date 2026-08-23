"""`golden_spearman.json` — the fixtures that police the Spearman port
(§5.6.1, §5.14.2).

§5.6 forbids the browser inferring, with exactly one exception: a
user-defined filter ("defenders over £6.0m with 400+ minutes") cannot be
precomputed, so `src/data/spearman.ts` is permitted as a deliberate port of
`backtest/report.py`'s method — Pearson over ties-averaged ranks. The
danger the spec names is a second implementation with its own tie-handling
producing a number no test covers, that silently disagrees with the paper
result. This file is what makes that impossible to do quietly: CI fails if
the TypeScript disagrees with any value here by more than 1e-9.

**The file is self-contained, and it has to be.** A golden ρ on its own is
unusable — the TS side needs the same inputs to compute from, and it cannot
read `panel.parquet`, which §5.3.4 does not commit. §5.14.8 requires a
fresh clone to work with no pipeline run, so a fixture depending on a build
artifact could not run in CI at all. Every value the goldens were computed
over is therefore embedded.

**ρ is computed from the rounded values in the file, not from the full
-precision ones behind them.** This is the subtle part and getting it
backwards would make the CI check unpassable: the TypeScript reads six
decimal places from this JSON, so a ρ derived from seventeen would
disagree in the tenth decimal and 1e-9 would fail — correctly, but for a
reason having nothing to do with the port. The rounding happens first and
the golden is the truth about *these* numbers.

Two properties are deliberately preserved in the sample rather than tidied
away, because they are where a reimplementation actually breaks:

**Ties.** They are the whole reason §5.6.1 insists on a port rather than
any Spearman: FPL data is dense with them, and the average-rank convention
is a choice a fresh implementation can silently make differently. Rounding
to six decimals creates a few more, which is help rather than harm.

**Nulls.** Four of the fourteen metrics do not exist before 2025-26, so
the sample carries real gaps. A port that ranks a null column instead of
dropping the pair reproduces the exact failure this project already hit
once: a *perfect* rank correlation reported as ρ=0.23 over n=10.

Deviation from §5.2.1's module list, recorded per §5.16: that layout names
no module for this file. It lives here rather than inside `correlations.py`
because the two answer different questions — one publishes correlations to
render, the other publishes inputs and answers to check an implementation
against — even though both lean on the same player-season frame.
"""

from __future__ import annotations

import logging
from itertools import combinations
from pathlib import Path

import polars as pl

from backtest.report import spearman_with_significance
from web.export.columns import MATRIX_METRICS
from web.export.contract import (
    GoldenPair,
    GoldenSample,
    GoldenSpearmanFile,
    build_header,
    json_safe,
)
from web.export.correlations import (
    GROUPS,
    PANEL_PATH,
    complete_pairs,
    correlation_basis,
    player_season_frame,
)
from web.export.normalize import load_frontend_config

logger = logging.getLogger("web.export.golden")

# §5.6.1's tolerance. Carried in the file so the TypeScript test reads the
# number it must meet from the fixture rather than hard-coding a second
# copy of it that can drift.
TOLERANCE = 1e-9

# Six decimals is far more resolution than any of these rates carry
# meaningfully, and it keeps the embedded sample to a readable size. The
# goldens are computed after this is applied -- see the module docstring.
PRECISION = 6

# Rows per group. Small on purpose: this fixture tests an algorithm, not a
# pipeline, and a wrong tie convention shows up just as clearly over thirty
# players as over three hundred.
SAMPLE_ROWS = 32


def stride_sample(df: pl.DataFrame, rows: int) -> pl.DataFrame:
    """An evenly spaced slice of `df`, deterministic across runs.

    Evenly spaced rather than the first `rows`: taking the head would draw
    entirely from one season and the low end of the element-id range, and
    a sample that never sees a high scorer is a weak test of a rank
    statistic. Deterministic rather than random because this file is
    committed, and a fixture that changed every build would produce a diff
    on every run and a CI failure on none.
    """
    if df.height <= rows:
        return df
    step = df.height / rows
    indices = sorted({int(i * step) for i in range(rows)})
    return df[indices]


def sample_for_group(people: pl.DataFrame, group: str, metrics: list[str]) -> GoldenSample:
    subset = people if group == "all" else people.filter(pl.col("position") == group)
    sampled = stride_sample(subset.sort(["season", "element_id"]), SAMPLE_ROWS)
    rows = [
        [None if value is None else round(value, PRECISION) for value in row]
        for row in sampled.select(metrics).iter_rows()
    ]
    return GoldenSample(group=group, metrics=metrics, rows=rows)


def pairs_for_sample(sample: GoldenSample) -> list[GoldenPair]:
    """Every metric pair over one group's embedded rows.

    Computed from `sample.rows` rather than from the frame it came from,
    so the goldens describe exactly the numbers a reader of this file can
    see. Anything else would be an answer to a question the file does not
    contain.
    """
    columns = {
        metric: pl.Series(metric, [row[index] for row in sample.rows], dtype=pl.Float64)
        for index, metric in enumerate(sample.metrics)
    }
    pairs: list[GoldenPair] = []
    for a, b in combinations(sample.metrics, 2):
        x, y = complete_pairs(columns[a], columns[b])
        stats = spearman_with_significance(x, y)
        pairs.append(
            GoldenPair(
                group=sample.group,
                a=a,
                b=b,
                n=int(stats["n"]),
                rho=json_safe(stats["rho"]),
            )
        )
    return pairs


def build_golden_spearman(
    panel: pl.DataFrame | None = None,
    panel_path: Path = PANEL_PATH,
    config: dict | None = None,
) -> GoldenSpearmanFile:
    """The fixture file: embedded inputs, and the answer for every pair."""
    if panel is None:
        if not panel_path.exists():
            raise FileNotFoundError(
                f"{panel_path} not found — run `python -m web.export panel` first. "
                "It is a build artifact and §5.3.4 does not commit it."
            )
        panel = pl.read_parquet(panel_path)

    config = config or load_frontend_config()
    per_fixture = config["normalization"]["minutes_per_fixture_floor"]

    metrics = [m for m in MATRIX_METRICS if m in panel.columns]
    people = player_season_frame(panel, metrics, per_fixture)

    samples = [sample_for_group(people, group, metrics) for group in GROUPS]
    pairs = [pair for sample in samples for pair in pairs_for_sample(sample)]

    computable = [p for p in pairs if p.rho is not None]
    if len(computable) < 50:
        # §5.6.1 asks for at least 50. A file that quietly shipped fewer
        # would leave the port thinly policed exactly where the spec was
        # most explicit about not letting it be.
        raise ValueError(
            f"only {len(computable)} pairs produced a rho; §5.6.1 requires at least 50"
        )

    logger.info(
        "%d golden pairs (%d computable) over %d groups x %d rows",
        len(pairs), len(computable), len(samples), SAMPLE_ROWS,
    )

    return GoldenSpearmanFile(
        header=build_header(
            rows=len(pairs),
            source_gameweek=None,  # fixtures describe a method, not a gameweek
            normalization_basis=correlation_basis(per_fixture),
        ),
        method="pearson over ties-averaged ranks, complete pairs only; see backtest/report.py:spearman",
        tolerance=TOLERANCE,
        precision=PRECISION,
        samples=samples,
        pairs=pairs,
    )
