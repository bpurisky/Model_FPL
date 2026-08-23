"""§5.11.1 over `golden_spearman.json`.

The premise of this file is that someone with nothing but the JSON can
recompute every answer in it. If that is false the §5.14.2 CI check is
unpassable and nobody finds out until `spearman.ts` exists, so the central
test here does exactly what the TypeScript will do: read the embedded
values, recompute, and compare within the file's own stated tolerance.
"""

from __future__ import annotations

import json
import math
from functools import lru_cache
from pathlib import Path

import polars as pl
import pytest

from backtest.report import spearman
from web.export.contract import GoldenSpearmanFile
from web.export.golden import (
    PRECISION,
    SAMPLE_ROWS,
    TOLERANCE,
    build_golden_spearman,
    stride_sample,
)
from web.export.panel import build_panel

GOLDEN_PATH = Path("data/web/v1/golden_spearman.json")


@lru_cache(maxsize=1)
def _built() -> GoldenSpearmanFile:
    """Built from the committed archive, not from the gitignored panel —
    the same reason the correlations tests do it that way."""
    return build_golden_spearman(panel=build_panel())


def _committed() -> dict | None:
    if not GOLDEN_PATH.exists():  # pragma: no cover
        return None
    return json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))


# --- the premise -----------------------------------------------------------


def test_every_golden_is_reproducible_from_the_embedded_values_alone():
    """The whole point of the file, and the exact procedure `spearman.ts`
    will follow: take the two columns out of `samples`, drop incomplete
    pairs, run the method, and land within `tolerance` of `rho`.

    If this fails, §5.14.2's CI check cannot be satisfied by any correct
    port — the fixture would be asking for an answer its own inputs do not
    produce.
    """
    payload = _committed()
    if payload is None:  # pragma: no cover
        pytest.skip("golden_spearman.json not generated yet -- run `python -m web.export golden`")

    samples = {s["group"]: s for s in payload["samples"]}
    tolerance = payload["tolerance"]
    checked = 0

    for pair in payload["pairs"]:
        sample = samples[pair["group"]]
        ia, ib = sample["metrics"].index(pair["a"]), sample["metrics"].index(pair["b"])
        both = [(r[ia], r[ib]) for r in sample["rows"] if r[ia] is not None and r[ib] is not None]

        assert len(both) == pair["n"], f"{pair['group']} {pair['a']}x{pair['b']}: n disagrees"
        if pair["rho"] is None:
            continue

        recomputed = spearman(
            pl.Series([x for x, _ in both], dtype=pl.Float64),
            pl.Series([y for _, y in both], dtype=pl.Float64),
        )
        assert abs(recomputed - pair["rho"]) <= tolerance, (
            f"{pair['group']} {pair['a']}x{pair['b']}: "
            f"file says {pair['rho']}, values give {recomputed}"
        )
        checked += 1

    assert checked >= 50, "§5.6.1 requires at least 50 pairs with a computed rho"


def test_the_embedded_values_are_rounded_to_the_stated_precision():
    """The goldens were computed *after* rounding. A value carrying more
    digits than `precision` claims would mean a consumer reading the file
    sees different numbers from the ones the answers came from, and 1e-9
    would fail for a reason unrelated to the port."""
    payload = _committed()
    if payload is None:  # pragma: no cover
        pytest.skip("golden_spearman.json not generated yet")

    precision = payload["precision"]
    for sample in payload["samples"]:
        for row in sample["rows"]:
            for value in row:
                if value is not None:
                    assert value == round(value, precision)


# --- the fixture has to contain the hard cases -----------------------------


def test_the_sample_contains_ties():
    """Ties are why §5.6.1 demands a port rather than any Spearman: the
    average-rank convention is a choice a fresh implementation can quietly
    make differently. A fixture of all-distinct values would not notice."""
    file = _built()

    tied = 0
    for sample in file.samples:
        for index in range(len(sample.metrics)):
            column = [r[index] for r in sample.rows if r[index] is not None]
            tied += len(column) - len(set(column))

    assert tied > 0, "no repeated values anywhere -- tie handling goes untested"


def test_the_sample_contains_nulls():
    """Four metrics do not exist before 2025-26. A port that ranks a null
    column instead of dropping the pair reproduces this project's own
    ρ=0.23-for-a-perfect-correlation failure."""
    file = _built()

    nulls = sum(
        1 for sample in file.samples for row in sample.rows for value in row if value is None
    )

    assert nulls > 0


def test_pairs_with_no_spread_report_a_null_rho_and_a_real_n():
    """`saves_per90` is 0.0 for every outfielder — present, not missing.
    The port should return the same nothing rather than a zero."""
    file = _built()

    degenerate = [p for p in file.pairs if p.rho is None]

    assert degenerate, "expected at least one degenerate pair in a real sample"
    assert all(p.n > 0 for p in degenerate), "n is still reported -- the rows exist"


# --- shape and provenance ---------------------------------------------------


def test_the_tolerance_and_precision_travel_in_the_file():
    """So the TypeScript reads both from the fixture rather than keeping
    second copies that can drift from it."""
    file = _built()

    assert file.tolerance == TOLERANCE == 1e-9
    assert file.precision == PRECISION
    assert "ties-averaged ranks" in file.method
    assert "report.py" in file.method


def test_every_pair_names_a_group_and_metrics_the_samples_carry():
    file = _built()
    samples = {s.group: s for s in file.samples}

    for pair in file.pairs:
        assert pair.group in samples
        assert pair.a in samples[pair.group].metrics
        assert pair.b in samples[pair.group].metrics
        assert pair.a != pair.b


def test_every_sample_row_is_as_wide_as_its_metric_list():
    """The matrix is positional — a short row would silently shift every
    metric after the gap."""
    file = _built()

    for sample in file.samples:
        assert sample.rows, f"{sample.group} has no rows"
        assert len(sample.rows) <= SAMPLE_ROWS
        for row in sample.rows:
            assert len(row) == len(sample.metrics)


def test_nothing_unserializable_reaches_the_json():
    payload = _built().model_dump_json()

    assert "NaN" not in payload and "Infinity" not in payload
    for pair in json.loads(payload)["pairs"]:
        assert pair["rho"] is None or math.isfinite(pair["rho"])


# --- determinism ------------------------------------------------------------


def test_the_sample_is_stable_across_builds():
    """A committed fixture that resampled every run would diff on every
    build and catch a regression on none."""
    panel = build_panel()

    first = build_golden_spearman(panel=panel)
    second = build_golden_spearman(panel=panel)

    assert [s.rows for s in first.samples] == [s.rows for s in second.samples]
    assert [p.rho for p in first.pairs] == [p.rho for p in second.pairs]


def test_stride_sample_spans_the_frame_rather_than_taking_the_head():
    """Taking the head would draw entirely from one season and the low end
    of the element-id range, which is a weak sample for a rank statistic."""
    df = pl.DataFrame({"i": list(range(100))})

    sampled = stride_sample(df, 10)["i"].to_list()

    assert len(sampled) == 10
    assert sampled[0] == 0
    assert sampled[-1] >= 90, "the sample must reach the far end of the frame"
    assert sampled == sorted(sampled)


def test_stride_sample_returns_everything_when_the_frame_is_small():
    df = pl.DataFrame({"i": [1, 2, 3]})

    assert stride_sample(df, 10)["i"].to_list() == [1, 2, 3]


def test_the_committed_golden_file_matches_a_fresh_build():
    payload = _committed()
    if payload is None:  # pragma: no cover
        pytest.skip("golden_spearman.json not generated yet")

    current = json.loads(_built().model_dump_json())
    for p in (payload, current):
        del p["header"]

    assert payload == current, "data/web/v1/golden_spearman.json is stale -- regenerate it"
