"""§5.11.1 over `scorecard.json`: that the reshaping preserves the report's
own numbers, and that the rollup rows exist because they cannot be
recovered from the detail.

Driven over small constructed frames rather than the real walk-forward —
that costs ~27 seconds and would dominate the suite. The one test that
reads real numbers reads the *committed* file instead of rebuilding it.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import polars as pl
import pytest

from backtest.report import mae, summarize
from web.export.scorecard import (
    EVENT_MODEL,
    build_calibration,
    build_error_by_event,
    build_rows,
    build_scorecard,
)

RESULT_COLUMNS = [
    "element_id", "prediction", "position", "total_points", "minutes",
    "goals_scored", "assists", "clean_sheets", "bonus", "bps",
    "season", "gw", "baseline", "error",
]


def _results(rows: list[dict]) -> pl.DataFrame:
    """`error` is derived rather than passed: the harness computes it, and
    a test that supplied its own could disagree with the harness and still
    pass."""
    for row in rows:
        unknown = set(row) - set(RESULT_COLUMNS)
        assert not unknown, f"unknown key(s) in test row: {unknown}"
    df = pl.DataFrame(
        {c: [row.get(c) for row in rows] for c in RESULT_COLUMNS},
        schema={
            "element_id": pl.Int64, "prediction": pl.Float64, "position": pl.Utf8,
            "total_points": pl.Int64, "minutes": pl.Int64, "goals_scored": pl.Int64,
            "assists": pl.Int64, "clean_sheets": pl.Int64, "bonus": pl.Int64,
            "bps": pl.Int64, "season": pl.Utf8, "gw": pl.Int64,
            "baseline": pl.Utf8, "error": pl.Float64,
        },
    )
    return df.with_columns((pl.col("prediction") - pl.col("total_points")).alias("error"))


def _row(element_id, prediction, points, *, gw=1, season="2025-26", model="m", position="MID", minutes=90):
    return {
        "element_id": element_id, "prediction": prediction, "position": position,
        "total_points": points, "minutes": minutes, "goals_scored": 0, "assists": 0,
        "clean_sheets": 0, "bonus": 0, "bps": 0, "season": season, "gw": gw,
        "baseline": model, "error": 0.0,
    }


def _decomposition() -> dict[str, list]:
    return {
        "predicted_components": [{"goals": 1.0, "minutes": 2.0}],
        "actual_components": [{"goals": 0.0, "minutes": 2.5}],
        "predicted_minutes_dist": [{"p_blank": 0.1, "p_short": 0.2, "p_full": 0.7}],
        "actual_minutes": [90],
    }


# --- the reshaping preserves the report's numbers ------------------------


def test_a_pooled_row_matches_report_summarize_exactly():
    """The whole premise of this module is that it reshapes rather than
    recomputes. If a pooled row ever disagreed with `report.summarize`
    over the same frame, the app and `backtest/report.json` would publish
    two different accuracies for the same model."""
    results = _results([
        _row(1, 4.0, 6, gw=1), _row(2, 2.0, 1, gw=1),
        _row(3, 5.0, 5, gw=2), _row(4, 1.0, 0, gw=2),
    ])

    pooled = [r for r in build_rows(results) if r.season is None][0]
    reference = summarize(results, ["baseline"])["m"]

    assert pooled.n == reference["n"]
    assert pooled.mae == pytest.approx(reference["mae"])
    assert pooled.rmse == pytest.approx(reference["rmse"])
    assert pooled.spearman_mean == pytest.approx(
        reference["spearman_within_position"]["mean"], nan_ok=True
    )


def test_the_pooled_row_is_not_recoverable_from_the_gameweek_rows():
    """Why the rollups ship as rows rather than being left to the client.

    Gameweeks carry different row counts, so the unweighted mean of
    per-gameweek MAEs is not the pooled MAE. A UI that averaged the detail
    would publish a third number matching neither this file nor the
    report — and §5.6 forbids it aggregating inferential statistics
    anyway. Constructed with a lopsided split so the gap is unmistakable.
    """
    results = _results(
        [_row(1, 10.0, 0, gw=1)]                                   # 1 row,  error 10
        + [_row(i, 1.0, 1, gw=2) for i in range(2, 12)]            # 10 rows, error 0
    )

    rows = build_rows(results)
    pooled = [r for r in rows if r.season is None][0]
    per_gw = [r for r in rows if r.gw is not None]

    assert pooled.mae == pytest.approx(10.0 / 11)
    naive = sum(r.mae for r in per_gw) / len(per_gw)
    assert naive == pytest.approx(5.0)
    assert pooled.mae != pytest.approx(naive)


def test_every_model_gets_all_three_grains():
    results = _results([
        _row(1, 4.0, 6, gw=1, season="2024-25", model="a"),
        _row(2, 4.0, 6, gw=2, season="2025-26", model="a"),
        _row(3, 4.0, 6, gw=1, season="2024-25", model="b"),
        _row(4, 4.0, 6, gw=2, season="2025-26", model="b"),
    ])

    rows = build_rows(results)

    assert len([r for r in rows if r.season is None and r.gw is None]) == 2
    assert len([r for r in rows if r.season is not None and r.gw is None]) == 4
    assert len([r for r in rows if r.gw is not None]) == 4


# --- degenerate slices (§5.3.3) ------------------------------------------


def test_a_single_row_gameweek_nulls_the_rank_statistic_but_keeps_the_error():
    """The two kinds of statistic behave differently at n=1 and both are
    right to.

    MAE and RMSE are error magnitudes, defined over one row: the model was
    2.0 out, and RMSE says so. A rank correlation is not — there is
    nothing to rank against — so `report.py` returns NaN and this file
    carries null. A 0.0 there would claim the model ranked the gameweek
    with no skill rather than that ranking was undefined.
    """
    results = _results([_row(1, 4.0, 6, gw=1)])

    row = [r for r in build_rows(results) if r.gw == 1][0]

    assert row.n == 1
    assert row.mae == pytest.approx(2.0)
    assert row.rmse == pytest.approx(2.0)
    assert row.spearman_mean is None
    assert [p.rho for p in row.spearman_by_position] == [None]


def test_no_nan_survives_into_the_json():
    """One degenerate gameweek must not take down the whole surface:
    `float('nan')` serializes to a bare `NaN` token that `JSON.parse`
    rejects outright."""
    results = _results([_row(1, 4.0, 6, gw=1), _row(2, 3.0, 3, gw=2), _row(3, 1.0, 0, gw=2)])

    payload = build_scorecard(results=results, decomposition=_decomposition()).model_dump_json()

    assert "NaN" not in payload and "Infinity" not in payload
    reparsed = json.loads(payload)
    for row in reparsed["rows"]:
        for field in ("mae", "rmse", "spearman_mean"):
            assert row[field] is None or math.isfinite(row[field])


# --- shape ---------------------------------------------------------------


def test_the_mean_spearman_carries_no_p_value():
    """`report.spearman_within_position_significance` deliberately omits a
    mean row: no sampling distribution describes the unweighted mean of
    four correlations, so a p-value beside it would be invented. The
    export must not put one back."""
    results = _results([
        _row(i, float(i), i, position=p)
        for p in ("GK", "DEF", "MID", "FWD") for i in range(1, 5)
    ])

    row = [r for r in build_rows(results) if r.season is None][0]

    assert row.spearman_mean is not None
    assert {p.position for p in row.spearman_by_position} == {"GK", "DEF", "MID", "FWD"}
    assert all(p.p_value is not None or p.rho is None for p in row.spearman_by_position)
    assert not hasattr(row, "spearman_mean_p_value")


def test_calibration_bins_are_integers_so_they_sort():
    """`qcut` labels bins as strings, where "10" sorts before "2"."""
    results = _results([_row(i, float(i), i) for i in range(1, 21)])

    bins = build_calibration(results)

    assert bins, "expected calibration bins"
    assert all(isinstance(b.bin, int) for b in bins)
    assert [b.bin for b in bins] == sorted(b.bin for b in bins)


def test_component_detail_covers_the_event_model_only():
    """The three scalar baselines predict one number and have no
    components to decompose; the file names which model the decomposition
    belongs to rather than leaving it to be inferred."""
    results = _results([_row(1, 4.0, 6), _row(2, 2.0, 1, model="trailing_mean")])

    file = build_scorecard(results=results, decomposition=_decomposition())

    assert file.event_model == EVENT_MODEL
    assert {c.component for c in file.component_decomposition} == {"goals", "minutes"}
    assert file.minutes_head.n == 1


def test_error_by_event_covers_every_model():
    """Phase 1's proxy decomposition is kept precisely because it *is*
    computable for the scalar baselines."""
    results = _results([_row(1, 4.0, 6, model="a"), _row(2, 2.0, 1, model="b")])

    buckets = build_error_by_event(results)

    assert {b.model for b in buckets} == {"a", "b"}


# --- the committed file ---------------------------------------------------


def test_the_committed_scorecard_is_structurally_complete():
    """A rebuild-and-compare guard would cost ~27 seconds, so this checks
    the shape and the invariants instead: every model covered at every
    grain, no NaN, and the pooled row present for each model.

    Stated limitation: this cannot catch a stale *value*. Only a rebuild
    can, and that belongs in the deploy workflow rather than in a suite
    that runs on every change.
    """
    path = Path("data/web/v1/scorecard.json")
    if not path.exists():  # pragma: no cover
        pytest.skip("scorecard.json not generated yet -- run `python -m web.export scorecard`")

    d = json.loads(path.read_text(encoding="utf-8"))
    models, seasons = set(d["models"]), set(d["seasons"])

    assert d["event_model"] in models
    assert len(d["rows"]) == d["header"]["rows"]
    pooled = {r["model"] for r in d["rows"] if r["season"] is None}
    assert pooled == models, "every model needs its pooled row"
    per_season = {(r["model"], r["season"]) for r in d["rows"] if r["season"] and r["gw"] is None}
    assert per_season == {(m, s) for m in models for s in seasons}
    assert {c["model"] for c in d["calibration"]} == models
    assert all(
        r[f] is None or math.isfinite(r[f])
        for r in d["rows"] for f in ("mae", "rmse", "spearman_mean")
    )


def test_the_committed_scorecard_agrees_with_the_published_headline():
    """The event model's pooled figures are quoted in the project's own
    records (MAE 1.0395, RMSE 2.1215, within-position Spearman 0.7202).
    If the export ever reshapes its way to different numbers, that is the
    single most visible thing it could get wrong."""
    path = Path("data/web/v1/scorecard.json")
    if not path.exists():  # pragma: no cover
        pytest.skip("scorecard.json not generated yet")

    d = json.loads(path.read_text(encoding="utf-8"))
    pooled = [
        r for r in d["rows"]
        if r["model"] == d["event_model"] and r["season"] is None and r["gw"] is None
    ][0]

    assert pooled["n"] == 83035
    assert pooled["mae"] == pytest.approx(1.0395, abs=5e-5)
    assert pooled["rmse"] == pytest.approx(2.1215, abs=5e-5)
    assert pooled["spearman_mean"] == pytest.approx(0.7202, abs=5e-5)


def test_the_committed_scorecard_beats_every_baseline_on_mae():
    """§4.4's acceptance bar, asserted against what actually shipped
    rather than against a docstring."""
    path = Path("data/web/v1/scorecard.json")
    if not path.exists():  # pragma: no cover
        pytest.skip("scorecard.json not generated yet")

    d = json.loads(path.read_text(encoding="utf-8"))
    pooled = {
        r["model"]: r for r in d["rows"] if r["season"] is None and r["gw"] is None
    }
    event = pooled.pop(d["event_model"])

    assert event["mae"] < min(r["mae"] for r in pooled.values())
