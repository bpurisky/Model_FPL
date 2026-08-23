"""`scorecard.json` — the backtest report made legible (§5.3.2, §5.4.7).

§5.4.7 is explicit that nothing here is new analysis: every number in this
file already exists in `backtest/report.py`, and this module's whole job is
to reshape it into something a UI can filter without re-aggregating —
which §5.6 forbids it doing anyway.

The reshaping is the point. `report.build_report` nests its summaries in
dicts keyed by `str(key_values)`, so a per-season-per-model entry arrives
as the literal string `"('2023-24', 'mean_reversion')"`. That is fine for
a report a human reads once and wrong for a surface that filters, so this
module emits **tidy rows** instead: one flat record per (model, season,
gameweek), with the rollups as rows of the same shape carrying nulls in
the columns they roll up.

`season: null` means pooled across seasons and `gw: null` means pooled
across gameweeks. Those nulls are structural rather than missing data,
which is a different use of null from §5.3.3's — the UI selects a grain by
filtering on them, and a reader who wants the pooled figure reads the row
that says so rather than averaging the detail rows and getting a subtly
different number (an unweighted mean of per-gameweek MAEs is not the
pooled MAE unless every gameweek has the same row count, and they do not).

Cost: this rebuilds the entire walk-forward — 332,140 predictions across
four models, plus a second pass over 83,035 rows for the event model's
component detail. About 27 seconds. It is a build step, not something to
call from a hot path or from most tests.

Not here, and deliberately:

**§5.4.7's shrinkage-plateau panel.** The `GOALS_CONCEDED_SHRINKAGE = 0.7`
ablation is recorded as prose in `analytics/projections.py` — its
endpoints and the 0.6-0.85 plateau — but the sweep itself was never kept.
Rendering that panel means re-running the walk-forward at several
shrinkage values, which needs `project_points` to accept the shrinkage as
a parameter rather than reading the module constant. That is a change to
published model code and belongs in its own deliberate step, not smuggled
into an exporter. §5.3.2's contract for this file does not list it.

**§5.4.7's Model Board accuracy panel.** Blocked on `board.py`, which is
itself blocked on the measured finding that the Rising/Declining buckets
have no detectable edge. A hit rate for buckets that may not ship would be
worse than no panel.
"""

from __future__ import annotations

import logging
from pathlib import Path

import polars as pl

from backtest.report import (
    calibration_curve,
    error_by_event_occurrence,
    mae,
    rmse,
    spearman_within_position,
    spearman_within_position_significance,
)
from web.export.contract import (
    CalibrationBin,
    ComponentError,
    EventErrorBucket,
    MinutesHead,
    PositionSpearman,
    ScorecardFile,
    ScorecardRow,
    build_header,
    json_safe,
)

logger = logging.getLogger("web.export.scorecard")

# The name `analytics.evaluate.build_season_baselines` registers the event
# model under. Named once here so the file can say which of its models is
# the one with a component decomposition rather than leaving the UI to
# guess from the string.
EVENT_MODEL = "event_model"


def _row(group: pl.DataFrame, model: str, season: str | None, gw: int | None) -> ScorecardRow:
    """One tidy record. Every statistic comes from `report.py` — none is
    reimplemented here, so the scorecard cannot drift from the report the
    backtest command writes."""
    significance = spearman_within_position_significance(group)
    return ScorecardRow(
        model=model,
        season=season,
        gw=gw,
        n=group.height,
        mae=json_safe(mae(group)),
        rmse=json_safe(rmse(group)),
        # `spearman_within_position` owns the definition of the mean
        # (unweighted across positions, NaN dropped); recomputing it from
        # `significance` here would be a second implementation of the same
        # number, which is how two figures that should agree stop agreeing.
        spearman_mean=json_safe(spearman_within_position(group)["mean"]),
        spearman_by_position=[
            PositionSpearman(
                position=position,
                rho=json_safe(stats["rho"]),
                n=int(stats["n"]),
                p_value=json_safe(stats["p_value"]),
            )
            for position, stats in sorted(significance.items())
        ],
    )


def build_rows(results: pl.DataFrame) -> list[ScorecardRow]:
    """Three grains, one shape: per (model, season, gameweek), per
    (model, season), and per model pooled.

    Emitted rather than left to the client because the pooled figure is
    not recoverable from the detail rows — gameweeks carry different row
    counts, so the mean of per-gameweek MAEs is not the pooled MAE, and a
    UI that averaged the detail would quietly publish a third number that
    matches neither this file nor `backtest/report.json`.
    """
    rows: list[ScorecardRow] = []
    for model in sorted(results["baseline"].unique().to_list()):
        model_df = results.filter(pl.col("baseline") == model)
        rows.append(_row(model_df, model, None, None))
        for season in sorted(model_df["season"].unique().to_list()):
            season_df = model_df.filter(pl.col("season") == season)
            rows.append(_row(season_df, model, season, None))
            for gw in sorted(season_df["gw"].unique().to_list()):
                rows.append(_row(season_df.filter(pl.col("gw") == gw), model, season, int(gw)))
    return rows


def build_calibration(results: pl.DataFrame) -> list[CalibrationBin]:
    """Deciles of predicted points against realized, per model.

    `calibration_curve` labels its bins as strings because polars' `qcut`
    does; they are cast back to int here so the UI can order them without
    knowing that, and because "10" sorts before "2" as a string.
    """
    bins: list[CalibrationBin] = []
    for model in sorted(results["baseline"].unique().to_list()):
        for entry in calibration_curve(results.filter(pl.col("baseline") == model)):
            bins.append(
                CalibrationBin(
                    model=model,
                    bin=int(entry["bin"]),
                    n=int(entry["n"]),
                    mean_prediction=json_safe(entry["mean_prediction"]),
                    mean_actual=json_safe(entry["mean_actual"]),
                )
            )
    return bins


def build_error_by_event(results: pl.DataFrame) -> list[EventErrorBucket]:
    buckets: list[EventErrorBucket] = []
    for model in sorted(results["baseline"].unique().to_list()):
        breakdown = error_by_event_occurrence(results.filter(pl.col("baseline") == model))
        for bucket, stats in breakdown.items():
            buckets.append(
                EventErrorBucket(
                    model=model,
                    bucket=bucket,
                    n=int(stats["n"]),
                    mae=json_safe(stats["mae"]),
                )
            )
    return buckets


def build_scorecard(
    results: pl.DataFrame | None = None,
    decomposition: dict[str, list] | None = None,
) -> ScorecardFile:
    """The whole file.

    Both inputs are injectable so tests can drive this over a handful of
    constructed rows; by default it runs the real walk-forward, which is
    the same computation `python -m analytics evaluate` performs and takes
    about as long.
    """
    from backtest.report import component_decomposition_mae, minutes_head_metrics

    if results is None:
        from analytics.evaluate import run_comparison

        logger.info("running walk-forward comparison (this takes ~15s)")
        results = run_comparison()
    if decomposition is None:
        from analytics.evaluate import run_component_decomposition

        logger.info("running component decomposition (this takes ~11s)")
        decomposition = run_component_decomposition()

    components = component_decomposition_mae(
        decomposition["predicted_components"], decomposition["actual_components"]
    )
    minutes = minutes_head_metrics(
        decomposition["predicted_minutes_dist"], decomposition["actual_minutes"]
    )

    rows = build_rows(results)
    seasons = sorted(results["season"].unique().to_list())
    models = sorted(results["baseline"].unique().to_list())

    return ScorecardFile(
        header=build_header(
            rows=len(rows),
            # The last gameweek the backtest actually reaches, not 38: a
            # season the archive holds only part of would otherwise have
            # the file claim coverage it does not have.
            source_gameweek=int(results.filter(pl.col("season") == seasons[-1])["gw"].max()),
            normalization_basis="walk_forward_within_position",
        ),
        models=models,
        seasons=seasons,
        event_model=EVENT_MODEL,
        rows=rows,
        calibration=build_calibration(results),
        error_by_event=build_error_by_event(results),
        component_decomposition=[
            ComponentError(component=name, mae=json_safe(value))
            for name, value in sorted(components.items())
        ],
        minutes_head=MinutesHead(
            brier_blank=json_safe(minutes["brier_blank"]),
            brier_short=json_safe(minutes["brier_short"]),
            brier_full=json_safe(minutes["brier_full"]),
            mae_expected_minutes=json_safe(minutes["mae_expected_minutes"]),
            n=int(minutes["n"]),
        ),
    )
