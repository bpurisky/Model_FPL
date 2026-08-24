"""`shrinkage.json` — the goals-conceded plateau, measured (§5.4.7).

> "**The defender/goals-conceded finding gets a permanent annotated
> panel.** The `GOALS_CONCEDED_SHRINKAGE = 0.7` plateau is the most
> interesting result the project has produced; a shrinkage-vs-metrics plot
> showing the 0.6-0.85 plateau belongs on screen, not only in a
> docstring."

Until now it was only in a docstring. `analytics/projections.py` records
the endpoints of the ablation in prose — at full weight the goals-conceded
term passes the MAE bar but pulls DEF Spearman below
`fixture_adjusted_trailing_mean`'s; at zero weight Spearman clears but MAE
no longer beats the baselines — but the sweep between them was never
kept, so nothing could draw the curve.

This keeps it. For each shrinkage value it re-runs the walk-forward for
the event model and records the two metrics the two bars are about, so
the panel shows *why* 0.7 rather than asserting it.

**The bars are computed here too, once.** A plateau is only a plateau
relative to something, and the something is the baselines: the best
baseline MAE, and `fixture_adjusted_trailing_mean`'s DEF Spearman. Both
are independent of shrinkage — the baselines do not use the term — so
they are measured on a single pass and drawn as reference lines.

**Cost.** One walk-forward per shrinkage value, event model only, plus
one pass for the baselines. About fifteen seconds each, so a ten-value
sweep is two and a half minutes. It is a build step and it is not wired
into `all` for that reason: nothing about this file changes when a
gameweek lands, because it describes the model rather than the season.
Run it when the model changes.

**Why the module constant grew a parameter.** `web/export/scorecard.py`
said this "needs `project_points` to accept the shrinkage as a parameter
rather than reading the module constant. That is a change to published
model code and belongs in its own deliberate step, not smuggled into an
exporter." This is that step, and the default is unchanged — every caller
that does not name a shrinkage gets 0.7, so the published model is
exactly the model every other number in the repo was measured on.
"""

from __future__ import annotations

import functools
import logging
from pathlib import Path

import polars as pl

from analytics.projections import GOALS_CONCEDED_SHRINKAGE, project_points
from web.export.contract import ShrinkageFile, ShrinkagePoint, build_header, json_safe

logger = logging.getLogger("web.export.shrinkage")

# The range the docstring's plateau claim covers, sampled finely enough
# through 0.6-0.85 to show it is flat rather than merely to touch its
# ends. Both endpoints are included because the argument for 0.7 is
# precisely that neither endpoint works.
SWEEP = (0.0, 0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 1.0)

# The position the finding is about. The ablation's whole content is that
# the damage at full weight is "entirely concentrated in DEF".
FOCUS_POSITION = "DEF"


def _season_frames() -> dict[str, pl.DataFrame]:
    from analytics.evaluate import NORMALIZED_DIR, SEASON_SCORING_CONFIG

    return {
        season: pl.read_parquet(NORMALIZED_DIR / f"{season}.parquet")
        for season in SEASON_SCORING_CONFIG
    }


def _event_model_at(season: str, shrinkage: float):
    """The event model with one term reweighted, and nothing else moved."""
    from analytics.evaluate import build_difficulty_table, SEASON_SCORING_CONFIG
    from analytics.scoring import load_scoring_config

    config = load_scoring_config(Path(SEASON_SCORING_CONFIG[season]))
    difficulty_table = build_difficulty_table(season)
    return functools.partial(
        project_points,
        config=config,
        difficulty_table=difficulty_table,
        goals_conceded_shrinkage=shrinkage,
    )


def _metrics(results: pl.DataFrame, model: str) -> tuple[float, float, float, int]:
    """MAE, mean within-position Spearman, DEF Spearman, and n."""
    from backtest.report import mae, spearman_within_position

    subset = results.filter(pl.col("baseline") == model)
    by_position = spearman_within_position(subset)
    return (
        mae(subset),
        by_position.get("mean", float("nan")),
        by_position.get(FOCUS_POSITION, float("nan")),
        subset.height,
    )


def build_shrinkage(
    sweep: tuple[float, ...] = SWEEP,
    season_dfs: dict[str, pl.DataFrame] | None = None,
) -> ShrinkageFile:
    """Re-run the walk-forward across `sweep` and keep both metrics."""
    from backtest.harness import BASELINES, walk_forward

    season_dfs = season_dfs if season_dfs is not None else _season_frames()

    # The baselines once. They do not use the goals-conceded term, so
    # re-running them per shrinkage value would burn nine walk-forwards to
    # reproduce the same two numbers.
    logger.info("baselines (once, for the reference bars)")
    baseline_batches = [
        walk_forward(df, season, BASELINES) for season, df in season_dfs.items()
    ]
    baselines = pl.concat([b for b in baseline_batches if b.height > 0])

    best_mae = float("inf")
    best_mae_model = ""
    for model in sorted(baselines["baseline"].unique().to_list()):
        model_mae, _, _, _ = _metrics(baselines, model)
        if model_mae < best_mae:
            best_mae, best_mae_model = model_mae, model
    _, fixture_adjusted_mean, fixture_adjusted_def, _ = _metrics(
        baselines, "fixture_adjusted_trailing_mean"
    )

    points: list[ShrinkagePoint] = []
    for shrinkage in sweep:
        logger.info("shrinkage %.2f", shrinkage)
        batches = [
            walk_forward(df, season, {"event_model": _event_model_at(season, shrinkage)})
            for season, df in season_dfs.items()
        ]
        results = pl.concat([b for b in batches if b.height > 0])
        point_mae, spearman_mean, spearman_def, n = _metrics(results, "event_model")
        points.append(
            ShrinkagePoint(
                shrinkage=shrinkage,
                mae=json_safe(point_mae),
                spearman_mean=json_safe(spearman_mean),
                spearman_focus=json_safe(spearman_def),
                n=n,
                # Both §4.4 bars, evaluated here so the surface renders a
                # verdict the export computed rather than one the browser
                # inferred from two numbers and a comparison (§5.6).
                beats_mae_bar=bool(point_mae < best_mae),
                beats_spearman_bar=bool(spearman_mean > fixture_adjusted_mean),
                beats_focus_bar=bool(spearman_def > fixture_adjusted_def),
            )
        )

    stated = [p.shrinkage for p in points if p.beats_mae_bar and p.beats_spearman_bar]
    focus = [p.shrinkage for p in points if p.beats_mae_bar and p.beats_focus_bar]
    logger.info("clearing §4.4 as stated (mean rho): %s", stated)
    logger.info("clearing it on DEF alone: %s", focus)

    return ShrinkageFile(
        header=build_header(
            rows=len(points),
            source_gameweek=None,  # this describes the model, not a gameweek
            normalization_basis="walk_forward_within_position",
        ),
        default=GOALS_CONCEDED_SHRINKAGE,
        focus_position=FOCUS_POSITION,
        baseline_mae=json_safe(best_mae),
        baseline_mae_model=best_mae_model,
        baseline_spearman_mean=json_safe(fixture_adjusted_mean),
        baseline_spearman_focus=json_safe(fixture_adjusted_def),
        baseline_spearman_model="fixture_adjusted_trailing_mean",
        points=points,
    )
