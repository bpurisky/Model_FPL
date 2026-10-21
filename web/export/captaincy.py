"""`captaincy.json` — the model's captaincy table, scored in public (§5.16 D18).

**This crosses §5.0.2, on one condition.** §5.0.2 keeps surfaces descriptive
and picks nothing; a captaincy table names the player the model would
captain. It ships because every pick it makes is also frozen before the
deadline, in the paper trade's own freeze (`papertrade/freeze.py`, the
`captaincy` block), and scored after it the way the launch gate scores
everything else: the pick's real points (doubled) against the highest-owned
player's and against the best of the model's own top five, plus the Brier
score of every frozen P(haul). `track_record` is that score, so the table
and the evidence it has earned are one file and cannot be read apart.

**Where the track record lives, and why.** It is in this file rather than in
`papertrade.json`: the paper trade file is the launch gate's contract and
D14 describes it as reporting a frozen shadow team, and a captaincy score
is neither a gate criterion nor part of the shadow team. The scoring code is
`papertrade.evaluate.evaluate_captaincy`, which the paper trade CLI also
prints, so the two cannot disagree. Freezes written before the field
existed are listed as skipped, never edited.

**The table is today's model, the record is each week's frozen model.** The
candidates are rebuilt on every export from the current data and may differ
slightly from what the next freeze records (injury flags move, and the
freeze reads FPL's own ownership and availability at that time). The record
only ever reads what was frozen.

The candidates are the same model as `projections.json` and `players.json`
(`web.export.projections.project_horizon`, first projected gameweek only);
P(haul) is `analytics.haul` (see its docstring for the method, what it
ignores, and its measured calibration). Committed data only, no live call:
ownership is the collector's latest snapshotted `selected_by_percent`.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path

import polars as pl

from analytics.captaincy import TOP_N, rank_candidates
from analytics.carryover import previous_season, prior_history
from analytics.deltas import state_as_of
from analytics.haul import HAUL_THRESHOLD, load_bonus_model
from analytics.scoreline import strengths_before
from analytics.scoring import load_scoring_config
from papertrade.actuals import load_actuals
from papertrade.evaluate import evaluate_captaincy
from papertrade.freeze import FREEZES_DIR, latest_frozen_gw
from web.export.contract import (
    CaptaincyCalibration,
    CaptaincyCandidate,
    CaptaincyFile,
    CaptaincyGw,
    CaptaincyOwned,
    CaptaincyPick,
    CaptaincySkipped,
    CaptaincyTrackRecord,
    ProjectionFixture,
    build_header,
    json_safe,
)
from web.export.correlations import PANEL_PATH
from web.export.normalize import load_frontend_config
from web.export.panel import HISTORICAL_DIR
from web.export.players import (
    DISTILLED_DIR,
    REFERENCE_FIXTURES,
    REFERENCE_TEAMS,
    SCORING_CONFIG,
    _availability,
    _season_fixtures,
    latest_gameweek,
    model_frame,
    target_roster,
)
from web.export.projections import project_horizon, team_fixtures

logger = logging.getLogger("web.export.captaincy")

METHOD = (
    "Top 20 by event-model xP for the next gameweek. P(haul) is the exact points distribution implied by "
    "the model's own heads (minutes, Poisson goals and assists, clean sheet, bonus shaped by past seasons); "
    "see analytics/haul.py."
)


def _owned(raw: dict | None) -> CaptaincyOwned | None:
    if raw is None:
        return None
    return CaptaincyOwned(
        element_id=raw["element_id"], name=raw.get("name"), ownership=json_safe(raw.get("ownership_pct"))
    )


def build_track_record(freezes_dir: Path, actuals: pl.DataFrame) -> CaptaincyTrackRecord:
    """Every scored frozen pick. A gameweek is a candidate for scoring when
    it has actuals and a freeze at or before the latest one on disk."""
    evaluated = sorted(set(actuals["gw"].to_list())) if actuals.height else []
    latest = latest_frozen_gw(freezes_dir)
    candidate_gws = [gw for gw in evaluated if gw <= (latest or 0)]
    result = evaluate_captaincy(candidate_gws, freezes_dir=freezes_dir, actuals=actuals)
    summary, cal = result["summary"], result["summary"]["calibration"]

    gameweeks = [
        CaptaincyGw(
            gw=g["gw"],
            pick=CaptaincyPick(
                element_id=g["pick"]["element_id"], name=g["pick"].get("name"), team=g["pick"].get("team"),
                xp=json_safe(g["pick"]["xp"]), p_haul=json_safe(g["pick"]["p_haul"]),
            ),
            pick_points=g["pick_points"],
            pick_points_x2=g["pick_points_x2"],
            pick_hauled=g["pick_hauled"],
            most_owned=_owned(g["most_owned"]),
            most_owned_points=g["most_owned_points"],
            most_owned_points_x2=g["most_owned_points_x2"],
            best_of_shortlist=CaptaincyOwned(
                element_id=g["best_of_shortlist"]["element_id"], name=g["best_of_shortlist"].get("name"), ownership=None
            ),
            best_of_shortlist_points=g["best_of_shortlist_points"],
            best_of_shortlist_points_x2=g["best_of_shortlist_points_x2"],
            shortlist_size=g["shortlist_size"],
        )
        for g in result["per_gw"]
    ]
    return CaptaincyTrackRecord(
        n_gameweeks=summary["n_gameweeks"],
        mean_pick_points_x2=json_safe(summary["mean_pick_points_x2"]),
        mean_most_owned_points_x2=json_safe(summary["mean_most_owned_points_x2"]),
        mean_best_of_shortlist_points_x2=json_safe(summary["mean_best_of_shortlist_points_x2"]),
        pick_haul_rate=json_safe(summary["pick_haul_rate"]),
        calibration=CaptaincyCalibration(
            n=cal["n"], brier=json_safe(cal["brier"]), brier_climatology=json_safe(cal["brier_climatology"]),
            mean_p_haul=json_safe(cal["mean_p_haul"]), haul_rate=json_safe(cal["haul_rate"]),
        ),
        gameweeks=gameweeks,
        # A gameweek with no freeze at all (before the paper trade began) is not
        # a skipped pick, it never existed; only freezes that cannot be scored are listed.
        skipped=[CaptaincySkipped(gw=s["gw"], reason=s["reason"]) for s in result["skipped"] if s["reason"] != "no freeze"],
    )


def build_captaincy(
    panel: pl.DataFrame | None = None,
    panel_path: Path = PANEL_PATH,
    historical_dir: Path = HISTORICAL_DIR,
    as_of: datetime | None = None,
    freezes_dir: Path = FREEZES_DIR,
    actuals: pl.DataFrame | None = None,
    distilled_dir: Path = DISTILLED_DIR,
) -> CaptaincyFile:
    """The next gameweek's top candidates and the frozen picks' record.
    `as_of` is when injury flags and ownership are read, and the header's
    `generated_at`."""
    as_of = as_of or datetime.now(timezone.utc)
    if panel is None:
        if not panel_path.exists():
            raise FileNotFoundError(
                f"{panel_path} not found — run `python -m web.export panel` first. "
                "It is a build artifact and §5.3.4 does not commit it."
            )
        panel = pl.read_parquet(panel_path)

    frame = model_frame(historical_dir)
    season, played = latest_gameweek(panel)
    if season not in set(frame["season"].unique().to_list()):
        raise ValueError(
            f"the panel's latest season {season} is not in the model frame "
            f"{sorted(frame['season'].unique().to_list())}; rebuild the panel."
        )
    train = frame.filter((pl.col("season") == season) & (pl.col("gw") <= played))
    roster = target_roster(frame, season, played)
    fixtures = _season_fixtures(season)
    last = int(fixtures["gw"].max()) if fixtures is not None and fixtures.height else None
    target_gw = played + 1

    candidates: list[CaptaincyCandidate] = []
    if last is not None and target_gw <= last:
        prev = HISTORICAL_DIR / f"{previous_season(season)}.parquet"
        strengths = strengths_before(train, fixtures, pl.read_parquet(prev) if prev.exists() else None, target_gw)
        config = load_scoring_config(Path(SCORING_CONFIG[season]))
        projected = project_horizon(
            train, roster, config, fixtures, strengths, [target_gw],
            prior=prior_history(season, roster), availability=_availability(season, as_of),
        )[target_gw]

        identity = {
            int(r["element_id"]): r
            for r in panel.filter((pl.col("season") == season) & (pl.col("gw") == played))
            .select("element_id", "name", "team").to_dicts()
        }
        state = state_as_of(distilled_dir, as_of)
        ownership = (
            {
                int(e): float(p)
                for e, p in zip(state["element_id"].to_list(), state["selected_by_percent"].to_list())
                if p is not None
            }
            if state.height
            else {}
        )
        meta = {
            eid: {"name": info["name"], "team": info["team"], "ownership_pct": ownership.get(eid)}
            for eid, info in identity.items()
        }
        ranked = rank_candidates(projected, config, meta, TOP_N, bonus_model=load_bonus_model(season))
        fixtures_by_team = {
            t.team: next(g.fixtures for g in t.gameweeks if g.gw == target_gw)
            for t in team_fixtures(pl.read_parquet(REFERENCE_FIXTURES), pl.read_parquet(REFERENCE_TEAMS), [target_gw])
        }
        for row in ranked:
            candidates.append(
                CaptaincyCandidate(
                    element_id=row["element_id"],
                    name=row["name"] or str(row["element_id"]),
                    team=row["team"],
                    position=row["position"],
                    opponents=fixtures_by_team.get(row["team"], []),
                    xp=json_safe(row["xp"]),
                    p_haul=json_safe(row["p_haul"]),
                    p_low=json_safe(row["p_low"]),
                    ownership=json_safe(row["ownership_pct"]),
                )
            )

    actuals = load_actuals() if actuals is None else actuals
    track_record = build_track_record(freezes_dir, actuals)
    logger.info(
        "captaincy: %d candidates for %s gw%d, %d frozen pick(s) scored",
        len(candidates), season, target_gw, track_record.n_gameweeks,
    )
    return CaptaincyFile(
        header=build_header(
            rows=len(candidates),
            source_gameweek=played,
            normalization_basis=load_frontend_config()["normalization"]["basis"],
            generated_at=as_of,
        ),
        season=season,
        gameweek=target_gw,
        haul_threshold=HAUL_THRESHOLD,
        method=METHOD,
        candidates=candidates,
        track_record=track_record,
    )
