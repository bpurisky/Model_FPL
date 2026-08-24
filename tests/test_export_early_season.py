"""Every export, against a season that has barely started.

The archive has three complete seasons in it, so every exporter is
developed and tested against 38 gameweeks of history and none of them is
naturally exercised at the shape they will actually meet first: one
gameweek, then two, then three. That shape is the *only* one the reader
sees during the opening month, which is when the tool is most used.

So this builds a synthetic current season from the real reference data —
real element ids, real clubs, real fixtures — appends it to the real
panel, and runs the whole export chain at 1, 2 and 3 gameweeks.

It is deliberately not asserting exact numbers. The synthetic stats are
made up and their values mean nothing; what is under test is that the
pipeline *reaches* the current season, that normalization produces real
z-scores rather than a column of nulls, and that the two thresholds in
`config/frontend.yaml` behave the way the surfaces assume they do.
"""

from __future__ import annotations

import collections

import polars as pl
import pytest

from squad.live import TRAIN_SCHEMA
from web.export.contract import CURRENT_SEASON

REFERENCE = "data/reference"
POSITION = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}


def _synthetic_actuals(gameweeks: int) -> pl.DataFrame:
    """Real players and clubs, plausible stats, for gw 1..N.

    Roughly two-thirds of each squad plays in any gameweek — the realistic
    shape, and the one that exercises the null paths that matter.
    """
    players = pl.read_parquet(f"{REFERENCE}/players.parquet")
    teams = pl.read_parquet(f"{REFERENCE}/teams.parquet")
    names = dict(zip(teams["id"].to_list(), teams["name"].to_list()))

    rows = []
    for gw in range(1, gameweeks + 1):
        for i, (element_id, team_id, kind) in enumerate(
            zip(players["id"].to_list(), players["team"].to_list(), players["element_type"].to_list())
        ):
            plays = (i + gw) % 3 != 0
            base = {
                name: (False if dtype == pl.Boolean else "" if dtype == pl.Utf8 else 0)
                for name, dtype in TRAIN_SCHEMA.items()
            }
            base.update(
                gw=gw,
                element_id=element_id,
                position=POSITION.get(kind, "MID"),
                team=names.get(team_id, "Unknown"),
                minutes=90 if plays else 0,
                total_points=(2 + (i % 6)) if plays else 0,
                goals_scored=1 if plays and i % 17 == 0 else 0,
                assists=1 if plays and i % 23 == 0 else 0,
                clean_sheets=1 if plays and i % 4 == 0 else 0,
                goals_conceded=(i % 3) if plays else 0,
                saves=(i % 5) if plays and kind == 1 else 0,
                bonus=(i % 4) if plays and i % 9 == 0 else 0,
                bps=(10 + i % 30) if plays else 0,
                expected_goals=0.02 * (i % 12) if plays else 0.0,
                expected_assists=0.01 * (i % 9) if plays else 0.0,
                expected_goals_conceded=0.05 * (i % 7) if plays else 0.0,
            )
            rows.append(base)

    return pl.DataFrame(rows, schema=TRAIN_SCHEMA).with_columns(
        pl.lit(CURRENT_SEASON).alias("season")
    )


@pytest.fixture(scope="module")
def early(request):
    """A panel with a synthetic current season appended, plus the store
    the projection path reads."""
    from web.export.current import enrich
    from web.export.panel import build_panel

    built: dict[int, tuple[pl.DataFrame, pl.DataFrame]] = {}

    def make(gameweeks: int):
        if gameweeks not in built:
            enriched = enrich(_synthetic_actuals(gameweeks))
            built[gameweeks] = (build_panel(current=enriched), enriched)
        return built[gameweeks]

    return make


@pytest.mark.parametrize("gameweeks", [1, 2, 3])
def test_the_panel_carries_the_current_season(early, gameweeks):
    panel, _ = early(gameweeks)
    current = panel.filter(pl.col("season") == CURRENT_SEASON)
    assert current.height > 0, "the current season never reached the panel"
    assert sorted(current["gw"].unique().to_list()) == list(range(1, gameweeks + 1))


@pytest.mark.parametrize("gameweeks", [1, 2, 3])
def test_normalization_produces_real_z_scores_from_the_first_gameweek(early, gameweeks):
    """The failure `web/export/current.py` exists to prevent, checked from
    the other end.

    A panel that reached the current season but normalized none of it
    looks complete and renders every z-score as null. The minutes floor
    scales with fixtures played precisely so this works in gameweek one.
    """
    panel, _ = early(gameweeks)
    current = panel.filter(pl.col("season") == CURRENT_SEASON)
    scored = current["xg_per90_z_pos"].drop_nulls().len()
    assert scored > 0, "no current-season row carries a z-score"
    # A floor that admitted almost nobody would be as bad as one that
    # admitted no one; half the squad is the rough shape the config aims at.
    assert scored > current.height * 0.25


@pytest.mark.parametrize("gameweeks", [1, 2, 3])
def test_players_and_board_follow_the_panel_to_the_current_season(early, gameweeks, monkeypatch):
    from web.export import players as players_mod
    from web.export.board import build_board

    panel, enriched = early(gameweeks)
    # `model_frame` reads the store directly, so it has to agree with the
    # panel or the projection path filters to a season it does not carry.
    monkeypatch.setattr(players_mod, "load_current_season", lambda: enriched)

    file = players_mod.build_players(panel=panel)
    assert file.season == CURRENT_SEASON
    assert file.gameweek == gameweeks
    assert file.projected_gameweek == gameweeks + 1

    # Every player gets a projection from gameweek one — Comparison's
    # decomposition and Explorer's projection columns both depend on it.
    projected = [p for p in file.players if p.projection is not None]
    assert len(projected) == len(file.players)
    assert all(p.projection.p_full is not None for p in projected), (
        "the minutes distribution is §5.4.4's one non-negotiable panel"
    )

    board = build_board(panel=panel)
    assert board.season == CURRENT_SEASON
    assert board.gameweek == gameweeks
    assert len(board.players) > 0


@pytest.mark.parametrize("gameweeks", [1, 2])
def test_every_classification_is_low_confidence_before_the_minimum(early, gameweeks):
    """§5.4.6's amber flag, and the surface renders it per card.

    Below `min_gameweeks` the board still ranks — level is knowable from
    one gameweek — but nothing it says should look settled.
    """
    from web.export.board import build_board

    panel, _ = early(gameweeks)
    board = build_board(panel=panel)
    assert gameweeks < board.min_gameweeks
    assert all(p.low_confidence for p in board.players), (
        "a classification resting on fewer gameweeks than the configured "
        "minimum must be flagged"
    )


def test_the_trend_buckets_do_not_exist_until_the_window_is_full(early):
    """Which is why the Model Board explains an empty `rising` rather than
    reporting it as no players matching a filter."""
    from web.export.board import build_board

    panel_one, _ = early(1)
    early_buckets = {p.bucket for p in build_board(panel=panel_one).players}
    assert "rising" not in early_buckets
    assert "declining" not in early_buckets
    # Level-based buckets are available immediately, which is the half of
    # the board that measured any edge at all.
    assert "optimal" in early_buckets

    panel_three, _ = early(3)
    board = build_board(panel=panel_three)
    assert board.trend_window == 3
    later = collections.Counter(p.bucket for p in board.players)
    assert later["rising"] > 0, "the trend window is full and nothing is rising"


@pytest.mark.parametrize("gameweeks", [1, 3])
def test_the_current_season_is_offered_and_marked_partial(early, gameweeks):
    """The Correlation Lab's season filter reads these summaries, and
    `partial` is what keeps a rho over one gameweek from looking like a
    rho over thirty-eight."""
    from web.export.observations import build_observations

    panel, _ = early(gameweeks)
    file = build_observations(panel=panel)
    summaries = {s.season: s for s in file.seasons}

    assert CURRENT_SEASON in summaries, "the current season is not offered at all"
    current = summaries[CURRENT_SEASON]
    assert current.partial is True
    assert current.gameweeks == gameweeks
    assert current.players > 0
    # And the completed seasons must not be dragged into looking partial.
    assert all(not s.partial for s in file.seasons if s.season != CURRENT_SEASON)


def test_correlations_include_the_current_season(early):
    from web.export.correlations import build_correlations

    panel, _ = early(3)
    file = build_correlations(panel=panel)
    assert CURRENT_SEASON in file.seasons
