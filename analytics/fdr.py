"""Custom fixture difficulty from team Elo (§4.3), reported alongside FPL's
own static rating so the difference is visible.

§3.1 suggests sourcing dynamic Elo from `olbauday/FPL-Core-Insights`
"rather than building one." That repo turned out to only have genuinely
per-gameweek dynamic Elo for one of the three backtest seasons (2025-26) —
2023-24 and 2024-25 only carry a single static snapshot each, which would
leak a team's *final* season strength into early-gameweek predictions if
used as a point-in-time feature. Given §0.3's point-in-time discipline is
non-negotiable and the alternative (Elo computed here, incrementally, from
match results already in data/historical/raw/{season}/fixtures.csv) is
fully self-contained, uniform across all three seasons, and trivially
point-in-time-correct by construction, that's what this module does
instead. Documented deviation, not an oversight.
"""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl

INITIAL_ELO = 1500.0
K_FACTOR = 20.0
HOME_ADVANTAGE = 60.0  # Elo points added to the home side's expectation


@dataclass(frozen=True)
class EloRatings:
    """Pre-match Elo for every fixture, in chronological order — the Elo a
    team carried *into* each match, never updated with that match's own
    result. This is what makes it safe to use as a predictive feature."""

    per_fixture: pl.DataFrame  # event, team_h, team_a, elo_home_pre, elo_away_pre
    final: dict[int, float]  # team_id -> Elo after the last processed match


def _expected_score(elo_a: float, elo_b: float) -> float:
    return 1.0 / (1.0 + 10 ** ((elo_b - elo_a) / 400.0))


def compute_elo_ratings(
    matches: pl.DataFrame,
    k: float = K_FACTOR,
    home_advantage: float = HOME_ADVANTAGE,
    initial_elo: float = INITIAL_ELO,
) -> EloRatings:
    """`matches` needs columns: fixture, event, team_h, team_a,
    team_h_score, team_a_score, kickoff_time (see
    backtest.backfill.load_match_results). Processed in kickoff_time order
    so each fixture's recorded pre-match Elo reflects only strictly earlier
    results within this DataFrame.
    """
    ordered = matches.sort("kickoff_time")
    ratings: dict[int, float] = {}
    rows = []

    for row in ordered.iter_rows(named=True):
        team_h, team_a = row["team_h"], row["team_a"]
        elo_h = ratings.get(team_h, initial_elo)
        elo_a = ratings.get(team_a, initial_elo)
        rows.append(
            {
                "fixture": row["fixture"],
                "event": row["event"],
                "team_h": team_h,
                "team_a": team_a,
                "elo_home_pre": elo_h,
                "elo_away_pre": elo_a,
            }
        )

        expected_h = _expected_score(elo_h + home_advantage, elo_a)
        if row["team_h_score"] > row["team_a_score"]:
            actual_h = 1.0
        elif row["team_h_score"] < row["team_a_score"]:
            actual_h = 0.0
        else:
            actual_h = 0.5

        ratings[team_h] = elo_h + k * (actual_h - expected_h)
        ratings[team_a] = elo_a + k * ((1 - actual_h) - (1 - expected_h))

    per_fixture = (
        pl.DataFrame(rows)
        if rows
        else pl.DataFrame(
            schema={
                "fixture": pl.Int64,
                "event": pl.Int64,
                "team_h": pl.Int64,
                "team_a": pl.Int64,
                "elo_home_pre": pl.Float64,
                "elo_away_pre": pl.Float64,
            }
        )
    )
    return EloRatings(per_fixture=per_fixture, final=ratings)


def custom_difficulty(elo_team: float, elo_opponent: float, is_home: bool, home_advantage: float = HOME_ADVANTAGE) -> float:
    """A continuous 1(easiest)-5(hardest) difficulty for `elo_team`'s
    fixture, on the same scale as FPL's own rating for direct comparison.

    difficulty = 3 (neutral) shifted by the Elo gap, net of home advantage,
    at a rate of 1 difficulty point per 200 Elo of opponent superiority.
    Simple and monotonic by design — the point of this feature is that it's
    dynamic and continuous where FPL's own is static and coarse (§4.3), not
    that the conversion formula itself is optimized.
    """
    adjustment = home_advantage if is_home else -home_advantage
    elo_gap = (elo_opponent) - (elo_team + adjustment)
    difficulty = 3.0 + elo_gap / 200.0
    return max(1.0, min(5.0, difficulty))


def team_gameweek_difficulty(matches: pl.DataFrame, teams: pl.DataFrame, **elo_kwargs) -> pl.DataFrame:
    """One row per (team name, gw): the custom difficulty of that team's
    fixture(s) that gameweek, mean-aggregated across a double gameweek's
    multiple fixtures — matching exactly how FPL's own opponent_difficulty
    is aggregated in backtest.backfill.normalize_season, so this joins
    directly onto data/historical/*.parquet on (team, gw).

    `teams` is backtest.backfill.load_teams's frame (id, name) — Elo is
    keyed by FPL's numeric team id; the historical dataset's `team` column
    is a readable name, and this bridges the two.
    """
    elo = compute_elo_ratings(matches, **elo_kwargs)
    if elo.per_fixture.height == 0:
        return pl.DataFrame(schema={"team": pl.Utf8, "gw": pl.Int64, "custom_difficulty": pl.Float64})

    home_rows = elo.per_fixture.select(
        pl.col("team_h").alias("team_id"),
        pl.col("event").alias("gw"),
        pl.struct(["elo_home_pre", "elo_away_pre"])
        .map_elements(lambda s: custom_difficulty(s["elo_home_pre"], s["elo_away_pre"], is_home=True), return_dtype=pl.Float64)
        .alias("difficulty"),
    )
    away_rows = elo.per_fixture.select(
        pl.col("team_a").alias("team_id"),
        pl.col("event").alias("gw"),
        pl.struct(["elo_home_pre", "elo_away_pre"])
        .map_elements(lambda s: custom_difficulty(s["elo_away_pre"], s["elo_home_pre"], is_home=False), return_dtype=pl.Float64)
        .alias("difficulty"),
    )
    per_team_gw = pl.concat([home_rows, away_rows]).group_by(["team_id", "gw"]).agg(pl.col("difficulty").mean().alias("custom_difficulty"))
    return per_team_gw.join(teams, left_on="team_id", right_on="id", how="left").select(
        pl.col("name").alias("team"), "gw", "custom_difficulty"
    )


def upcoming_team_difficulty(elo_final: dict[int, float], fixtures: pl.DataFrame, teams: pl.DataFrame, **kwargs) -> pl.DataFrame:
    """§4.3/§5's live counterpart to `team_gameweek_difficulty`: that
    function derives difficulty from `EloRatings.per_fixture`, which only
    has an entry for a fixture already present in `compute_elo_ratings`'s
    input — i.e. one with a final score, so Elo can update on it. A squad
    recommendation needs difficulty for fixtures that haven't been played
    yet, where the pairing (team_h/team_a/event) is known well in advance
    but there is no score to process.

    Takes `elo_final` (an `EloRatings.final` dict — each team's rating as of
    the most recent *completed* match, from any Elo history the caller has
    already built) and applies it directly to `fixtures` (event, team_h,
    team_a — no scores required) via the same `custom_difficulty` used
    everywhere else, so an upcoming fixture is scored on the strength each
    side actually carries into it, not a stale end-of-last-season snapshot.
    A team with no Elo yet (e.g. newly promoted, §3.2) gets `initial_elo`.
    """
    initial = kwargs.get("initial_elo", INITIAL_ELO)
    home_advantage = kwargs.get("home_advantage", HOME_ADVANTAGE)
    if fixtures.height == 0:
        return pl.DataFrame(schema={"team": pl.Utf8, "gw": pl.Int64, "custom_difficulty": pl.Float64})

    def _rows(home_col: str, away_col: str, is_home: bool) -> pl.DataFrame:
        return fixtures.select(
            pl.col(home_col).alias("team_id"),
            pl.col("event").alias("gw"),
            pl.struct([home_col, away_col])
            .map_elements(
                lambda s: custom_difficulty(
                    elo_final.get(s[home_col], initial), elo_final.get(s[away_col], initial), is_home, home_advantage
                ),
                return_dtype=pl.Float64,
            )
            .alias("difficulty"),
        )

    per_team_gw = pl.concat([_rows("team_h", "team_a", True), _rows("team_a", "team_h", False)]).group_by(
        ["team_id", "gw"]
    ).agg(pl.col("difficulty").mean().alias("custom_difficulty"))
    return per_team_gw.join(teams, left_on="team_id", right_on="id", how="left").select(
        pl.col("name").alias("team"), "gw", "custom_difficulty"
    )


def build_fdr_comparison(elo_ratings: EloRatings, fpl_difficulty: pl.DataFrame) -> pl.DataFrame:
    """Side by side: our Elo-based difficulty and FPL's own published
    rating for the same fixtures (§4.3 — "report both ... so the
    difference is visible"). `fpl_difficulty` is
    backtest.backfill._load_fixture_difficulty's frame (fixture,
    team_h_difficulty, team_a_difficulty); joined on `fixture`.
    """
    df = elo_ratings.per_fixture.with_columns(
        pl.struct(["elo_home_pre", "elo_away_pre"])
        .map_elements(lambda s: custom_difficulty(s["elo_home_pre"], s["elo_away_pre"], is_home=True), return_dtype=pl.Float64)
        .alias("custom_difficulty_home"),
        pl.struct(["elo_home_pre", "elo_away_pre"])
        .map_elements(lambda s: custom_difficulty(s["elo_away_pre"], s["elo_home_pre"], is_home=False), return_dtype=pl.Float64)
        .alias("custom_difficulty_away"),
    )
    return df.join(fpl_difficulty, on="fixture", how="left")
