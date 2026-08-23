"""The column registry (§5.3.5) — one entry per exported column.

The spec calls this "the most important new artifact in v2 and the
highest-leverage file in the frontend", because getting an entry wrong
makes every downstream surface wrong in the same way and none of them
will complain. Three fields carry most of that weight:

`role` drives mark inference (§5.4.2), and is authored here rather than
guessed in TypeScript.

`higher_is_better` orients the diverging scale. xGC is a stat to be low
on, and a heat map that colours it like xG is not merely unhelpful, it
tells the user the opposite of the truth.

`position_relevance` is §5.7 made machine-readable. The builder *dims*
columns marked `none` for the filtered position and never hides them:
hiding teaches the user the column does not exist, dimming teaches them
it does not matter here, and the second is the actual lesson.

The metric set answers §5.15 Q2 ("which ~15-20 metrics?"). Sixteen, in
four groups. Two exclusions are worth stating because they were in the
spec's own illustrative weight profiles:

`creativity_per90` and the rest of ICT are absent because FPL stopped
publishing them. Verified 2026-08-23, one gameweek into the season:
influence, creativity, threat and ict_index are all exactly 0.0 across
all 604 elements, in the same payload where 176 players have a recorded
start and 111 have xG. They are registered for the seasons where they
are real, and end via `available_to_season`.

`shots_in_box_per90` is absent because it exists in no source this
project has -- not the FPL API, not the vaastav archive -- and
`threat_per90` is not a substitute for it: same dead column, and even
alive it would triple-count signal `xg_per90` already carries directly.
"""

from __future__ import annotations

import polars as pl

from web.export.contract import ColumnSpec

# Seasons in which a stat family carries real measurement. Grounded in the
# committed panel rather than in the FPL rule history: verified 2026-08-23
# that tackles, recoveries, clearances_blocks_interceptions and
# defensive_contribution are *entirely* null for 2023-24 and 2024-25 and
# fully populated from 2025-26. That is four of the sixteen metrics, and
# §5.3.3's whole point is that those rows must render as not-applicable
# rather than as zero.
DEFENSIVE_ACTIONS_FROM = "2025-26"

# ICT was published through 2025-26 and is dead from 2026-27 (see module
# docstring). Registered so three seasons of real data stay usable.
ICT_UNTIL = "2025-26"

_ALL = ("GK", "DEF", "MID", "FWD")


def _relevance(gk: str, de: str, mi: str, fw: str) -> dict[str, str]:
    return dict(zip(_ALL, (gk, de, mi, fw)))


def _metric(
    key: str,
    label: str,
    definition: str,
    *,
    relevance: dict[str, str],
    higher_is_better: bool = True,
    source: str = "fpl_api",
    unit: str | None = "per90",
    fmt: str = ".2f",
    available_from: str | None = None,
    available_to: str | None = None,
) -> ColumnSpec:
    """A normalizable per-90 rate. Every one of these gets `_z_pos`,
    `_pct_pos` and `_n_pos` companions from normalize.py (§5.7.2)."""
    return ColumnSpec(
        key=key,
        label=label,
        role="quantitative",
        unit=unit,
        format=fmt,
        definition=definition,
        source=source,  # type: ignore[arg-type]
        grain="player_gameweek",
        normalizable=True,
        normalized_key=f"{key}_z_pos",
        position_relevance=relevance,  # type: ignore[arg-type]
        higher_is_better=higher_is_better,
        available_from_season=available_from,
        available_to_season=available_to,
    )


def _context(
    key: str,
    label: str,
    definition: str,
    *,
    role: str = "categorical",
    unit: str | None = None,
    fmt: str = "",
    grain: str = "player_gameweek",
) -> ColumnSpec:
    """Identity and fixture context: never normalized, never compared
    across players as a quantity."""
    return ColumnSpec(
        key=key,
        label=label,
        role=role,  # type: ignore[arg-type]
        unit=unit,
        format=fmt,
        definition=definition,
        source="fpl_api",
        grain=grain,  # type: ignore[arg-type]
        normalizable=False,
        normalized_key=None,
        position_relevance=_relevance("context", "context", "context", "context"),  # type: ignore[arg-type]
        higher_is_better=None,
        available_from_season=None,
    )


# --------------------------------------------------------------------------
# identity and fixture context
# --------------------------------------------------------------------------

_CONTEXT: list[ColumnSpec] = [
    _context("season", "Season", "The season this row belongs to, as FPL labels it."),
    _context("gw", "GW", "Gameweek number within the season.", role="ordinal", fmt="d"),
    _context("element_id", "Player ID", "FPL's element id. Stable within a season, not across seasons."),
    _context("name", "Player", "Player's FPL web name."),
    _context("team", "Team", "Club the player was registered to for this gameweek."),
    _context("position", "Position", "GK, DEF, MID or FWD. The group every normalized value is computed against."),
    _context("opponent_team", "Opponent", "Club faced in this fixture."),
    _context("was_home", "Home", "Whether the fixture was played at home."),
    _context("kickoff_time", "Kickoff", "Fixture kickoff, UTC.", role="temporal"),
    _context("n_fixtures", "Fixtures", "Fixtures the player's team played this gameweek. 2 in a double, 0 in a blank.", role="quantitative", fmt="d"),
]

# --------------------------------------------------------------------------
# volume — real quantities, but not rates, so not normalized as rates
# --------------------------------------------------------------------------

_VOLUME: list[ColumnSpec] = [
    _context("minutes", "Minutes", "Minutes played this gameweek.", role="quantitative", unit="minutes", fmt="d"),
    _context("total_points", "Points", "FPL points scored this gameweek under the season's own scoring rules.", role="quantitative", unit="points", fmt="d"),
    _context("value", "Price", "FPL price at the time of the gameweek, in tenths of a million.", role="quantitative", unit="tenths_m", fmt="d"),
    _context("selected", "Ownership", "Squads selecting this player at the time of the gameweek.", role="quantitative", unit="count", fmt="d"),
]

# --------------------------------------------------------------------------
# the sixteen metrics (§5.15 Q2)
# --------------------------------------------------------------------------

_ATTACKING: list[ColumnSpec] = [
    _metric("xg_per90", "xG per 90", "Expected goals per 90 minutes played.",
            relevance=_relevance("none", "secondary", "primary", "primary")),
    _metric("xa_per90", "xA per 90", "Expected assists per 90 minutes played.",
            relevance=_relevance("none", "secondary", "primary", "secondary")),
    _metric("xgi_per90", "xGI per 90", "Expected goal involvements (xG + xA) per 90 minutes played.",
            relevance=_relevance("none", "secondary", "primary", "primary")),
    _metric("goals_per90", "Goals per 90", "Goals scored per 90 minutes played.",
            relevance=_relevance("none", "secondary", "primary", "primary")),
    _metric("assists_per90", "Assists per 90", "Assists per 90 minutes played.",
            relevance=_relevance("none", "secondary", "primary", "secondary")),
    _metric("bonus_per90", "Bonus per 90", "FPL bonus points per 90 minutes played.",
            relevance=_relevance("secondary", "secondary", "secondary", "secondary")),
]

_DEFENSIVE: list[ColumnSpec] = [
    _metric("xgc_per90", "xGC per 90", "Expected goals conceded while on the pitch, per 90 minutes.",
            relevance=_relevance("primary", "primary", "context", "none"), higher_is_better=False),
    _metric("goals_conceded_per90", "Conceded per 90", "Goals conceded while on the pitch, per 90 minutes.",
            relevance=_relevance("primary", "primary", "context", "none"), higher_is_better=False),
    _metric("clean_sheet_prob", "Clean sheet %",
            "Modelled probability the player's team concedes no goals in the next fixture. "
            "Team-level by construction: FPL credits a clean sheet only at 60 minutes, so a "
            "player-level figure would embed minutes risk that minutes_reliability already carries.",
            relevance=_relevance("primary", "primary", "secondary", "none"),
            source="model", unit="probability", fmt=".0%"),
    _metric("saves_per90", "Saves per 90", "Saves per 90 minutes played. Meaningful for goalkeepers only.",
            relevance=_relevance("primary", "none", "none", "none")),
    _metric("defensive_contribution_per90", "Def. contribution per 90",
            "Gameweeks meeting the defensive-contribution threshold, per 90 minutes. "
            "The rule did not exist before 2025-26, so earlier seasons are null rather than zero.",
            relevance=_relevance("none", "primary", "secondary", "context"),
            available_from=DEFENSIVE_ACTIONS_FROM),
]

_INVOLVEMENT: list[ColumnSpec] = [
    _metric("bps_per90", "BPS per 90", "Bonus Points System score per 90 minutes played.",
            relevance=_relevance("secondary", "secondary", "secondary", "secondary")),
    _metric("tackles_per90", "Tackles per 90", "Tackles per 90 minutes played.",
            relevance=_relevance("none", "primary", "secondary", "context"),
            available_from=DEFENSIVE_ACTIONS_FROM),
    _metric("recoveries_per90", "Recoveries per 90", "Ball recoveries per 90 minutes played.",
            relevance=_relevance("none", "secondary", "primary", "context"),
            available_from=DEFENSIVE_ACTIONS_FROM),
    _metric("cbi_per90", "CBI per 90", "Clearances, blocks and interceptions per 90 minutes played.",
            relevance=_relevance("context", "primary", "secondary", "none"),
            available_from=DEFENSIVE_ACTIONS_FROM),
]

_AVAILABILITY: list[ColumnSpec] = [
    _metric("minutes_reliability", "Minutes reliability",
            "Empirical probability of a 60-plus-minute appearance over the model's trailing "
            "window. This is analytics/features.py's p_full, reused rather than redefined, so "
            "it inherits that head's published Brier score (0.1007 over 83,035 predictions).",
            relevance=_relevance("primary", "primary", "primary", "primary"),
            source="model", unit="probability", fmt=".0%"),
]

# --------------------------------------------------------------------------
# ICT — real through 2025-26, discontinued by FPL thereafter
# --------------------------------------------------------------------------

_ICT: list[ColumnSpec] = [
    _metric("influence_per90", "Influence per 90", "FPL's Influence index per 90 minutes. Discontinued by FPL from 2026-27.",
            relevance=_relevance("secondary", "secondary", "secondary", "secondary"), available_to=ICT_UNTIL),
    _metric("creativity_per90", "Creativity per 90", "FPL's Creativity index per 90 minutes. Discontinued by FPL from 2026-27.",
            relevance=_relevance("none", "secondary", "primary", "secondary"), available_to=ICT_UNTIL),
    _metric("threat_per90", "Threat per 90", "FPL's Threat index per 90 minutes. Discontinued by FPL from 2026-27.",
            relevance=_relevance("none", "secondary", "primary", "primary"), available_to=ICT_UNTIL),
]


REGISTRY: list[ColumnSpec] = [
    *_CONTEXT,
    *_VOLUME,
    *_ATTACKING,
    *_DEFENSIVE,
    *_INVOLVEMENT,
    *_AVAILABILITY,
    *_ICT,
]

# The sixteen that answer §5.15 Q2, in matrix order. ICT is registered but
# not in the matrix set: it is dead for the season the user is actually
# looking at, and a hero surface whose columns are all not-applicable is
# worse than a smaller hero.
MATRIX_METRICS: list[str] = [c.key for c in (*_ATTACKING, *_DEFENSIVE, *_INVOLVEMENT, *_AVAILABILITY)]


def by_key() -> dict[str, ColumnSpec]:
    return {c.key: c for c in REGISTRY}


def normalizable_keys() -> list[str]:
    return [c.key for c in REGISTRY if c.normalizable]


def companion_keys(key: str) -> tuple[str, str, str]:
    """The three columns §5.7.2 requires alongside every normalizable
    metric. `_n_pos` is not optional -- a z-score over eleven qualifying
    goalkeepers carries different weight from one over two hundred
    midfielders, and the UI cannot say so unless the export says so."""
    return f"{key}_z_pos", f"{key}_pct_pos", f"{key}_n_pos"


def per90_expr(source_column: str, key: str) -> pl.Expr:
    """A per-90 rate from a raw panel column.

    Null when the player did not play, never zero: a player who did not
    appear has no rate, and averaging a manufactured 0.0 into a positional
    mean would drag it toward whoever sat on the bench (§5.3.3).
    """
    return (
        pl.when(pl.col("minutes") > 0)
        .then(pl.col(source_column) / pl.col("minutes") * 90)
        .otherwise(None)
        .alias(key)
    )


# Which raw panel column each per-90 metric derives from. Kept beside the
# registry so the §5.11.1 completeness test can walk both together.
PER90_SOURCES: dict[str, str] = {
    "xg_per90": "expected_goals",
    "xa_per90": "expected_assists",
    "xgi_per90": "expected_goal_involvements",
    "goals_per90": "goals_scored",
    "assists_per90": "assists",
    "bonus_per90": "bonus",
    "xgc_per90": "expected_goals_conceded",
    "goals_conceded_per90": "goals_conceded",
    "saves_per90": "saves",
    "defensive_contribution_per90": "defensive_contribution",
    "bps_per90": "bps",
    "tackles_per90": "tackles",
    "recoveries_per90": "recoveries",
    "cbi_per90": "clearances_blocks_interceptions",
    "influence_per90": "influence",
    "creativity_per90": "creativity",
    "threat_per90": "threat",
}
