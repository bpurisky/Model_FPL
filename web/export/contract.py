"""The schema of record for every exported file (§5.3.1).

These pydantic models are hand-written, and `web/app/src/data/schema.ts`
mirrors them in zod by hand as well. §5.12.2 requires a test asserting the
two agree; generating one from the other is permitted, silent divergence
is not. Until the TypeScript side exists, `contract_shape()` below is what
that test will compare against, so it is the one function here that other
code should not reach around.

Every exported file carries the same header. Two of its fields are
non-negotiable per §5.3.1 and both are about traceability rather than
convenience:

`model_git_sha` — every number on screen must lead back to the exact code
that produced it. The same argument as papertrade/freeze.py's provenance
block, applied to the export rather than to a freeze.

`normalization_basis` — a z-score is meaningless without the population it
was computed against, and §5.7.4 requires the UI to state that basis in a
tooltip. A basis recorded in the file is the only way the tooltip can be
truthful without re-deriving it in TypeScript, which §5.6 forbids.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Any, Literal, get_args, get_origin

from pydantic import BaseModel, ConfigDict

CONTRACT_VERSION = 1

# See `json_safe`. Enough to preserve every real distinction in these
# exports, few enough to absorb parallel-reduction jitter in the last bits
# of a double.
SIGNIFICANT_DIGITS = 12

Role = Literal["quantitative", "categorical", "ordinal", "temporal"]
Relevance = Literal["primary", "secondary", "context", "none"]
Grain = Literal["player", "player_gameweek", "fixture", "metric_pair", "model_gameweek"]
Source = Literal["fpl_api", "vaastav_archive", "derived", "model"]


class _Strict(BaseModel):
    """Extra fields are a contract violation, not something to tolerate.

    The opposite choice from collector/schemas.py's `_LenientModel`, and
    deliberately so: there the payload belongs to FPL and drift is news to
    be reported, here the payload belongs to this repo and drift is a bug
    on our own side of the wire.
    """

    model_config = ConfigDict(extra="forbid")


class Header(_Strict):
    """§5.3.1's header object, on every exported file."""

    contract_version: int = CONTRACT_VERSION
    generated_at: datetime
    source_gameweek: int | None
    scoring_config: str
    model_git_sha: str | None
    normalization_basis: str
    rows: int


class ColumnSpec(_Strict):
    """One entry in `columns.json` (§5.3.5).

    The registry is called out in the spec as "the highest-leverage file
    in the frontend", on the grounds that getting it wrong makes every
    downstream surface wrong in the same way. That is the reason for
    `extra="forbid"` above and for the required fields below: a column
    that reaches the UI without a definition or a `higher_is_better` is a
    column the UI will render confidently and wrongly.
    """

    key: str
    label: str
    role: Role
    unit: str | None
    format: str
    definition: str
    source: Source
    grain: Grain
    normalizable: bool
    normalized_key: str | None
    position_relevance: dict[Literal["GK", "DEF", "MID", "FWD"], Relevance]
    higher_is_better: bool | None
    available_from_season: str | None

    # --- deviation from §5.3.5's shape, recorded per §5.16 -------------
    # The spec's registry can say when a column starts and has no way to
    # say when one stops. That is not hypothetical: FPL publishes
    # influence/creativity/threat/ict_index as literal 0.0 for all 604
    # elements in 2026-27 while populating everything else, so the series
    # ends without the values ever going missing. Without this field the
    # only honest options are to drop three seasons of real ICT data or
    # to let a column of manufactured zeros render as measurement, and
    # §5.3.3 exists precisely to prevent the second.
    available_to_season: str | None = None

    def applies_to_season(self, season: str) -> bool:
        """Whether this column carries real measurement in `season`.

        Callers use this to write null rather than zero (§5.3.3). String
        comparison is safe because FPL season labels sort lexically in
        chronological order ("2023-24" < "2024-25").
        """
        if self.available_from_season and season < self.available_from_season:
            return False
        if self.available_to_season and season > self.available_to_season:
            return False
        return True


class ColumnsFile(_Strict):
    header: Header
    columns: list[ColumnSpec]


class CorrelationCell(_Strict):
    """One metric pair, within one position group (§5.3.2).

    `rho` and `p_value` are nullable and `n` is not. A pair can fail to
    produce a correlation — too few complete rows, or a metric with no
    spread in the group — and the honest report of that is a null rho
    beside the n that explains it, never a 0.0 that reads as "measured, no
    relationship" (§5.3.3).

    `n` is the count of rows where *both* metrics are present, not the
    size of the group. The two differ by a lot here: four metrics exist
    only from 2025-26, so their cells draw on a third of the pooled
    population.
    """

    group: str
    a: str
    b: str
    rho: float | None
    n: int
    p_value: float | None


class GroupSummary(_Strict):
    """One entry of the position filter (§5.4.1).

    `mixed_position` is the flag §5.7.5's caution copy hangs off. It is a
    property of the population rather than of any one cell, which is why
    it lives here and is not repeated 91 times.
    """

    key: str
    n_player_seasons: int
    mixed_position: bool


class CorrelationsFile(_Strict):
    """`correlations.json`.

    `min_n_cell` travels *in the file* rather than being read from
    `config/frontend.yaml` by the UI: the threshold that decides whether a
    cell is hatched has to be the one that was in force when these numbers
    were computed, and a browser reading today's config against a file
    built last week would hatch the wrong cells.
    """

    header: Header
    basis: str
    min_n_cell: int
    seasons: list[str]
    metrics: list[str]
    groups: list[GroupSummary]
    cells: list[CorrelationCell]


class PositionSpearman(_Strict):
    """Within-position rank correlation for one model, one slice.

    Carried per position rather than as a single pooled figure because
    pooling positions is the §5.7.1 distortion in its original form: a
    model that only knows forwards outscore defenders would post a
    flattering pooled rho while ranking nobody correctly inside their own
    group, which is the only ranking an FPL manager ever acts on.
    """

    position: str
    rho: float | None
    n: int
    p_value: float | None


class ScorecardRow(_Strict):
    """One model over one slice of the walk-forward results (§5.3.2).

    `season` and `gw` are both nullable and the nulls are structural, not
    missing data: `gw: null` is the season rollup and `season: null` is
    the pooled-everything row. The UI reads the grain it wants by
    filtering rather than by re-aggregating, which §5.6 forbids it doing
    anyway.

    `spearman_mean` is the unweighted mean across positions and carries no
    p-value on purpose — `report.spearman_within_position_significance`
    declines to invent one, because no sampling distribution describes the
    mean of four correlations.
    """

    model: str
    season: str | None
    gw: int | None
    n: int
    mae: float | None
    rmse: float | None
    spearman_mean: float | None
    spearman_by_position: list[PositionSpearman]


class CalibrationBin(_Strict):
    """One decile of predicted points against what actually happened.

    §5.4.7 wants the diagonal drawn: a well-calibrated model tracks it,
    and systematic departure is bias the pooled MAE hides.
    """

    model: str
    bin: int
    n: int
    mean_prediction: float | None
    mean_actual: float | None


class EventErrorBucket(_Strict):
    """Phase 1's error decomposition — MAE split by what kind of event
    occurred. Kept for the scalar baselines, which have no per-component
    prediction to compare against; the event model additionally gets the
    true decomposition below."""

    model: str
    bucket: str
    n: int
    mae: float | None


class ComponentError(_Strict):
    """The event model's true per-component decomposition: predicted point
    contribution against the realized one, per scoring bucket."""

    component: str
    mae: float | None


class MinutesHead(_Strict):
    """§4.4 evaluates the minutes head separately, on the grounds that
    §4.2 calls it the highest-leverage and most failure-prone component.
    Folding it into a pooled MAE would hide exactly the thing worth
    watching."""

    brier_blank: float | None
    brier_short: float | None
    brier_full: float | None
    mae_expected_minutes: float | None
    n: int


class ScorecardFile(_Strict):
    """`scorecard.json`.

    `component_decomposition` and `minutes_head` cover the event model
    alone — they compare predicted components against realized ones, and
    the three scalar baselines predict a single number with no components
    to decompose. `event_model` names which model that is rather than
    leaving the reader to infer it.
    """

    header: Header
    models: list[str]
    seasons: list[str]
    event_model: str
    rows: list[ScorecardRow]
    calibration: list[CalibrationBin]
    error_by_event: list[EventErrorBucket]
    component_decomposition: list[ComponentError]
    minutes_head: MinutesHead


class FixtureRow(_Strict):
    """One fixture, with both difficulty ratings (§5.3.2, §4.3).

    `difficulty_basis` is the field that keeps the two Elo figures from
    being read as one number. A played fixture reports the rating each
    club carried *into* it, which is what the model actually knew; an
    unplayed one reports the rating they hold today. Both are useful and
    they answer different questions, so the row says which it is rather
    than leaving a planning surface to assume.

    FPL's own ratings are integers 1-5 and fixed for the season; ours is
    continuous on the same scale so the two can share an axis.
    """

    fixture: int
    gw: int | None
    team_h: str
    team_a: str
    kickoff_time: datetime | None
    played: bool
    team_h_difficulty: int | None
    team_a_difficulty: int | None
    custom_difficulty_home: float | None
    custom_difficulty_away: float | None
    difficulty_basis: Literal["pre_match", "current_elo"]


class FixturesFile(_Strict):
    """`fixtures.json`.

    `elo_matches` and `elo_seeded_from` are provenance, not decoration.
    Elo computed over eight matches and Elo computed over three seasons
    produce the same-looking number on the same 1-5 scale, and only these
    two fields distinguish them — `data/historical/raw/` is a restorable
    cache rather than committed data, so a build that could not seed
    produces a rating that means much less. The UI must be able to say so.
    """

    header: Header
    season: str
    elo_matches: int
    elo_seeded_from: list[str]
    # Clubs with no match history in the archive, whose Elo is therefore
    # the initial rating rather than a measured one. That initial value is
    # the league mean, so an unseeded club is rated exactly average — which
    # systematically flatters newly promoted sides, and does it without
    # anything looking wrong. Named here so the UI can mark those fixtures
    # instead of rendering an assumption as a measurement.
    unseeded_teams: list[str]
    fixtures: list[FixtureRow]


class GoldenSample(_Strict):
    """The inputs one group's golden pairs were computed over (§5.6.1).

    `rows` is a dense matrix — one list per player-season, one entry per
    entry in `metrics`, aligned by position — rather than a list of named
    objects, because every metric shares the same rows and repeating
    fourteen keys per row would triple the file to say nothing new.

    Nulls are real and load-bearing: four metrics do not exist before
    2025-26, and a port that ranks a null column rather than dropping the
    pair reproduces a failure this project has already had.
    """

    group: str
    metrics: list[str]
    rows: list[list[float | None]]


class GoldenPair(_Strict):
    """One metric pair's Python-computed answer, over the sample above.

    `rho` is nullable for the same reason `CorrelationCell.rho` is — a
    metric with no spread in the sample has no correlation to report — and
    a port should return the same nothing rather than a zero.
    """

    group: str
    a: str
    b: str
    n: int
    rho: float | None


class GoldenSpearmanFile(_Strict):
    """`golden_spearman.json`.

    `tolerance` and `precision` travel in the file so the TypeScript test
    reads both from the fixture instead of keeping second copies that can
    drift from it. `precision` is not decoration: the goldens were
    computed *after* rounding to it, which is the only way a consumer
    reading these numbers can reproduce these answers to 1e-9.
    """

    header: Header
    method: str
    tolerance: float
    precision: int
    samples: list[GoldenSample]
    pairs: list[GoldenPair]


class PositionWeights(_Strict):
    """One position's composite profile (§5.4.6).

    Exported so the surface can render a "How this is scored" panel: the
    user must be able to read the model's opinion, not only receive its
    output. Negative weights are meaningful and must survive the contract
    — conceding is bad for a defender, and a scoring layer that cannot
    express that is not modelling football.
    """

    position: str
    weights: dict[str, float]


class BoardBucketAccuracy(_Strict):
    """What a bucket was actually worth, measured (§5.4.6, §5.4.7).

    "If the app is going to classify players as rising, it must report how
    often rising players subsequently outperformed." This is that report,
    and it travels with the classification rather than beside it, so a
    surface cannot show the bucket without being able to show the number.

    `lift` is the bucket's mean forward points minus everyone else's over
    the same gameweeks. It is negative for `rising`, and that is the
    finding rather than a bug.
    """

    bucket: str
    n: int
    # Which population `forward_points_other` covers. Optimal takes the
    # top quartile, so a momentum bucket compared against "everyone else"
    # would be scored against a pool the good players were already removed
    # from — it would read as negative however well momentum worked.
    comparison: str
    forward_points: float | None
    forward_points_other: float | None
    lift: float | None


class BoardPlayer(_Strict):
    """One card, or one row of the ranked list — the same record serves
    both surfaces (§5.4.6).

    `drivers` names the metrics that moved *this* player's composite,
    ranked by their contribution rather than by weight, so a card says
    something about him rather than reciting the profile.

    `low_confidence` is §5.4.6's amber flag. A classification below the
    gameweek floor still renders, because hiding it would teach the reader
    the player does not exist, but it renders marked.
    """

    element_id: int
    name: str
    team: str
    position: str
    composite: float | None
    percentile: float | None
    rank: int
    bucket: str
    drivers: list[str]
    gameweeks_seen: int
    low_confidence: bool


class BoardFile(_Strict):
    """`board.json`.

    Carries both surfaces over one composite: `players` is ordered by
    `rank` within position and is the ranked list, while `bucket` on each
    record is the three-bucket classification. They are one dataset
    because a player is one player — the alternative was two files that
    could disagree about him.
    """

    header: Header
    season: str
    gameweek: int
    trend_window: int
    min_gameweeks: int
    weights: list[PositionWeights]
    bucket_accuracy: list[BoardBucketAccuracy]
    players: list[BoardPlayer]


class PlayerMetric(_Strict):
    """One rate metric for one player, with the basis §5.7.4 needs.

    `z` and `percentile` are null for a player below the minutes floor —
    unknown, not average (§5.3.3).

    The population size lives in `PlayersFile.population` rather than
    here. It is a property of the position group, not of the player: every
    forward in the file shares one `n` for xG, so carrying it per player
    would repeat the same 64 numbers thirteen thousand times and invite
    the reader to think two players' z-scores rest on different samples.
    Same argument as `min_n_cell` in `correlations.json` and
    `mixed_position` in `GroupSummary`.
    """

    value: float | None
    z: float | None
    percentile: float | None


class PlayerProjection(_Strict):
    """The event model's expectation for the projected gameweek.

    `total` is the sum of `components` by construction — the exporter adds
    them rather than calling a second function, so the headline figure
    cannot disagree with the decomposition panel that breaks it down.

    The three minutes probabilities are the head §4.2 calls the
    highest-leverage and most failure-prone component, carried separately
    rather than folded into the total for exactly that reason.
    """

    components: dict[str, float | None]
    total: float | None
    p_blank: float | None
    p_short: float | None
    p_full: float | None


class PlayerRow(_Strict):
    """One element (§5.3.2).

    `price` is in tenths of a million and `selected` is a squad count, both
    as the archive records them. `selected` is null for current-season rows
    until the collector captures `total_players` — see
    `web/export/current.py`, which explains why a percentage must not be
    written into a count column.
    """

    element_id: int
    name: str
    team: str
    position: str
    price: int | None
    selected: int | None
    gameweeks: int
    actuals: dict[str, float | None]
    metrics: dict[str, PlayerMetric]
    projection: PlayerProjection | None


class PlayersFile(_Strict):
    """`players.json`.

    `projection_basis` distinguishes a real next-fixture forecast from a
    fixture-neutral statement about the player. For a completed season
    there is no next gameweek to have a fixture, so difficulty falls back
    to neutral and the projection stops being about a specific match —
    reading one as the other would be reading it wrong.
    """

    header: Header
    season: str
    gameweek: int
    projected_gameweek: int
    projection_basis: Literal["next_fixture", "fixture_neutral"]
    # position -> metric -> the size of the group every z-score in that
    # position was computed against (§5.7.4's tooltip needs it, and it is
    # a group property rather than a player one).
    population: dict[str, dict[str, int]]
    players: list[PlayerRow]


class SeasonSummary(_Strict):
    """What a season actually covers.

    `partial` is the field that keeps the season selector honest. A
    correlation over two gameweeks and one over thirty-eight look
    identical once they are both a rho, and the current season joins this
    file the moment it has a single recorded gameweek. Offering four
    seasons that look alike when one of them is two matches old would
    invite exactly the reading the rest of this contract works to prevent.
    """

    season: str
    gameweeks: int
    players: int
    partial: bool


class ObservationRow(_Strict):
    """One eligible player-season, with every matrix metric.

    `values` is positional, aligned to `ObservationsFile.metrics` — the
    same convention `GoldenSample.rows` uses, so a reader meets it once
    rather than twice. Nulls are real: four metrics do not exist before
    2025-26, and a client-side correlation must drop the incomplete pair
    rather than rank a column containing them.
    """

    season: str
    element_id: int
    name: str
    team: str
    position: str
    values: list[float | None]


class ObservationsFile(_Strict):
    """`observations.json` — the values behind the matrix (§5.6.1).

    Exists so the browser can answer a question precomputation cannot:
    Spearman does not compose, so rho over a chosen subset of seasons
    cannot be assembled from per-season matrices, and shipping one matrix
    per subset is 2^n. These rows answer every subset.

    The population is deliberately identical to the one `correlations.py`
    correlates — same eligibility floor, same minutes-weighted season
    rates — because the client-side matrix for "all seasons" sits next to
    the precomputed one and the two must agree.
    """

    header: Header
    basis: str
    seasons: list[SeasonSummary]
    metrics: list[str]
    rows: list[ObservationRow]


def build_header(
    *,
    rows: int,
    source_gameweek: int | None,
    normalization_basis: str,
    scoring_config: str = "scoring_2026_27.yaml",
    generated_at: datetime | None = None,
) -> Header:
    """The one place a header is constructed, so no exporter can omit a
    field by writing its own dict.

    `model_git_sha` is read here rather than passed in for the same reason
    papertrade/freeze.py reads it at freeze time: a sha supplied by the
    caller is a sha that can be wrong.
    """
    from papertrade.freeze import model_git_sha

    return Header(
        contract_version=CONTRACT_VERSION,
        generated_at=generated_at or datetime.now(timezone.utc),
        source_gameweek=source_gameweek,
        scoring_config=scoring_config,
        model_git_sha=model_git_sha(),
        normalization_basis=normalization_basis,
        rows=rows,
    )


def json_safe(value: float | None) -> float | None:
    """NaN and infinity, turned into the null they actually mean.

    Two separate reasons, and either alone would be enough. `float('nan')`
    serializes to a bare `NaN` token that `JSON.parse` rejects, so one
    degenerate cell takes down the whole surface rather than rendering as
    not-applicable. And NaN is not a measurement: the statistics in
    `backtest/report.py` return it for a degenerate input — an empty
    group, a series with no spread, fewer than three pairs — which is
    §5.3.3's null, never a zero.

    Lives here rather than in any one exporter because it is a property of
    the contract boundary: this is what may cross it.

    Values are also rounded to `SIGNIFICANT_DIGITS`, which is about
    reproducibility rather than tidiness. Polars aggregates groups in
    parallel, so floating-point summation order varies between runs:
    `scorecard.json`'s calibration means were observed differing in their
    last one to three digits across two builds of identical committed
    data. That makes a published number irreproducible — §5.3.1 wants
    every figure traceable to the code that produced it, which is empty if
    the same code produces a different figure — and it defeats the
    unchanged-file check in `__main__.write_json`, so a rebuild would
    commit noise forever.

    Significant digits, not decimal places. A p-value here can legitimately
    be 2.9e-119, and rounding that to twelve decimals would publish zero —
    a claim of certainty rather than a very small number. Twelve
    significant digits discards the last few bits of a double, which is
    where the non-determinism lives, and is four orders of magnitude finer
    than the tightest tolerance anything downstream asks for (§5.6.1's
    1e-9 on |rho| <= 1).
    """
    if value is None:
        return None
    value = float(value)
    if not math.isfinite(value):
        return None
    return float(f"{value:.{SIGNIFICANT_DIGITS}g}")


def contract_shape() -> dict[str, Any]:
    """A structural description of every model in this module, for the
    §5.12.2 test that asserts `schema.ts` agrees.

    Deliberately shallower than pydantic's own JSON schema: it compares
    field names, whether each is required, and a coarse type name. A
    literal JSON-schema diff would fail on pydantic-specific detail that
    zod has no way to express, which would make the test noise rather
    than a guard.
    """
    shape: dict[str, Any] = {}
    models = (
        Header, ColumnSpec, ColumnsFile,
        CorrelationCell, GroupSummary, CorrelationsFile,
        PositionSpearman, ScorecardRow, CalibrationBin, EventErrorBucket,
        ComponentError, MinutesHead, ScorecardFile,
        FixtureRow, FixturesFile,
        GoldenSample, GoldenPair, GoldenSpearmanFile,
        PositionWeights, BoardBucketAccuracy, BoardPlayer, BoardFile,
        PlayerMetric, PlayerProjection, PlayerRow, PlayersFile,
        SeasonSummary, ObservationRow, ObservationsFile,
    )
    for model in models:
        fields = {}
        for name, field in model.model_fields.items():
            fields[name] = {
                "required": field.is_required(),
                "type": _type_name(field.annotation),
            }
        shape[model.__name__] = fields
    return shape


def _type_name(annotation: Any) -> str:
    """A coarse, zod-expressible name for an annotation.

    `get_origin` rather than `__origin__`: PEP 604 unions (`float | None`)
    are `types.UnionType` and carry no `__origin__` at all, so reading the
    attribute directly sent every nullable field in this module — `unit`,
    `higher_is_better`, `source_gameweek`, `model_git_sha` and the rest —
    down the fallback path and described them all as a bare "union". The
    §5.12.2 test would then have accepted a `schema.ts` that got each of
    their inner types wrong, which is most of what that test is for.
    """
    origin = get_origin(annotation)
    if origin is not None:
        args = [a for a in get_args(annotation) if a is not type(None)]
        optional = len(args) < len(get_args(annotation))
        if origin is dict:
            base = "record"
        elif origin is list:
            base = "array"
        elif len(args) == 1:
            base = _type_name(args[0])
        else:
            base = "union"
        return f"{base}?" if optional else base
    if annotation is datetime:
        return "datetime"
    if annotation in (int, float, str, bool):
        return annotation.__name__
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        return annotation.__name__
    return "union"  # Literal[...] and friends


__all__ = [
    "CONTRACT_VERSION",
    "SIGNIFICANT_DIGITS",
    "ColumnSpec",
    "ColumnsFile",
    "BoardBucketAccuracy",
    "BoardFile",
    "BoardPlayer",
    "CalibrationBin",
    "ComponentError",
    "CorrelationCell",
    "CorrelationsFile",
    "EventErrorBucket",
    "FixtureRow",
    "FixturesFile",
    "GoldenPair",
    "GoldenSample",
    "GoldenSpearmanFile",
    "GroupSummary",
    "Header",
    "MinutesHead",
    "PlayerMetric",
    "PlayerProjection",
    "PlayerRow",
    "PlayersFile",
    "ObservationRow",
    "ObservationsFile",
    "PositionSpearman",
    "PositionWeights",
    "ScorecardFile",
    "ScorecardRow",
    "SeasonSummary",
    "json_safe",
    "build_header",
    "contract_shape",
]