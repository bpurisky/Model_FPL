"""`board.json` — the Model Board (§5.4.6).

Two surfaces over one composite, because they answer different questions
and only one of them has measurable edge.

**The ranking** is the composite score itself, ordered within position.
This is where the evidence is: players in the top quartile of a
short-window composite went on to score 3.736 points per gameweek over
the next three against 3.065 for everyone else, a lift of +0.671 measured
over 8,753 player-gameweeks.

**The buckets** are §5.4.6's Optimal / Rising / Declining. Optimal is the
top of the ranking and inherits its evidence. Declining is momentum, and
composite momentum does not work — see below. Rising is now a gain in role
among players near the top, which is mostly a level call. They ship because the
spec asks for them and because §5.4.6 requires the board to publish its
own hit rate, which is exactly the mechanism for saying so on screen.
Shipping them silently, or dropping them silently, would both be worse
than shipping them with the number attached.

**What was measured, twice, against two different definitions.** A fitted
trend slope on the underlying metrics gives rho 0.01-0.02 at every window
from 3 to 10 and every horizon from 1 to 8. A *consistent* rise — the
composite up in each of three consecutive gameweeks, which is a much
stronger condition than a positive slope — gives a forward-points lift of
**-0.092**: it predicts slightly *worse*. Combining it with the level
filter makes that filter worse (+0.650 against +0.671) while cutting the
population six-fold. Declining behaves the same way in reverse: a
monotone fall lifts +0.061, the wrong direction.

**And a shorter lens is the wrong direction too.** Top-quartile composite
lifts +0.682 on a 3-game window, +0.712 on 6, and +0.842 on 10. What the
ranking measures is quality, not form, and it measures it better the
longer it looks.

**Rising was redefined on 2026-09-23**, because the monotone rule was
worse than useless and skewed toward weak players. It could only fire
outside the top quartile, since Optimal wins collisions. It took any rise
however small. And low-minute players, whose per-90s swing most, produced
the most accidental runs. Nine alternatives were backtested over 17,370
non-optimal player-gameweeks (2023-24 to 2025-26), scored two ways: raw
lift against other non-optimal players, and *level-matched* lift against
players in the same position and percentile decile. Only the second shows
whether the trend adds anything the level does not already say:

    rule                                        raw     level-matched
    composite up every GW (old Rising)        -0.083   -0.274
      ... + upper half                         +0.218   -0.304
    large rise for that player (t >= 1.5)      +0.113   +0.029
    reliability +0.10, 60'+ every GW           +0.340   +0.098
    upper half, no trend at all                +0.678      -

Every rule that includes the level floor looks good raw and collapses
once level is matched, except one. **Rising is now: upper half of the
position, `minutes_reliability` up by 0.10 across the window, and 60+
minutes in every gameweek of it** — a player near the top whose role is
growing. It is the only definition whose trend beat same-level peers, and
only modestly and not every season (-0.06, +0.05, +0.31). Most of its raw
lift is the level floor. Treat it as "near the top and earning more
minutes", not as the model seeing form before the points arrive.

**Measured on what actually ships**, with each bucket compared inside the
pool it is drawn from — Optimal takes the top quartile, so scoring a
momentum bucket against "everyone else" would score it against a pool the
good players were already removed from:

    optimal     n=6242   +0.724   vs all classified players
    rising      n=854    +0.574   vs other non-optimal players
    declining   n=3708   -0.176   vs other non-optimal players

Declining keeps the monotone rule. A flagged player really does score
less than their non-optimal peers, and for a *warning* that is the right
sign.

`bucket_accuracy` is therefore not decoration and not defensive
documentation. It is the finding, carried in the file, so the surface
cannot present a bucket as insight without also showing what it is worth.

**Weights.** `config/frontend.yaml:board.position_weights`, fitted rather
than adopted from §5.4.6's illustrative profiles, which name three columns
that do not exist. The derivation and the two rules that shape it are
documented there. They are exported here because §5.4.6 requires them
rendered on screen: the user must be able to read the model's opinion,
not only receive its output.

**Which season.** The latest one present in the panel. Right now that is
the most recent completed season; the moment `data/current_season/` holds
a gameweek the panel carries 2026-27 and this follows it automatically.
The file names the season and gameweek it describes so the surface can
say which, rather than implying that last season's form is this week's.
"""

from __future__ import annotations

import logging
from pathlib import Path

import polars as pl

from web.export.contract import (
    BoardBucketAccuracy,
    BoardFile,
    BoardPlayer,
    PositionWeights,
    build_header,
    json_safe,
)
from web.export.correlations import PANEL_PATH, POSITIONS
from web.export.normalize import load_frontend_config

logger = logging.getLogger("web.export.board")

BUCKETS = ("optimal", "rising", "declining", "neutral")


def composite_expr(weights: dict[str, float], available: set[str]) -> pl.Expr:
    """The weighted sum of within-position z-scores (§5.4.6).

    Computed over the z-scores rather than the raw metrics because the
    weights are only comparable on a common scale — a weight of 0.2 on
    xGI and 0.1 on BPS means nothing if one runs 0-1 and the other 0-40.

    A player missing a metric contributes nothing for it rather than
    dragging the sum toward zero: `sum_horizontal` skips nulls, and the
    divisor below is the weight actually applied, so a defender with no
    defensive-contribution history is scored on what is known about him
    instead of being penalised for the gap (§5.3.3).
    """
    terms, magnitudes = [], []
    for metric, weight in weights.items():
        column = f"{metric}_z_pos"
        if column not in available:
            continue
        terms.append(pl.col(column).fill_null(0.0) * weight)
        magnitudes.append(pl.col(column).is_not_null().cast(pl.Float64) * abs(weight))
    if not terms:
        return pl.lit(None, dtype=pl.Float64)
    applied = pl.sum_horizontal(magnitudes)
    return pl.when(applied > 0).then(pl.sum_horizontal(terms) / applied).otherwise(None)


def latest_gameweek(panel: pl.DataFrame) -> tuple[str, int]:
    season = sorted(panel["season"].unique().to_list())[-1]
    return season, int(panel.filter(pl.col("season") == season)["gw"].max())


def with_composite(panel: pl.DataFrame, weights: dict[str, dict[str, float]]) -> pl.DataFrame:
    """Per-row composite, computed per position with that position's own
    profile. Never across positions — §5.4.6 is explicit, and §5.7.1 is
    the reason: a defender will not post a forward's xG."""
    available = set(panel.columns)
    frames = []
    for position in POSITIONS:
        subset = panel.filter(pl.col("position") == position)
        if not subset.height:
            continue
        frames.append(
            subset.with_columns(composite_expr(weights.get(position, {}), available).alias("composite"))
        )
    return pl.concat(frames) if frames else panel.with_columns(
        pl.lit(None, dtype=pl.Float64).alias("composite")
    )


def with_momentum(
    scored: pl.DataFrame,
    window: int,
    *,
    reliability_gain: float,
    min_minutes: int,
) -> pl.DataFrame:
    """The two trend flags, and the short-window mean.

    **Declining** is the composite falling in every gameweek of the window.
    Monotone rather than a fitted slope, because a slope can be negative
    while the series zig-zags.

    **Rising** is a gain in role: `minutes_reliability` up by at least
    `reliability_gain` across the window while the player gets at least
    `min_minutes` in every gameweek of it. The mirror of the declining
    rule was the original definition, and it measured worse than the
    players it was picked from. See the module docstring. The
    upper-half level floor is applied in `classify`, where the percentile
    exists.

    A frame without `minutes` or `minutes_reliability` gets no Rising
    flags rather than an error: the rule cannot be evaluated, which is
    different from a player failing it.
    """
    group = ["season", "element_id"]
    scored = scored.sort(["season", "element_id", "gw"])

    lags = [pl.col("composite").shift(i).over(group).alias(f"_c{i}") for i in range(1, window)]
    scored = scored.with_columns(
        *lags,
        pl.col("composite").rolling_mean(window, min_samples=window).over(group).alias("composite_window"),
        pl.col("gw").cum_count().over(group).alias("gameweeks_seen"),
    )

    declining = pl.lit(True)
    for i in range(1, window):
        earlier = pl.col(f"_c{i}")
        later = pl.col("composite") if i == 1 else pl.col(f"_c{i - 1}")
        declining = declining & (later < earlier)

    if {"minutes", "minutes_reliability"} <= set(scored.columns):
        gain = pl.col("minutes_reliability") - pl.col("minutes_reliability").shift(window - 1).over(group)
        playing = pl.col("minutes").rolling_min(window, min_samples=window).over(group) >= min_minutes
        rising = (gain >= reliability_gain) & playing
    else:
        rising = pl.lit(False)

    return scored.with_columns(
        rising.fill_null(False).alias("is_rising"),
        declining.fill_null(False).alias("is_declining"),
    ).drop([f"_c{i}" for i in range(1, window)])


def classify(
    scored: pl.DataFrame, optimal_quantile: float, rising_min_percentile: float
) -> pl.DataFrame:
    """One bucket per player, within position.

    Optimal wins over the momentum buckets where they collide: it is the
    classification with measured edge, and a card can only say one thing.
    Rising also requires the player to sit at `rising_min_percentile` or
    above. That makes Rising the band just below Optimal, where it
    has something to rise into.
    """
    peers = ["season", "gw", "position"]
    ranked = scored.drop_nulls("composite").with_columns(
        # Against this gameweek's peers, not against every player-gameweek
        # in the archive. A board is a statement about who to look at now,
        # so rank 1 must mean the best goalkeeper this week rather than the
        # best goalkeeper-gameweek of the last three seasons.
        pl.col("composite").quantile(optimal_quantile).over(peers).alias("_cutoff"),
        (pl.col("composite").rank("average").over(peers) - 1)
        .truediv(pl.max_horizontal(pl.len().over(peers) - 1, pl.lit(1)))
        .alias("percentile"),
        pl.col("composite").rank("ordinal", descending=True).over(peers).cast(pl.Int64).alias("rank"),
    )
    return ranked.with_columns(
        pl.when(pl.col("composite") >= pl.col("_cutoff"))
        .then(pl.lit("optimal"))
        .when(pl.col("is_rising") & (pl.col("percentile") >= rising_min_percentile))
        .then(pl.lit("rising"))
        .when(pl.col("is_declining"))
        .then(pl.lit("declining"))
        .otherwise(pl.lit("neutral"))
        .alias("bucket")
    ).drop("_cutoff")


def driving_metrics(row: dict, weights: dict[str, float], limit: int = 3) -> list[str]:
    """The metrics that actually moved this player's composite (§5.4.6:
    "the two or three metrics that drove the classification, named").

    Ranked by each term's contribution — weight times the player's own
    z-score — rather than by weight alone, so a card names what is true of
    *this* player rather than reciting the profile.
    """
    contributions = []
    for metric, weight in weights.items():
        z = row.get(f"{metric}_z_pos")
        if z is None:
            continue
        contributions.append((abs(weight * z), metric))
    contributions.sort(reverse=True)
    return [metric for _, metric in contributions[:limit]]


def bucket_accuracy(scored: pl.DataFrame) -> list[BoardBucketAccuracy]:
    """What each bucket was actually worth, over the whole archive.

    §5.4.6's own requirement — "if the app is going to classify players as
    rising, it must report how often rising players subsequently
    outperformed" — computed rather than asserted. `lift` is the bucket's
    mean forward points minus everyone else's, in the same gameweeks.
    """
    group = ["season", "element_id"]
    forward = scored.sort(["season", "element_id", "gw"]).with_columns(
        pl.mean_horizontal(
            [pl.col("total_points").shift(-i).over(group) for i in (1, 2, 3)]
        ).alias("forward_points")
    ).drop_nulls(["forward_points", "bucket"])

    rows: list[BoardBucketAccuracy] = []
    for bucket in BUCKETS:
        # Optimal takes the top quartile, so comparing a momentum bucket
        # against "everyone else" would score it against a pool the good
        # players have already been removed from — it would read as
        # negative however well momentum worked. Each momentum bucket is
        # therefore compared inside the pool it is actually drawn from.
        if bucket == "optimal":
            pool, comparison = forward, "all classified players"
        else:
            pool, comparison = (
                forward.filter(pl.col("bucket") != "optimal"),
                "other non-optimal players",
            )
        inside = pool.filter(pl.col("bucket") == bucket)
        outside = pool.filter(pl.col("bucket") != bucket)
        if not inside.height or not outside.height:
            continue
        mean_in = float(inside["forward_points"].mean())
        mean_out = float(outside["forward_points"].mean())
        rows.append(
            BoardBucketAccuracy(
                bucket=bucket,
                n=inside.height,
                comparison=comparison,
                forward_points=json_safe(mean_in),
                forward_points_other=json_safe(mean_out),
                lift=json_safe(mean_in - mean_out),
            )
        )
    return rows


def build_board(
    panel: pl.DataFrame | None = None,
    panel_path: Path = PANEL_PATH,
    config: dict | None = None,
) -> BoardFile:
    """The ranking, the buckets, and what each bucket is worth."""
    if panel is None:
        if not panel_path.exists():
            raise FileNotFoundError(
                f"{panel_path} not found — run `python -m web.export panel` first. "
                "It is a build artifact and §5.3.4 does not commit it."
            )
        panel = pl.read_parquet(panel_path)

    config = config or load_frontend_config()
    settings = config["board"]
    weights = settings["position_weights"]

    rising = settings["rising"]
    scored = classify(
        with_momentum(
            with_composite(panel, weights),
            settings["trend_window"],
            reliability_gain=rising["reliability_gain"],
            min_minutes=rising["min_minutes"],
        ),
        settings["optimal_quantile"],
        rising["min_percentile"],
    )

    # Accuracy is measured over every season in the panel; the board itself
    # describes only the latest. A hit rate computed on the same handful of
    # gameweeks it is describing would be a number about noise.
    accuracy = bucket_accuracy(scored)

    season, gameweek = latest_gameweek(panel)
    current = scored.filter((pl.col("season") == season) & (pl.col("gw") == gameweek))

    players: list[BoardPlayer] = []
    for row in current.sort(["position", "rank"]).iter_rows(named=True):
        players.append(
            BoardPlayer(
                element_id=int(row["element_id"]),
                name=row["name"],
                team=row["team"],
                position=row["position"],
                composite=json_safe(row["composite"]),
                percentile=json_safe(row["percentile"]),
                rank=int(row["rank"]),
                bucket=row["bucket"],
                drivers=driving_metrics(row, weights.get(row["position"], {})),
                gameweeks_seen=int(row["gameweeks_seen"] or 0),
                # §5.4.6's amber flag. Below the floor the classification
                # still renders — hiding it would teach the reader the
                # player does not exist — but it renders marked.
                low_confidence=int(row["gameweeks_seen"] or 0) < settings["min_gameweeks"],
            )
        )

    logger.info(
        "board: %d players at %s gw%d (%s)",
        len(players), season, gameweek,
        ", ".join(f"{b}={sum(1 for p in players if p.bucket == b)}" for b in BUCKETS),
    )

    return BoardFile(
        header=build_header(
            rows=len(players),
            source_gameweek=gameweek,
            normalization_basis=config["normalization"]["basis"],
        ),
        season=season,
        gameweek=gameweek,
        trend_window=settings["trend_window"],
        min_gameweeks=settings["min_gameweeks"],
        weights=[
            PositionWeights(position=position, weights=weights[position])
            for position in POSITIONS
            if position in weights
        ],
        bucket_accuracy=accuracy,
        players=players,
    )
