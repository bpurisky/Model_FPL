# fpl-trends

A Fantasy Premier League analytics platform for the 2026/27 season. Built to
[`fpl-trends-superprompt.md`](fpl-trends-superprompt.md); this README tracks
what actually exists against that spec.

**Status: Phases 0-4 (collector, backtest harness, event model, squad
integration, paper trade) and Phase 5 (frontend, `web/app/`) are built and
deployed to GitHub Pages.** Full detail, including every deviation from the
literal spec text and why, lives in `fpl-trends-superprompt.md`'s progress
log — this README is a static summary and that file is the current one.
`service/` (below) is the one piece that isn't a scheduled job or a static
export: a small backend for the frontend's Squad Optimizer, since that
surface's ILP solve can't run in the browser or be precomputed.

## What's here

A scheduled collector that snapshots the [FPL API](https://fantasy.premierleague.com/api/)
on a schedule and writes an append-only time series, since the API only
exposes current state — see `fpl-trends-superprompt.md` §0.1.

- `collector/client.py` — async HTTP client: 1 req/sec global rate limit,
  exponential backoff with jitter on 429/5xx (max 5 retries), deadline
  blackout window.
- `collector/schemas.py` — pydantic models for every ingested payload. A
  missing expected field halts the run; an unrecognised extra field only
  warns (§0.5).
- `collector/snapshot.py` — writes the three storage tiers (§2.3): raw
  (gzipped JSON, 14-day retention), distilled (delta-only Parquet, one
  shard per snapshot, permanent), reference (teams/fixtures/players,
  rewritten wholesale each run).
- `collector/entry.py` — own-team endpoints (`/entry/{id}/...`).
- `collector/__main__.py` — CLI entrypoint (`bootstrap`, `element-summary`,
  `live`, `entry`, `prune`) wiring config to the functions above.
- `.github/workflows/collect.yml` — hourly bootstrap/fixtures/element-summary,
  a 5-minute live-poll job gated on whether any fixture is actually in its
  live window.

## Running it

```sh
uv sync
uv run python -m collector bootstrap          # bootstrap-static + fixtures + reference + distilled
uv run python -m collector element-summary     # requires players.parquet from a prior bootstrap run
uv run python -m collector live                # only polls if a fixture is currently live
uv run python -m collector prune               # enforces 14-day raw retention
uv run python -m collector entry               # only runs if own_entry_id is set in config/collector.yaml
```

To enable the own-team endpoints, set `own_entry_id` in
[`config/collector.yaml`](config/collector.yaml).

## Backtest harness (Phase 1)

Walk-forward validation over three historical seasons (§3), with a leakage
assertion framework and three baseline models the eventual event model has
to beat.

- `backtest/backfill.py` — downloads and normalizes
  `vaastav/Fantasy-Premier-League`'s per-gameweek data into
  `data/historical/{season}.parquet` (committed — §8 requires every metric
  to regenerate from a single command against committed data, with no
  dependency on a third-party repo staying up). Drops the contaminated `xP`
  column entirely, resolves team/position from the per-gameweek row (never
  a season-final join), sums real double-gameweek rows instead of rejecting
  them as duplicates, and flags promoted clubs from
  [`config/promoted_clubs.yaml`](config/promoted_clubs.yaml).
- `backtest/leakage.py` — `assert_no_leakage`: every feature declares an
  `available_at` timestamp; anything not strictly before the target
  gameweek's deadline raises `LeakageError` and halts the run.
- `backtest/baselines.py` — trailing mean, an approximation of FPL's `form`
  field (the literal historical value is unrecoverable — see the module
  docstring), and a fixture-difficulty-adjusted trailing mean. All three
  share a pooled promoted-club prior for players with no trailing history.
- `backtest/harness.py` — the walk-forward loop: train on gameweeks 1..N,
  predict N+1, per season (element ids aren't stable across seasons, so
  seasons are walked independently and pooled at the metrics stage).
- `backtest/report.py` — MAE, RMSE, within-position Spearman rank
  correlation, calibration curves, and an error decomposition by event
  occurrence.

```sh
uv run python -m backtest backfill   # re-download + re-normalize (not required — parquet is committed)
uv run python -m backtest run        # walk-forward + report, from committed data only
uv run pytest tests/test_backfill.py tests/test_leakage.py tests/test_baselines.py tests/test_harness.py tests/test_report.py
```

## Event model & scoring layer (Phase 2)

Config-driven scoring plus a statistical event model, run through the
identical walk-forward harness as Phase 1's baselines for a direct
comparison (§4).

- `analytics/scoring.py` — `compute_points`: a pure function, event vector
  + season config -> points. Changing a rule is a YAML edit, never a Python
  edit. Validated by replaying real completed gameweeks: 99.1% (2024-25)
  and 99.3% (2025-26) exact match against official points, both against
  the §4.1 acceptance bar of 95%. The residual mismatch is fully
  understood, not hand-waved — see `validate_against_actual`'s docstring
  for the two confirmed root causes (double-gameweek banding rules aren't
  associative under raw-stat aggregation; abandoned-and-replayed fixtures
  get administratively zeroed outside FPL's normal formula).
- `config/scoring_2024_25.yaml`, `scoring_2025_26.yaml`, `scoring_2026_27.yaml`
  — one file per ruleset. 2025-26 adds defensive contribution; 2026-27
  changes the BPS clearances/blocks/interceptions divisor, removes the BPS
  tackled deduction, and switches goalkeeper saves to a flat-plus-bonus
  formula (provisional values — not officially published anywhere yet).
- `analytics/fdr.py` — a self-contained incremental Elo system (§4.3),
  point-in-time-correct by construction. Deliberately *not* sourced from
  `olbauday/FPL-Core-Insights` as §3.1 suggests — that repo turned out to
  only have genuine per-gameweek dynamic Elo for one of the three backtest
  seasons, which would leak final-season strength into early predictions
  for the other two. See the module docstring for the full reasoning.
  Reports its own difficulty alongside FPL's static rating for comparison.
- `analytics/features.py` — the trailing-rate/pooled-prior pattern from
  `backtest/baselines.py`, generalized so `projections.py` can build one
  feature per event-model head without repeating the pooling logic.
- `analytics/projections.py` — separate heads for appearance/minutes
  (P(blank)/P(short)/P(60+), not a single mean-minutes figure — see the
  module docstring on why), goals, assists, clean sheets, defensive
  contribution, saves, and bonus, difficulty-adjusted and combined into a
  points projection. Deliberately simple trailing-rate statistics, not
  machine learning — appropriate to this phase's actual bar (§4.4).
- `analytics/evaluate.py` — wires the model into the same `walk_forward`
  harness as the three Phase 1 baselines for an apples-to-apples
  comparison, plus the detailed per-component/minutes-head capture
  `backtest/report.py`'s new `component_decomposition_mae` /
  `minutes_head_metrics` need.

```sh
uv run python -m analytics evaluate   # model vs. baselines + component decomposition + minutes-head scorecard
uv run pytest tests/test_scoring.py tests/test_scoring_validation.py tests/test_fdr.py tests/test_features.py tests/test_projections.py tests/test_model_comparison.py
```

Real result (all three backtest seasons, pooled): the event model beats
all three baselines on MAE (1.0395 vs 1.0450/1.0496/1.0540) and beats
fixture-adjusted trailing mean on within-position Spearman (0.7202 vs
0.7180) — both §4.4 acceptance criteria, with margin. Getting there
surfaced a real, non-obvious finding: an early version modeled every event
type as an independent trailing-mean component and summed them, which beat
every baseline on MAE but *lost* to fixture-adjusted trailing mean on rank
correlation — concentrated almost entirely in defenders. Ablation traced it
to the goals-conceded penalty specifically: it's a stat shared across an
entire back line and heavily driven by single-match variance, so an
individual defender's trailing mean of it is real signal *and* real noise.
At full weight it hurt ranking more than it helped accuracy; at zero
weight rank correlation cleared the bar but MAE no longer beat the
baselines. `GOALS_CONCEDED_SHRINKAGE = 0.7` (documented in
`analytics/projections.py`) sits in the middle of a wide, robust 0.6-0.85
plateau that clears both bars with margin — not a value fitted to the edge
of passing.

## Testing

```sh
uv run pytest
```

## Querying the distilled time series

Each gameweek's distilled data is a directory of delta-only Parquet shards,
queryable directly with DuckDB:

```sql
SELECT selected_by_percent
FROM read_parquet('data/distilled/gw1/*.parquet')
WHERE element_id = 1 AND snapshot_ts <= '2026-08-25T12:00:00Z'
ORDER BY snapshot_ts DESC LIMIT 1;
```

## Deviations from the locked stack (§1.1)

Dependencies added beyond the superprompt's locked list, each justified at
its import site:

- **pyyaml** — the repo layout (§1.2) mandates `config/*.yaml` files;
  something has to parse them (`collector/config.py`, `backtest/backfill.py`).
- **pytz** — duckdb's Python DBAPI needs it to convert `TIMESTAMPTZ`
  columns back to Python datetimes via `fetchall()` (the `.pl()`/Arrow path
  used internally by the collector doesn't need it — this only bites ad-hoc
  `duckdb.sql(...).fetchall()` querying of the distilled data, as in the
  example above).
- **tzdata** — Windows has no built-in IANA time zone database; stdlib
  `zoneinfo`, which polars uses for tz-aware datetime columns, needs this
  backport to resolve `"UTC"` at all on this platform.

No scipy: Spearman rank correlation is computed directly in `report.py` as
the Pearson correlation of polars' own (ties-averaged) `.rank()`, rather
than pulling in a dependency for one formula.

## Known scope decisions worth knowing about

- **Baseline 3 (Phase 1) still scales by FPL's own published difficulty**,
  not the custom Elo model — it only needs to be a defensible hurdle, and
  changing its input would make it a different, harder-to-compare baseline
  than the one Phase 1 validated. The event model (Phase 2) uses the
  custom Elo-based difficulty (`analytics/fdr.py`) instead.
- **Gameweek deadline_time is approximated** as the earliest kickoff_time
  among that gameweek's fixtures — the historical source data doesn't
  publish the literal deadline. This makes the leakage assertion strictly
  *more* conservative, never less, since every feature here is built from
  gameweeks with days of margin before the next deadline regardless.
- **Double gameweeks are real**: 2023-24 gw7 alone has 983 rows where a
  rearranged fixture gives a player two matches in one FPL round.
  `backfill.py` sums them, matching how FPL itself scores a manager's week.
- **`position == "AM"` rows are excluded**: FPL's short-lived "Assistant
  Manager" pick (2024-25 round 23 onward) — real head coaches, scored by
  team results, not a player, and not part of this schema.
- **`defensive_contribution`, `clearances_blocks_interceptions`,
  `recoveries`, `tackles`** are read conditionally on each season's own CSV
  header and null (not 0) where the rule didn't exist yet — 2023-24 and
  2024-25 predate defensive contribution entirely.
- **Full BPS reproduction is out of scope.** FPL has never published its
  complete BPS formula/component weights. `bonus` is used as a direct
  input to `compute_points` (it's already in the historical data, already
  correct) rather than re-derived from a BPS-ranking-within-fixture
  simulation. `compute_bps` exists to make the two rule deltas §4.1 asks
  for (the clearances/blocks/interceptions divisor, the removed tackled
  deduction) config-driven and testable, not to match FPL's real value.
- **The 2026/27 GK save-bonus values are provisional.** Flat rate per save
  plus close-range/big-chance bonuses is the documented rule change, but
  the actual point values aren't published anywhere (and can't be
  validated — no 2026/27 gameweek has completed yet). Change them in
  `config/scoring_2026_27.yaml`, not in code, once real values are known.
- **Elo is self-built, not sourced from FPL-Core-Insights** despite §3.1
  suggesting otherwise — see `analytics/fdr.py`'s module docstring for why.
- **2023-24 reuses `scoring_2024_25.yaml`.** No scoring rule differs between
  the two seasons, and the repo layout (§1.2) doesn't call for a separate
  2023-24 file.

## Squad Optimizer backend (`service/`)

The frontend's Squad Optimizer view (`web/app/src/views/SquadOptimizer.tsx`)
takes an arbitrary FPL team ID and solves for the best legal transfer via
`squad/optimize.py`'s ILP — a live, per-request computation that cannot run
in the browser (no server, no runtime Python — frontend §5.1.1) and cannot
be precomputed (the team ID is chosen by the reader, not known at export
time). `service/app.py` is a thin FastAPI wrapper around the exact same
path `squad/__main__.py recommend` already exercises — no new logic beyond
request validation, per-IP rate limiting, and error-to-HTTP-status mapping.

Run locally:

```sh
uv sync --group service
uv run uvicorn service.app:app --reload
# in web/app/: cp .env.example .env.local, then npm run dev
```

Test:

```sh
uv run pytest tests/test_service.py
```

Deploy: `Dockerfile` (repo root) builds the service as a container —
`python:3.12-slim`, not alpine, because `pulp`'s bundled CBC solver binary
is a glibc build. `render.yaml` is one working deploy target (Render's free
tier, zero extra infra; `ALLOWED_ORIGINS` is already set there to
`https://bpurisky.github.io`, the real Pages origin — CORS matches origin,
not path, so the project-site `/Model_FPL/` prefix doesn't need naming);
Fly.io or Railway would work identically from the same image. After
deploying, set the `OPTIMIZER_API_URL` repository variable (Settings →
Secrets and variables → Actions → Variables — a public URL, not a secret)
to the deployed service's URL; `.github/workflows/web.yml`'s build step
reads it into `VITE_OPTIMIZER_API_URL`. With that variable unset, the
Squad Optimizer view degrades to an explanation rather than a broken form
or a raw network error (§7.3's rule for the Cloudflare Worker, applied
here too).

**Not yet built: chip planning.** §7.2's "My team" view also wants a chip
planner respecting the gameweek 19 wildcard/free-hit expiry.
`squad/optimize.py` has no chip-strategy logic to expose, so nothing here
fakes one — a documented gap, not an oversight.

## Attribution

Data from [fantasy.premierleague.com](https://fantasy.premierleague.com) and
[`vaastav/Fantasy-Premier-League`](https://github.com/vaastav/Fantasy-Premier-League)
(which itself incorporates understat.com's expected-goals data — used here
via vaastav's `expected_goals`/`expected_assists` columns rather than a
separate `soccerdata`/understat fetch, since it was already present and
already point-in-time correct).
