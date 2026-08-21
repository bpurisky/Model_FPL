# fpl-trends

A Fantasy Premier League analytics platform for the 2026/27 season. Built to
[`fpl-trends-superprompt.md`](fpl-trends-superprompt.md); this README tracks
what actually exists against that spec.

**Status: Phase 0 (collector) + Phase 1 (backtest harness) done.** Phases
2-5 (event model, squad optimizer, paper trade, frontend) are not built yet.

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

- **FPL's own historical difficulty ratings**, not a custom Elo model, back
  baseline 3's fixture adjustment. The Elo-based FDR replacement is §4.3,
  Phase 2 — baseline 3 only needs to be a defensible hurdle, not the answer.
- **Gameweek deadline_time is approximated** as the earliest kickoff_time
  among that gameweek's fixtures — the historical source data doesn't
  publish the literal deadline. This makes the leakage assertion strictly
  *more* conservative, never less, since every feature here is built from
  gameweeks with days of margin before the next deadline regardless.
- **Double gameweeks are real**: 2023-24 gw7 alone has 983 rows where a
  rearranged fixture gives a player two matches in one FPL round.
  `backfill.py` sums them, matching how FPL itself scores a manager's week.

## Attribution

Data from [fantasy.premierleague.com](https://fantasy.premierleague.com) and
[`vaastav/Fantasy-Premier-League`](https://github.com/vaastav/Fantasy-Premier-League).
Phase 2 will additionally draw on `olbauday/FPL-Core-Insights` and
understat.com via `soccerdata`.
