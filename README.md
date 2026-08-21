# fpl-trends

A Fantasy Premier League analytics platform for the 2026/27 season. Built to
[`fpl-trends-superprompt.md`](fpl-trends-superprompt.md); this README tracks
what actually exists against that spec.

**Status: Phase 0 (collector) only.** Phases 1-5 (backtest harness, event
model, squad optimizer, paper trade, frontend) are not built yet.

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

Two dependencies were added beyond the superprompt's locked list, each
justified at its import site:

- **pyyaml** — the repo layout (§1.2) mandates `config/*.yaml` files;
  something has to parse them (`collector/config.py`).
- **pytz** — duckdb's Python DBAPI needs it to convert `TIMESTAMPTZ`
  columns back to Python datetimes via `fetchall()` (the `.pl()`/Arrow path
  used internally by the collector doesn't need it — this only bites ad-hoc
  `duckdb.sql(...).fetchall()` querying of the distilled data, as in the
  example above).

## Attribution

Data from [fantasy.premierleague.com](https://fantasy.premierleague.com).
Phase 1 will additionally draw on `vaastav/Fantasy-Premier-League`,
`olbauday/FPL-Core-Insights`, and understat.com via `soccerdata`.
