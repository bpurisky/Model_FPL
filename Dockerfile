# Deploys service/app.py -- the Squad Optimizer's backend (see its module
# docstring). Not part of the frontend build (that's web/app.yml's Vite/
# GitHub Pages path); this container is the one piece of fpl-trends that
# needs to run as a long-lived process rather than a scheduled job or a
# static export, because squad/optimize.py's ILP solve is a live,
# per-request computation over an arbitrary team ID.
#
# python:3.12-slim (Debian) rather than alpine: pulp's bundled CBC binary
# is a glibc build, and alpine's musl libc will not run it.
FROM python:3.12-slim

WORKDIR /app

RUN pip install --no-cache-dir uv

# Dependencies first, so an app-code-only change doesn't invalidate the
# (slow) dependency layer.
COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-dev --group service

COPY . .

# squad/live.py:build_elo_final_for_current_teams reads
# data/historical/raw/{season}/{fixtures,teams}.csv directly, not the
# committed data/historical/{season}.parquet -- and that raw directory is
# gitignored (backtest/backfill.py's own module docstring flags this exact
# hazard: "if you add another consumer of RAW_CACHE_DIR, it needs the same
# guarantee, not an assumption that some earlier step already populated
# it"). .github/workflows/web.yml already runs this before web.export for
# the identical reason; this container is the other RAW_CACHE_DIR consumer
# the docstring warns about, and a long-lived service has no earlier CI
# step to inherit the cache from, so it has to populate it at build time.
RUN uv run python -m backtest.backfill

ENV PYTHONUNBUFFERED=1
EXPOSE 8000

# Render (and most PaaS hosts) inject $PORT and expect the process to bind
# to it; 8000 is the local/dev fallback. Shell form so ${PORT} expands.
CMD uv run uvicorn service.app:app --host 0.0.0.0 --port ${PORT:-8000}
