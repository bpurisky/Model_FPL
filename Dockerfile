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

ENV PYTHONUNBUFFERED=1
EXPOSE 8000

# Render (and most PaaS hosts) inject $PORT and expect the process to bind
# to it; 8000 is the local/dev fallback. Shell form so ${PORT} expands.
CMD uv run uvicorn service.app:app --host 0.0.0.0 --port ${PORT:-8000}
