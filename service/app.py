"""HTTP service backing the frontend's Squad Optimizer surface.

Why this module exists at all: `squad/optimize.py` is an integer linear
program (`pulp`/CBC), and an ILP solve *is* inference, not a reduction of
one -- fpl-trends-frontend-superprompt-v2.md's own architecture rules
(no API server, no DB connection, no runtime Python in the browser; no
inference client-side) rule out running it as part of the otherwise
static, no-build-step frontend. The 2026-08-25 progress-log entry in
fpl-trends-superprompt.md records that decision explicitly and says the
Squad Optimizer stays a stub "until an operator explicitly chooses to
add a backend or a browser-side solver to the architecture". This module
is that choice, made 2026-09-02: a small, stateless HTTP wrapper around
the CLI path that already exists (`squad/__main__.py:cmd_recommend`) --
no logic here beyond request validation, per-IP rate limiting, error
mapping, and response shaping. The solve itself is unchanged.

Run locally:
    uv run uvicorn service.app:app --reload

Deploy: see Dockerfile at the repo root and render.yaml for one option.
ALLOWED_ORIGINS (comma-separated) must be set in production to the
deployed frontend's real origin(s) -- it defaults to "*" only so local
development works with no configuration.
"""

from __future__ import annotations

import logging
import os
import time
from collections import defaultdict, deque
from pathlib import Path
from typing import Any

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

from collector.client import FPLNotFoundError, FPLServerError
from collector.config import load_config
from squad.live import LiveData, build_projections, fetch_live_data, live_data_caveat
from squad.optimize import OptimizationResult, Player, pair_transfers_by_position, optimize_squad, template_risk_flags

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger("service.app")

CONFIG_PATH = Path(os.environ.get("COLLECTOR_CONFIG", "config/collector.yaml"))

# Bounds on the one user-controlled input that changes solve cost. §5.4
# ("the horizon control demonstrably changes output") wants this open to
# the user, not fixed -- but an unauthenticated public endpoint still
# needs a ceiling, since each extra gameweek adds a full XI/captain
# variable block to the ILP. 6 covers a full half-season chip window with
# room to spare and keeps a single solve well under a second in testing.
MAX_HORIZON_LENGTH = 6
MIN_GW, MAX_GW = 1, 38
DEFAULT_HIT_COST = 4
POSITION_ORDER = {"GK": 0, "DEF": 1, "MID": 2, "FWD": 3}

# In-memory, per-process, per-IP. This is one instance behind a free-tier
# host, same reasoning as §7.1's own "add basic per-IP rate limiting" for
# the Cloudflare Worker: an unauthenticated endpoint that fans out to the
# live FPL API and runs a solver on every call needs a cheap abuse guard,
# not a distributed one. Resets on redeploy/restart -- acceptable, since
# the thing being protected is FPL's API and this process's own CPU, not
# a record that needs to survive a restart.
RATE_LIMIT_WINDOW_SECONDS = 60.0
RATE_LIMIT_MAX_REQUESTS = 6
_request_log: dict[str, deque[float]] = defaultdict(deque)


def _client_ip(request: Request) -> str:
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def _rate_limited(client_ip: str) -> bool:
    now = time.monotonic()
    window = _request_log[client_ip]
    while window and now - window[0] > RATE_LIMIT_WINDOW_SECONDS:
        window.popleft()
    if len(window) >= RATE_LIMIT_MAX_REQUESTS:
        return True
    window.append(now)
    return False


class RecommendRequest(BaseModel):
    entry_id: int = Field(gt=0)
    horizon: list[int] | None = None
    max_transfers: int | None = Field(default=None, ge=0)
    hit_cost: int = Field(default=DEFAULT_HIT_COST, ge=0)

    @field_validator("horizon")
    @classmethod
    def _validate_horizon(cls, value: list[int] | None) -> list[int] | None:
        if value is None:
            return value
        if not value:
            raise ValueError("horizon must be non-empty if given")
        if len(value) > MAX_HORIZON_LENGTH:
            raise ValueError(f"horizon may not name more than {MAX_HORIZON_LENGTH} gameweeks")
        if any(gw < MIN_GW or gw > MAX_GW for gw in value):
            raise ValueError(f"horizon gameweeks must be between {MIN_GW} and {MAX_GW}")
        return value


def _allowed_origins() -> list[str]:
    raw = os.environ.get("ALLOWED_ORIGINS")
    if raw:
        return [origin.strip() for origin in raw.split(",") if origin.strip()]
    logger.warning("ALLOWED_ORIGINS is not set; defaulting to '*' (fine for local dev, not for a real deploy)")
    return ["*"]


def _player_json(live: LiveData, pool_by_id: dict[int, Player], element_id: int) -> dict[str, Any]:
    player = pool_by_id[element_id]
    return {
        "element_id": element_id,
        "name": live.web_names.get(element_id, f"#{element_id}"),
        "position": player.position,
        "club": player.club,
        "now_cost": player.now_cost,
    }


def shape_response(
    entry_id: int, live: LiveData, result: OptimizationResult, horizon: list[int]
) -> dict[str, Any]:
    """`OptimizationResult` -> the JSON contract `data/optimizer.ts` parses.
    Deliberately flat and explicit rather than a dataclass `asdict()` --
    the frontend's zod schema is the real contract, and shaping it by hand
    here is what keeps the two able to disagree loudly instead of
    silently, the same argument `web/export/contract.py` makes for the
    static exports.
    """
    pool_by_id = {p.element_id: p for p in live.pool}
    current_ids = {sp.element_id for sp in live.squad.players}

    transfers = [
        {
            "out": _player_json(live, pool_by_id, out_id),
            "in": _player_json(live, pool_by_id, in_id),
        }
        for out_id, in_id in pair_transfers_by_position(result.transfers_out, result.transfers_in, pool_by_id)
    ]

    flags = template_risk_flags(result.transfers_out, live.ownership)
    template_risk = [
        {"element_id": eid, "name": live.web_names.get(eid, f"#{eid}"), "message": message}
        for eid, message in flags.items()
    ]

    starting_xi = {
        str(gw): [
            {**_player_json(live, pool_by_id, eid), "captain": eid == result.captain[gw]}
            for eid in sorted(
                result.starting_xi[gw], key=lambda eid: (POSITION_ORDER[pool_by_id[eid].position], eid)
            )
        ]
        for gw in horizon
    }
    bench_order = [_player_json(live, pool_by_id, eid) for eid in result.bench_order]

    return {
        "entry_id": entry_id,
        "data_gw": live.data_gw,
        "history_gws": live.history_gws,
        "teams_with_played_data": live.teams_with_played_data,
        "teams_total": live.teams_total,
        "caveat": live_data_caveat(live),
        "free_transfers": live.free_transfers,
        "bank": live.squad.bank,
        "horizon": horizon,
        "transfers": transfers,
        "hits_taken": result.hits_taken,
        "bank_after": result.bank_after,
        "template_risk": template_risk,
        "starting_xi": starting_xi,
        "bench_order": bench_order,
        "squad_size": len(result.squad),
        "unchanged_from_current": len(current_ids & result.squad),
    }


async def run_recommendation(payload: RecommendRequest) -> dict[str, Any]:
    cfg = load_config(CONFIG_PATH)
    live = await fetch_live_data(cfg, payload.entry_id, horizon=payload.horizon)
    horizon = payload.horizon or [live.next_event, live.next_event + 1, live.next_event + 2]

    projections = build_projections(live.train_df, live.target_roster, live.scoring_config, live.difficulty_table, horizon)
    result = optimize_squad(
        live.squad,
        live.pool,
        projections,
        horizon=horizon,
        free_transfers=live.free_transfers,
        max_transfers=payload.max_transfers,
        hit_cost=payload.hit_cost,
    )
    return shape_response(payload.entry_id, live, result, horizon)


def create_app() -> FastAPI:
    app = FastAPI(title="fpl-trends squad optimizer", version="1")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_allowed_origins(),
        allow_methods=["GET", "POST"],
        allow_headers=["content-type"],
    )

    @app.get("/api/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/api/recommend")
    async def recommend(payload: RecommendRequest, request: Request) -> dict[str, Any]:
        client_ip = _client_ip(request)
        if _rate_limited(client_ip):
            raise HTTPException(status_code=429, detail="rate limit exceeded -- try again in a minute")

        try:
            return await run_recommendation(payload)
        except FPLNotFoundError as exc:
            raise HTTPException(status_code=404, detail=f"entry {payload.entry_id} not found on the FPL API") from exc
        except FPLServerError as exc:
            raise HTTPException(status_code=502, detail=f"the FPL API returned an error: {exc}") from exc
        except httpx.TransportError as exc:
            raise HTTPException(status_code=502, detail=f"could not reach the FPL API: {exc}") from exc
        except RuntimeError as exc:
            # fetch_live_data's own documented failure modes -- e.g. "entry
            # has no gameweek history yet" before gw1's deadline. Not a
            # server fault, so 422 rather than 500.
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except Exception:
            logger.exception("unhandled error recommending for entry %d", payload.entry_id)
            raise HTTPException(status_code=500, detail="internal error running the solver") from None

    return app


app = create_app()
