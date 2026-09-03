"""service/app.py -- the HTTP wrapper around squad/optimize.py's live ILP
solve (see service/app.py's module docstring for why this exists at all:
the 2026-08-25 decision to keep the Squad Optimizer frontend stubbed
"until an operator explicitly chooses to add a backend", reversed
2026-09-02).

`run_recommendation` (the one function that actually calls the live FPL
API via `squad.live.fetch_live_data`) is monkeypatched throughout --
that live-data wiring is already covered by tests/test_squad_live.py and
tests/test_optimize.py. What is under test here is everything this
module adds on top: request validation, per-IP rate limiting, and the
mapping from squad/live.py's and squad/optimize.py's own exceptions to
HTTP status codes.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

import service.app as app_module
from collector.client import FPLNotFoundError, FPLServerError

CANNED_RESULT = {
    "entry_id": 123,
    "data_gw": 3,
    "history_gws": 3,
    "teams_with_played_data": 20,
    "teams_total": 20,
    "caveat": "provisional early-season read",
    "free_transfers": 1,
    "bank": 5,
    "horizon": [4, 5, 6],
    "transfers": [],
    "hits_taken": 0,
    "bank_after": 5,
    "template_risk": [],
    "starting_xi": {"4": []},
    "bench_order": [],
    "squad_size": 15,
    "unchanged_from_current": 15,
}


@pytest.fixture(autouse=True)
def _reset_rate_limit():
    """Rate limiting is module-level, per-process state (service/app.py's
    own docstring explains why: one instance, no external store needed).
    Tests share that state unless cleared between runs."""
    app_module._request_log.clear()
    yield
    app_module._request_log.clear()


@pytest.fixture
def client() -> TestClient:
    return TestClient(app_module.app)


def test_health(client: TestClient):
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_recommend_success(client: TestClient, monkeypatch):
    async def fake_run(payload):
        assert payload.entry_id == 123
        return CANNED_RESULT

    monkeypatch.setattr(app_module, "run_recommendation", fake_run)

    response = client.post("/api/recommend", json={"entry_id": 123})
    assert response.status_code == 200
    assert response.json() == CANNED_RESULT


def test_entry_not_found_maps_to_404(client: TestClient, monkeypatch):
    async def fake_run(payload):
        raise FPLNotFoundError("https://fantasy.premierleague.com/api/entry/999/history/")

    monkeypatch.setattr(app_module, "run_recommendation", fake_run)

    response = client.post("/api/recommend", json={"entry_id": 999})
    assert response.status_code == 404
    assert "999" in response.json()["detail"]


def test_fpl_server_error_maps_to_502(client: TestClient, monkeypatch):
    async def fake_run(payload):
        raise FPLServerError("https://fantasy.premierleague.com/api/bootstrap-static/", 503)

    monkeypatch.setattr(app_module, "run_recommendation", fake_run)

    response = client.post("/api/recommend", json={"entry_id": 123})
    assert response.status_code == 502


def test_no_gameweek_history_yet_maps_to_422(client: TestClient, monkeypatch):
    async def fake_run(payload):
        raise RuntimeError("entry 123 has no gameweek history yet (has gw1's deadline passed?)")

    monkeypatch.setattr(app_module, "run_recommendation", fake_run)

    response = client.post("/api/recommend", json={"entry_id": 123})
    assert response.status_code == 422
    assert "no gameweek history" in response.json()["detail"]


def test_unexpected_error_maps_to_500_without_leaking_internals(client: TestClient, monkeypatch):
    async def fake_run(payload):
        raise ValueError("pool is missing currently-owned players: [42]")

    monkeypatch.setattr(app_module, "run_recommendation", fake_run)

    response = client.post("/api/recommend", json={"entry_id": 123})
    assert response.status_code == 500
    assert "42" not in response.json()["detail"]  # internals not echoed to the caller


def test_invalid_entry_id_rejected_before_any_solve(client: TestClient, monkeypatch):
    called = {"n": 0}

    async def fake_run(payload):
        called["n"] += 1
        return CANNED_RESULT

    monkeypatch.setattr(app_module, "run_recommendation", fake_run)

    response = client.post("/api/recommend", json={"entry_id": 0})
    assert response.status_code == 422
    assert called["n"] == 0


@pytest.mark.parametrize(
    "horizon,reason",
    [
        ([], "empty"),
        (list(range(1, app_module.MAX_HORIZON_LENGTH + 2)), "too long"),
        ([0], "gw below range"),
        ([39], "gw above range"),
    ],
)
def test_invalid_horizon_rejected(client: TestClient, monkeypatch, horizon, reason):
    async def fake_run(payload):
        return CANNED_RESULT

    monkeypatch.setattr(app_module, "run_recommendation", fake_run)

    response = client.post("/api/recommend", json={"entry_id": 123, "horizon": horizon})
    assert response.status_code == 422, reason


def test_rate_limit_blocks_after_the_configured_ceiling(client: TestClient, monkeypatch):
    async def fake_run(payload):
        return CANNED_RESULT

    monkeypatch.setattr(app_module, "run_recommendation", fake_run)

    statuses = [
        client.post("/api/recommend", json={"entry_id": 123}).status_code
        for _ in range(app_module.RATE_LIMIT_MAX_REQUESTS + 1)
    ]
    assert statuses[:-1] == [200] * app_module.RATE_LIMIT_MAX_REQUESTS
    assert statuses[-1] == 429


def test_rate_limit_is_per_ip(client: TestClient, monkeypatch):
    async def fake_run(payload):
        return CANNED_RESULT

    monkeypatch.setattr(app_module, "run_recommendation", fake_run)

    for _ in range(app_module.RATE_LIMIT_MAX_REQUESTS):
        client.post("/api/recommend", json={"entry_id": 123}, headers={"x-forwarded-for": "1.1.1.1"})

    blocked = client.post("/api/recommend", json={"entry_id": 123}, headers={"x-forwarded-for": "1.1.1.1"})
    assert blocked.status_code == 429

    other_ip = client.post("/api/recommend", json={"entry_id": 123}, headers={"x-forwarded-for": "2.2.2.2"})
    assert other_ip.status_code == 200


def test_shape_response_pairs_transfers_and_marks_the_captain():
    """A thin end-to-end check of shape_response against real
    squad/optimize.py types, rather than a canned dict -- this is the one
    function in the module with logic worth exercising directly."""
    from squad.live import LiveData
    from squad.optimize import OptimizationResult, Player
    from squad.reconstruct import SquadState

    pool = [
        Player(element_id=1, position="GK", club="ARS", now_cost=50),
        Player(element_id=2, position="GK", club="LIV", now_cost=55),
        Player(element_id=3, position="FWD", club="ARS", now_cost=100),
        Player(element_id=4, position="FWD", club="LIV", now_cost=110),
    ]
    result = OptimizationResult(
        squad=frozenset({2, 3, 4}),
        transfers_out=frozenset({1}),
        transfers_in=frozenset({2}),
        starting_xi={4: frozenset({2, 3, 4})},
        captain={4: 4},
        bench_order=(),
        bank_after=5,
        hits_taken=0,
        objective_value=42.0,
    )
    from datetime import datetime, timezone

    live = LiveData(
        squad=SquadState(as_of=datetime.now(timezone.utc), players=(), bank=5),
        pool=pool,
        free_transfers=1,
        current_event=3,
        next_event=4,
        train_df=None,  # unused by shape_response
        target_roster=None,  # unused by shape_response
        difficulty_table=None,  # unused by shape_response
        scoring_config={},
        ownership={1: 45.0},  # the player sold OUT -- template risk flags a sale, not a buy
        web_names={1: "Keeper1", 2: "Keeper2", 3: "Striker3", 4: "Striker4"},
        now_cost_by_id={1: 50, 2: 55, 3: 100, 4: 110},
        data_gw=3,
        teams_with_played_data=20,
        teams_total=20,
        history_gws=3,
    )

    body = app_module.shape_response(999, live, result, [4])

    assert body["entry_id"] == 999
    assert body["transfers"] == [{"out": {"element_id": 1, "name": "Keeper1", "position": "GK", "club": "ARS", "now_cost": 50}, "in": {"element_id": 2, "name": "Keeper2", "position": "GK", "club": "LIV", "now_cost": 55}}]
    assert len(body["template_risk"]) == 1
    assert body["template_risk"][0]["element_id"] == 1
    assert "45.0%" in body["template_risk"][0]["message"]
    # Sorted GK/DEF/MID/FWD then element_id, matching squad/__main__.py's
    # own CLI report order rather than an alphabetical accident.
    assert body["starting_xi"]["4"] == [
        {"element_id": 2, "name": "Keeper2", "position": "GK", "club": "LIV", "now_cost": 55, "captain": False},
        {"element_id": 3, "name": "Striker3", "position": "FWD", "club": "ARS", "now_cost": 100, "captain": False},
        {"element_id": 4, "name": "Striker4", "position": "FWD", "club": "LIV", "now_cost": 110, "captain": True},
    ]
    assert body["squad_size"] == 3
    assert body["unchanged_from_current"] == 0
