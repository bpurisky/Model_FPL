"""Shared fixtures: synthetic FPL payloads, built fresh per call so tests
can mutate them freely without bleeding state into one another."""

from __future__ import annotations

import copy

import pytest


def _base_bootstrap_payload() -> dict:
    return {
        "events": [
            {
                "id": 1,
                "name": "Gameweek 1",
                "deadline_time": "2026-08-14T17:30:00Z",
                "finished": True,
                "is_current": False,
                "is_next": False,
                "is_previous": True,
            },
            {
                "id": 2,
                "name": "Gameweek 2",
                "deadline_time": "2026-08-21T17:30:00Z",
                "finished": False,
                "is_current": True,
                "is_next": False,
                "is_previous": False,
            },
            {
                "id": 3,
                "name": "Gameweek 3",
                "deadline_time": "2026-08-28T17:30:00Z",
                "finished": False,
                "is_current": False,
                "is_next": True,
                "is_previous": False,
            },
        ],
        "teams": [
            {"id": 1, "name": "Arsenal", "short_name": "ARS", "strength": 4},
            {"id": 2, "name": "Everton", "short_name": "EVE", "strength": 3},
        ],
        "elements": [
            {
                "id": 101,
                "web_name": "Saka",
                "team": 1,
                "element_type": 3,
                "now_cost": 100,
                "selected_by_percent": "45.2",
                "transfers_in_event": 1000,
                "transfers_out_event": 200,
                "form": "5.5",
                "status": "a",
                "chance_of_playing_next_round": 100,
                "news_added": None,
                "ep_next": "6.1",
            },
            {
                "id": 102,
                "web_name": "Calvert-Lewin",
                "team": 2,
                "element_type": 4,
                "now_cost": 55,
                "selected_by_percent": "3.1",
                "transfers_in_event": 50,
                "transfers_out_event": 900,
                "form": "1.0",
                "status": "d",
                "chance_of_playing_next_round": 75,
                "news_added": "2026-08-19T09:00:00Z",
                "ep_next": "1.5",
            },
        ],
    }


def _base_fixtures_payload() -> list[dict]:
    return [
        {
            "id": 1,
            "event": 2,
            "team_h": 1,
            "team_a": 2,
            "kickoff_time": "2026-08-22T14:00:00Z",
            "finished": False,
        }
    ]


@pytest.fixture
def bootstrap_payload() -> dict:
    return copy.deepcopy(_base_bootstrap_payload())


@pytest.fixture
def fixtures_payload() -> list[dict]:
    return copy.deepcopy(_base_fixtures_payload())


@pytest.fixture
def legacy_bootstrap_payload() -> dict:
    """Pre-2025/26 shape: current/next event carried as top-level integers,
    with no per-event is_current/is_next booleans at all."""
    payload = copy.deepcopy(_base_bootstrap_payload())
    payload["current_event"] = 2
    payload["next_event"] = 3
    for event in payload["events"]:
        del event["is_current"]
        del event["is_next"]
        del event["is_previous"]
    return payload
