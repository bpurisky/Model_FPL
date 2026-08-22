"""§0.5 / §2.4: missing expected fields halt validation; unexpected extra
fields only warn; cross-season current/next-event drift is handled."""

from __future__ import annotations

import logging

import pytest
from pydantic import ValidationError

from collector.schemas import (
    fixture_is_played,
    parse_bootstrap_static,
    parse_fixtures,
    resolve_current_event,
    resolve_next_event,
)


def test_valid_payload_parses(bootstrap_payload, fixtures_payload, caplog):
    bootstrap = parse_bootstrap_static(bootstrap_payload, logging.getLogger("test"))
    fixtures = parse_fixtures(fixtures_payload, logging.getLogger("test"))
    assert len(bootstrap.elements) == 2
    assert len(fixtures) == 1


def test_missing_required_field_is_hard_error(bootstrap_payload):
    del bootstrap_payload["elements"][0]["now_cost"]
    with pytest.raises(ValidationError):
        parse_bootstrap_static(bootstrap_payload, logging.getLogger("test"))


def test_unexpected_extra_field_only_warns(bootstrap_payload, caplog):
    bootstrap_payload["elements"][0]["a_brand_new_field_fpl_added"] = "surprise"
    caplog.set_level(logging.WARNING)
    bootstrap = parse_bootstrap_static(bootstrap_payload, logging.getLogger("test"))
    assert len(bootstrap.elements) == 2  # did not raise
    assert any("a_brand_new_field_fpl_added" in record.message for record in caplog.records)


def test_resolve_current_event_prefers_per_event_booleans(bootstrap_payload):
    bootstrap = parse_bootstrap_static(bootstrap_payload, logging.getLogger("test"))
    assert resolve_current_event(bootstrap, bootstrap_payload) == 2
    assert resolve_next_event(bootstrap, bootstrap_payload) == 3


def test_resolve_current_event_falls_back_to_legacy_top_level_ints(legacy_bootstrap_payload):
    bootstrap = parse_bootstrap_static(legacy_bootstrap_payload, logging.getLogger("test"))
    assert resolve_current_event(bootstrap, legacy_bootstrap_payload) == 2
    assert resolve_next_event(bootstrap, legacy_bootstrap_payload) == 3


def test_resolve_current_event_returns_none_when_unresolvable(bootstrap_payload):
    for event in bootstrap_payload["events"]:
        event["is_current"] = False
    bootstrap = parse_bootstrap_static(bootstrap_payload, logging.getLogger("test"))
    assert resolve_current_event(bootstrap, bootstrap_payload) is None


def test_fixture_is_played_accepts_provisional_finish():
    """The live case this predicate exists for: gw1's Friday fixture on
    2026-08-22 reported `finished: false` with `finished_provisional: true`
    eighteen hours after full time."""
    assert fixture_is_played({"finished": False, "finished_provisional": True})


def test_fixture_is_played_rejects_a_match_not_yet_played():
    assert not fixture_is_played({"finished": False, "finished_provisional": False})


def test_fixture_is_played_accepts_a_confirmed_finish():
    assert fixture_is_played({"finished": True, "finished_provisional": True})


def test_fixture_is_played_tolerates_absent_flags():
    """Reads raw payload dicts, some of which (test doubles, an older
    cached payload) predate the field — absent means not played, not a
    KeyError mid-recommendation."""
    assert not fixture_is_played({})


def test_missing_finished_provisional_is_a_hard_error(fixtures_payload):
    """It is depended on now (fixture_is_played), so under this module's
    own rule its absence is real drift and must halt the run rather than
    silently default to a fixture looking unplayed forever."""
    del fixtures_payload[0]["finished_provisional"]
    with pytest.raises(ValidationError):
        parse_fixtures(fixtures_payload, logging.getLogger("test"))


def test_fixture_scores_are_parsed_when_present(fixtures_payload):
    fixtures_payload[0].update({"finished": True, "finished_provisional": True, "team_h_score": 3, "team_a_score": 0})
    fixture = parse_fixtures(fixtures_payload, logging.getLogger("test"))[0]
    assert (fixture.team_h_score, fixture.team_a_score) == (3, 0)
