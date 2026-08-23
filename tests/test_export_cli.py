"""§5.11.1 over the export CLI: that a rebuild which changes nothing
writes nothing."""

from __future__ import annotations

import json

from web.export.__main__ import _body, write_json


def _payload(rho: float, generated_at: str, sha: str) -> str:
    return json.dumps(
        {
            "header": {
                "generated_at": generated_at,
                "model_git_sha": sha,
                "rows": 1,
            },
            "cells": [{"rho": rho}],
        },
        indent=2,
    )


def test_a_rebuild_with_the_same_numbers_does_not_rewrite_the_file(tmp_path):
    """These files are committed and the deploy workflow rebuilds them on
    every collector run. `generated_at` and `model_git_sha` move every
    time regardless of what the numbers did, so an unconditional write
    would commit all five files hourly and bury a real change."""
    first, changed = write_json(_payload(0.5, "2026-08-23T10:00:00Z", "aaa"), "x.json", tmp_path)
    assert changed is True
    before = first.read_text(encoding="utf-8")

    _, changed_again = write_json(_payload(0.5, "2026-08-24T11:00:00Z", "bbb"), "x.json", tmp_path)

    assert changed_again is False
    assert first.read_text(encoding="utf-8") == before, "the older sha stays, and truthfully"


def test_a_changed_number_does_rewrite(tmp_path):
    write_json(_payload(0.5, "2026-08-23T10:00:00Z", "aaa"), "x.json", tmp_path)

    path, changed = write_json(_payload(0.6, "2026-08-23T10:00:00Z", "aaa"), "x.json", tmp_path)

    assert changed is True
    assert '"rho": 0.6' in path.read_text(encoding="utf-8")


def test_the_body_comparison_ignores_only_the_volatile_header_fields(tmp_path):
    same = _body(_payload(0.5, "2026-08-23T10:00:00Z", "aaa"))
    later = _body(_payload(0.5, "2030-01-01T00:00:00Z", "zzz"))
    different = _body(_payload(0.5001, "2026-08-23T10:00:00Z", "aaa"))

    assert same == later
    assert same != different
    assert '"rows": 1' in same, "the rest of the header still counts"
