"""Captaincy candidates: the next gameweek's players ranked by xP, each
with P(haul) (`analytics.haul`).

One function shared by the weekly freeze (`papertrade/freeze.py`, which
records it before the deadline so it can be scored) and the web export
(`web/export/captaincy.py`, which publishes it), so the table a reader sees
and the pick that gets scored are built by the same code.

The pick is the top candidate by xP, with ties broken by element id. It is
the model's expected-points ranking and nothing else: no ownership,
differential or fixture-swing adjustment is layered on. Whether that is a
good rule is what the paper trade's captaincy record measures.
"""

from __future__ import annotations

from typing import Any

import polars as pl

from analytics.haul import HAUL_THRESHOLD, LOW_THRESHOLD, BonusModel, haul_summary
from analytics.projections import expected_points_from_projection

TOP_N = 20  # candidates shipped and frozen
SHORTLIST = 5  # the "best captain among the model's shortlist" baseline


def rank_candidates(
    projected: pl.DataFrame,
    config: dict[str, Any],
    meta: dict[int, dict[str, Any]],
    top_n: int = TOP_N,
    bonus_model: BonusModel | None = None,
) -> list[dict[str, Any]]:
    """The `top_n` players by xP for the projected gameweek.

    `projected` is `project_event_vectors`' output joined with the scoreline
    columns, for one gameweek. `meta` maps element_id to `name`, `team` and
    `ownership_pct` (any may be missing). Players with no fixture are never
    candidates. `bonus_model` is `analytics.haul.load_bonus_model`'s. Each row: element_id, name, team, position, xp, p_haul,
    p_low, ownership_pct.
    """
    rows = []
    for row in projected.to_dicts():
        if not row.get("n_fixtures"):
            continue
        xp = expected_points_from_projection(row, config)
        rows.append((xp, row))
    rows.sort(key=lambda item: (-item[0], int(item[1]["element_id"])))
    out = []
    for xp, row in rows[:top_n]:
        eid = int(row["element_id"])
        summary = haul_summary(row, config, bonus_model)
        info = meta.get(eid, {})
        out.append(
            {
                "element_id": eid,
                "name": info.get("name"),
                "team": info.get("team", row.get("team")),
                "position": row["position"],
                "xp": float(xp),
                "p_haul": summary["p_haul"],
                "p_low": summary["p_low"],
                "ownership_pct": info.get("ownership_pct"),
            }
        )
    return out


def most_owned(meta: dict[int, dict[str, Any]]) -> dict[str, Any] | None:
    """The highest-owned player, the 'captain the template pick' baseline.
    Ties break to the lower element id."""
    owned = [(info["ownership_pct"], eid) for eid, info in meta.items() if info.get("ownership_pct") is not None]
    if not owned:
        return None
    pct, eid = min(owned, key=lambda item: (-item[0], item[1]))
    return {"element_id": eid, "name": meta[eid].get("name"), "team": meta[eid].get("team"), "ownership_pct": pct}


def freeze_record(candidates: list[dict[str, Any]], owned: dict[str, Any] | None) -> dict[str, Any]:
    """The `captaincy` block written into a freeze. `pick` is the top
    candidate, `shortlist` the top SHORTLIST; `candidates` is every row so
    the calibration can score all of their P(haul)s."""
    return {
        "haul_threshold": HAUL_THRESHOLD,
        "low_threshold": LOW_THRESHOLD,
        "method": "exact_enumeration",
        "pick": candidates[0] if candidates else None,
        "shortlist": candidates[:SHORTLIST],
        "candidates": candidates,
        "most_owned": owned,
    }
