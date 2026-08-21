"""Point-in-time discipline, enforced in code (§0.3, §3.3).

Every feature used to predict a gameweek carries a declared `available_at`
timestamp. Before training or evaluation, assert for every feature:

    feature.available_at < gameweek.deadline_time

A violation raises. It does not warn.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


class LeakageError(RuntimeError):
    """A feature's available_at is not strictly before the target deadline."""


@dataclass(frozen=True)
class Feature:
    name: str
    element_id: int
    value: float | None
    available_at: datetime


def assert_no_leakage(features: list[Feature], deadline_time: datetime, *, context: str = "") -> None:
    violations = [f for f in features if not (f.available_at < deadline_time)]
    if violations:
        sample = violations[:5]
        detail = ", ".join(f"{f.name}[element_id={f.element_id}] available_at={f.available_at}" for f in sample)
        more = f" (+{len(violations) - 5} more)" if len(violations) > 5 else ""
        prefix = f"{context}: " if context else ""
        raise LeakageError(
            f"{prefix}{len(violations)} feature(s) available at or after deadline_time={deadline_time}: {detail}{more}"
        )
