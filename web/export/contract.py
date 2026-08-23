"""The schema of record for every exported file (§5.3.1).

These pydantic models are hand-written, and `web/app/src/data/schema.ts`
mirrors them in zod by hand as well. §5.12.2 requires a test asserting the
two agree; generating one from the other is permitted, silent divergence
is not. Until the TypeScript side exists, `contract_shape()` below is what
that test will compare against, so it is the one function here that other
code should not reach around.

Every exported file carries the same header. Two of its fields are
non-negotiable per §5.3.1 and both are about traceability rather than
convenience:

`model_git_sha` — every number on screen must lead back to the exact code
that produced it. The same argument as papertrade/freeze.py's provenance
block, applied to the export rather than to a freeze.

`normalization_basis` — a z-score is meaningless without the population it
was computed against, and §5.7.4 requires the UI to state that basis in a
tooltip. A basis recorded in the file is the only way the tooltip can be
truthful without re-deriving it in TypeScript, which §5.6 forbids.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

CONTRACT_VERSION = 1

Role = Literal["quantitative", "categorical", "ordinal", "temporal"]
Relevance = Literal["primary", "secondary", "context", "none"]
Grain = Literal["player", "player_gameweek", "fixture", "metric_pair", "model_gameweek"]
Source = Literal["fpl_api", "vaastav_archive", "derived", "model"]


class _Strict(BaseModel):
    """Extra fields are a contract violation, not something to tolerate.

    The opposite choice from collector/schemas.py's `_LenientModel`, and
    deliberately so: there the payload belongs to FPL and drift is news to
    be reported, here the payload belongs to this repo and drift is a bug
    on our own side of the wire.
    """

    model_config = ConfigDict(extra="forbid")


class Header(_Strict):
    """§5.3.1's header object, on every exported file."""

    contract_version: int = CONTRACT_VERSION
    generated_at: datetime
    source_gameweek: int | None
    scoring_config: str
    model_git_sha: str | None
    normalization_basis: str
    rows: int


class ColumnSpec(_Strict):
    """One entry in `columns.json` (§5.3.5).

    The registry is called out in the spec as "the highest-leverage file
    in the frontend", on the grounds that getting it wrong makes every
    downstream surface wrong in the same way. That is the reason for
    `extra="forbid"` above and for the required fields below: a column
    that reaches the UI without a definition or a `higher_is_better` is a
    column the UI will render confidently and wrongly.
    """

    key: str
    label: str
    role: Role
    unit: str | None
    format: str
    definition: str
    source: Source
    grain: Grain
    normalizable: bool
    normalized_key: str | None
    position_relevance: dict[Literal["GK", "DEF", "MID", "FWD"], Relevance]
    higher_is_better: bool | None
    available_from_season: str | None

    # --- deviation from §5.3.5's shape, recorded per §5.16 -------------
    # The spec's registry can say when a column starts and has no way to
    # say when one stops. That is not hypothetical: FPL publishes
    # influence/creativity/threat/ict_index as literal 0.0 for all 604
    # elements in 2026-27 while populating everything else, so the series
    # ends without the values ever going missing. Without this field the
    # only honest options are to drop three seasons of real ICT data or
    # to let a column of manufactured zeros render as measurement, and
    # §5.3.3 exists precisely to prevent the second.
    available_to_season: str | None = None

    def applies_to_season(self, season: str) -> bool:
        """Whether this column carries real measurement in `season`.

        Callers use this to write null rather than zero (§5.3.3). String
        comparison is safe because FPL season labels sort lexically in
        chronological order ("2023-24" < "2024-25").
        """
        if self.available_from_season and season < self.available_from_season:
            return False
        if self.available_to_season and season > self.available_to_season:
            return False
        return True


class ColumnsFile(_Strict):
    header: Header
    columns: list[ColumnSpec]


def build_header(
    *,
    rows: int,
    source_gameweek: int | None,
    normalization_basis: str,
    scoring_config: str = "scoring_2026_27.yaml",
    generated_at: datetime | None = None,
) -> Header:
    """The one place a header is constructed, so no exporter can omit a
    field by writing its own dict.

    `model_git_sha` is read here rather than passed in for the same reason
    papertrade/freeze.py reads it at freeze time: a sha supplied by the
    caller is a sha that can be wrong.
    """
    from papertrade.freeze import model_git_sha

    return Header(
        contract_version=CONTRACT_VERSION,
        generated_at=generated_at or datetime.now(timezone.utc),
        source_gameweek=source_gameweek,
        scoring_config=scoring_config,
        model_git_sha=model_git_sha(),
        normalization_basis=normalization_basis,
        rows=rows,
    )


def contract_shape() -> dict[str, Any]:
    """A structural description of every model in this module, for the
    §5.12.2 test that asserts `schema.ts` agrees.

    Deliberately shallower than pydantic's own JSON schema: it compares
    field names, whether each is required, and a coarse type name. A
    literal JSON-schema diff would fail on pydantic-specific detail that
    zod has no way to express, which would make the test noise rather
    than a guard.
    """
    shape: dict[str, Any] = {}
    for model in (Header, ColumnSpec, ColumnsFile):
        fields = {}
        for name, field in model.model_fields.items():
            fields[name] = {
                "required": field.is_required(),
                "type": _type_name(field.annotation),
            }
        shape[model.__name__] = fields
    return shape


def _type_name(annotation: Any) -> str:
    """A coarse, zod-expressible name for an annotation."""
    origin = getattr(annotation, "__origin__", None)
    if origin is not None:
        args = [a for a in getattr(annotation, "__args__", ()) if a is not type(None)]
        optional = len(args) < len(getattr(annotation, "__args__", ()))
        if origin is dict:
            base = "record"
        elif origin is list:
            base = "array"
        elif len(args) == 1:
            base = _type_name(args[0])
        else:
            base = "union"
        return f"{base}?" if optional else base
    if annotation is datetime:
        return "datetime"
    if annotation in (int, float, str, bool):
        return annotation.__name__
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        return annotation.__name__
    return "union"  # Literal[...] and friends


__all__ = [
    "CONTRACT_VERSION",
    "ColumnSpec",
    "ColumnsFile",
    "Header",
    "build_header",
    "contract_shape",
]