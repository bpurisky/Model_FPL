"""§5.12.2: a test asserting `contract.py` and `schema.ts` describe the
same shape.

It runs here rather than in vitest because the boundary needs guarding
from today and the TypeScript toolchain arrives at milestone 5B. §5.3.1
permits generating one side from the other; both are hand-written instead,
which is what makes this test worth having — a generated mirror agrees by
construction and would catch nothing.

The parser is deliberately shallow, matching `contract_shape()`'s own
coarseness: field names, whether each is required, and a broad type. A
literal comparison would fail on pydantic and zod detail neither can
express in the other's terms, which would make this noise rather than a
guard.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from web.export.contract import CONTRACT_VERSION, contract_shape

SCHEMA_TS = Path("web/app/src/data/schema.ts")

# `export const Name = z.object({ ... });`
_MODEL = re.compile(r"^export const (\w+) = z\.object\(\{$(.*?)^\}\);$", re.M | re.S)
# `export const Name = z.enum([...])` — discovered rather than listed, so
# adding a closed-set alias to schema.ts does not require editing this file.
_ENUM = re.compile(r"^export const (\w+) = z\.enum\(", re.M)
# `  field: <zod expression>,`
_FIELD = re.compile(r"^\s{2}(\w+):\s*(.+?),\s*$", re.M)


def _enum_aliases(source: str) -> set[str]:
    return set(_ENUM.findall(source))


def _zod_type(expression: str, enums: frozenset[str] = frozenset()) -> tuple[str, bool]:
    """Reduce a zod expression to `contract_shape()`'s vocabulary.

    Modifiers count only where they *trail* the expression. Substring
    matching would read the inner `.nullable()` of
    `z.array(z.array(z.number().nullable()))` as making the array itself
    nullable, which is a different claim: the matrix is always present,
    and the values inside it are what can be absent.
    """
    nullable = optional = False
    base = expression.strip()
    while True:
        if base.endswith(".nullable()"):
            nullable, base = True, base[: -len(".nullable()")]
        elif base.endswith(".optional()"):
            optional, base = True, base[: -len(".optional()")]
        else:
            break

    if base.startswith("z.array("):
        name = "array"
    elif base.startswith("z.record("):
        name = "record"
    elif base.startswith("z.string().datetime()"):
        name = "datetime"
    elif base.startswith("z.string()"):
        name = "str"
    elif base.startswith("z.boolean()"):
        name = "bool"
    elif base.startswith("z.number().int()"):
        name = "int"
    elif base.startswith("z.number()"):
        name = "float"
    elif base.startswith("z.enum(") or base in enums:
        # A closed set of strings — `contract_shape` calls a Literal a union.
        name = "union"
    else:
        name = base  # a referenced model, e.g. `Header`

    return f"{name}?" if nullable else name, optional


def _parse_schema_ts() -> dict[str, dict[str, dict]]:
    source = SCHEMA_TS.read_text(encoding="utf-8")
    enums = frozenset(_enum_aliases(source))
    models: dict[str, dict[str, dict]] = {}
    for name, body in _MODEL.findall(source):
        fields = {}
        for field, expression in _FIELD.findall(body):
            type_name, optional = _zod_type(expression, enums)
            fields[field] = {"required": not optional, "type": type_name}
        models[name] = fields
    return models


@pytest.fixture(scope="module")
def parsed() -> dict[str, dict[str, dict]]:
    if not SCHEMA_TS.exists():  # pragma: no cover
        pytest.skip("schema.ts not written yet")
    models = _parse_schema_ts()
    assert models, "parsed no models -- schema.ts broke the conventions its header describes"
    return models


def test_schema_ts_matches_the_python_contract(parsed):
    """The §5.12.2 guard. Every model, every field, both directions."""
    python = contract_shape()
    mirrored = {name: fields for name, fields in parsed.items() if name in python}

    assert set(mirrored) == set(python), (
        "model sets differ -- "
        f"missing from schema.ts: {sorted(set(python) - set(mirrored))}; "
        f"unknown to contract.py: {sorted(set(parsed) - set(python) - _ENUM_ONLY)}"
    )

    for name, expected in python.items():
        actual = mirrored[name]
        assert set(actual) == set(expected), f"{name}: field sets differ"
        for field, meta in expected.items():
            assert actual[field]["type"] == meta["type"], (
                f"{name}.{field}: contract.py says {meta['type']}, "
                f"schema.ts says {actual[field]['type']}"
            )
            assert actual[field]["required"] == meta["required"], (
                f"{name}.{field}: required disagrees"
            )


# Zod-side enums have no pydantic model of their own -- they are Literals
# inlined into the fields that use them.
_ENUM_ONLY = frozenset(_enum_aliases(SCHEMA_TS.read_text(encoding="utf-8"))) if SCHEMA_TS.exists() else frozenset()


def test_every_python_model_is_mirrored(parsed):
    """Stated separately from the field comparison so a newly added
    exporter model fails with an obvious message rather than a diff."""
    missing = set(contract_shape()) - set(parsed)

    assert not missing, f"schema.ts has no mirror for: {sorted(missing)}"


def test_nullable_fields_are_nullable_on_both_sides(parsed):
    """The §5.3.3 distinction is the one the app is most likely to lose:
    a zod schema that omits `.nullable()` on a rho rejects the honest
    'no correlation is defined here' payload outright."""
    python = contract_shape()

    for name, fields in python.items():
        for field, meta in fields.items():
            if meta["type"].endswith("?"):
                assert parsed[name][field]["type"].endswith("?"), (
                    f"{name}.{field} is nullable in Python and not in zod"
                )


def test_the_expected_contract_version_agrees():
    """§5.12 requires the build to fail loudly on a version mismatch
    rather than degrade silently, which only works if the app's expected
    version is the one the exporter stamps."""
    source = SCHEMA_TS.read_text(encoding="utf-8")
    match = re.search(r"EXPECTED_CONTRACT_VERSION = (\d+)", source)

    assert match, "schema.ts must declare EXPECTED_CONTRACT_VERSION"
    assert int(match.group(1)) == CONTRACT_VERSION


def test_the_parser_actually_reads_the_file(parsed):
    """A parser returning nothing would make every assertion above pass
    vacuously — the exact failure mode of a regex-based guard."""
    assert len(parsed) >= 18
    assert parsed["CorrelationCell"]["rho"] == {"required": True, "type": "float?"}
    assert parsed["ScorecardRow"]["gw"] == {"required": True, "type": "int?"}
    assert parsed["ColumnSpec"]["position_relevance"]["type"] == "record"
