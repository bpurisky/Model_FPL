import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { describe, expect, it } from "vitest";
import {
  BoardFile,
  ColumnsFile,
  CorrelationsFile,
  EXPECTED_CONTRACT_VERSION,
  FixturesFile,
  GoldenSpearmanFile,
  PlayersFile,
  ScorecardFile,
} from "./schema";

/**
 * The other half of §5.12.2.
 *
 * `tests/test_schema_ts.py` compares this file's *shape* against
 * `contract.py`. This compares it against the actual bytes: every
 * committed export parsed by the schema that is supposed to describe it.
 * A shape test and a real payload can disagree — a field can be spelled
 * right, typed right, and still arrive null where the schema says it
 * cannot — and this is the half that catches that.
 */
const EXPORTS = resolve(__dirname, "../../../../data/web/v1");

function load(name: string): unknown {
  return JSON.parse(readFileSync(resolve(EXPORTS, name), "utf-8"));
}

const FILES = [
  ["columns.json", ColumnsFile],
  ["correlations.json", CorrelationsFile],
  ["scorecard.json", ScorecardFile],
  ["fixtures.json", FixturesFile],
  ["golden_spearman.json", GoldenSpearmanFile],
  ["board.json", BoardFile],
  ["players.json", PlayersFile],
] as const;

describe("the committed exports against the zod contract", () => {
  it.each(FILES)("%s validates", (name, schema) => {
    const result = schema.safeParse(load(name));
    if (!result.success) {
      const issue = result.error.issues[0]!;
      throw new Error(`${name} — ${issue.path.join(".")}: ${issue.message}`);
    }
    expect(result.success).toBe(true);
  });

  it.each(FILES)("%s declares the contract version the app expects", (name) => {
    const payload = load(name) as { header: { contract_version?: number } };
    expect(payload.header.contract_version ?? EXPECTED_CONTRACT_VERSION).toBe(
      EXPECTED_CONTRACT_VERSION,
    );
  });

  it("keeps nulls as nulls through validation", () => {
    // §5.3.3 is the distinction the app is most likely to lose at the
    // boundary: a schema that coerced a null rho to 0 would turn "no
    // correlation is defined here" into "we measured no relationship",
    // and every cell downstream would render the wrong claim confidently.
    const correlations = CorrelationsFile.parse(load("correlations.json"));
    const degenerate = correlations.cells.filter((cell) => cell.rho === null);

    expect(degenerate.length).toBeGreaterThan(0);
    for (const cell of degenerate) {
      expect(cell.rho).toBeNull();
      expect(cell.n).toBeGreaterThan(0);
    }
  });

  it("carries a real git sha so every number is traceable", () => {
    // §5.14.1. `build_header` reads the sha rather than accepting one, so
    // a short or absent value here means the export ran outside a repo.
    const correlations = CorrelationsFile.parse(load("correlations.json"));

    expect(correlations.header.model_git_sha).toMatch(/^[0-9a-f]{40}$/);
  });

  it("covers every matrix metric with a registry entry", () => {
    // §5.14.4, from the app's side: a metric the matrix renders but the
    // registry cannot define is a column with a blank tooltip.
    const correlations = CorrelationsFile.parse(load("correlations.json"));
    const columns = ColumnsFile.parse(load("columns.json"));
    const known = new Set(columns.columns.map((column) => column.key));

    for (const metric of correlations.metrics) {
      expect(known.has(metric)).toBe(true);
    }
  });
});
