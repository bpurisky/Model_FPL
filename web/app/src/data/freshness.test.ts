import { describe, expect, it } from "vitest";
import { freshness } from "./load";
import { headerFromMetadata } from "../query/session";

const now = new Date("2026-10-21T12:00:00Z");

describe("freshness", () => {
  it("is not stale when the numbers are old but the export reran since", () => {
    const f = freshness("2026-09-25T00:00:00Z", "2026-10-21T09:00:00Z", now);
    expect(f).toEqual({ age: 3, stale: false, unchanged: true });
  });

  it("is stale when the export itself stopped running", () => {
    const f = freshness("2026-10-19T00:00:00Z", "2026-10-20T00:00:00Z", now);
    expect(f.stale).toBe(true);
    expect(f.age).toBe(36);
  });

  it("falls back to generated_at when no export time was stamped", () => {
    expect(freshness("2026-10-21T10:00:00Z", undefined, now)).toEqual({
      age: 2,
      stale: false,
      unchanged: false,
    });
    expect(freshness("2026-10-20T00:00:00Z", undefined, now).stale).toBe(true);
  });
});

describe("headerFromMetadata", () => {
  const header = {
    contract_version: 1,
    generated_at: "2026-10-21T12:00:00Z",
    source_gameweek: null,
    scoring_config: "scoring_2026_27.yaml",
    model_git_sha: "abc",
    normalization_basis: "x",
    rows: 3,
    current_season: "2026-27",
  };

  it("reads the stamped header", () => {
    const kv = [{ key: "ARROW:schema", value: "…" }, { key: "fpl_trends_header", value: JSON.stringify(header) }];
    expect(headerFromMetadata(kv)?.rows).toBe(3);
  });

  it("is null for a file built before the header existed, or a malformed one", () => {
    expect(headerFromMetadata(undefined)).toBeNull();
    expect(headerFromMetadata([{ key: "fpl_trends_header", value: "{not json" }])).toBeNull();
    expect(headerFromMetadata([{ key: "fpl_trends_header", value: "{}" }])).toBeNull();
  });
});
