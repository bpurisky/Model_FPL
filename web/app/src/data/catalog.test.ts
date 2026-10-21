import { readdirSync, readFileSync } from "node:fs";
import { resolve } from "node:path";
import { describe, expect, it } from "vitest";
import { CADENCE_TEXT, DATA_FILES } from "./catalog";

/**
 * The Data & API page documents what the data layer actually serves, so
 * this test derives "what the data layer serves" from independent places
 * and requires the catalog to cover all of them: the file names `load.ts`
 * passes to `loadFile`, and the exports committed under `data/web/v1/`
 * (minus the `golden_` test fixtures, which the build does not ship).
 * Parquet artifacts are not committed, so they are checked against the
 * URLs the query layer fetches.
 */

const SRC = resolve(__dirname, "..");
const EXPORT_DIR = resolve(__dirname, "../../../../data/web/v1");
const names = new Set(DATA_FILES.map((file) => file.name));

describe("data catalog", () => {
  it("has no duplicate entries", () => {
    expect(names.size).toBe(DATA_FILES.length);
  });

  it("covers every file load.ts fetches", () => {
    const source = readFileSync(resolve(SRC, "data/load.ts"), "utf-8");
    const loaded = [...source.matchAll(/loadFile\(\s*"([^"]+)"/g)].map((m) => m[1]!);
    expect(loaded.length).toBeGreaterThan(0);
    expect(loaded.filter((name) => !names.has(name))).toEqual([]);
  });

  it("covers every JSON export committed under data/web/v1", () => {
    const committed = readdirSync(EXPORT_DIR).filter(
      (name) => name.endsWith(".json") && !name.startsWith("golden_"),
    );
    expect(committed.length).toBeGreaterThan(0);
    expect(committed.filter((name) => !names.has(name))).toEqual([]);
  });

  it("lists no JSON file that is not a real export", () => {
    const committed = new Set(readdirSync(EXPORT_DIR));
    const missing = DATA_FILES.filter((f) => f.format === "json" && !committed.has(f.name));
    expect(missing.map((f) => f.name)).toEqual([]);
  });

  it("covers every parquet file the query layer fetches", () => {
    const fetched = [
      readFileSync(resolve(SRC, "query/session.ts"), "utf-8"),
      readFileSync(resolve(SRC, "query/series.ts"), "utf-8"),
    ].flatMap((source) =>
      [...source.matchAll(/\$\{DATA_BASE\}\/([\w.]+\.parquet)/g)].map((m) => m[1]!),
    );
    expect(fetched.length).toBe(2);
    expect(fetched.filter((name) => !names.has(name))).toEqual([]);
  });

  it("gives every JSON file a loader and a stated cadence", () => {
    for (const file of DATA_FILES) {
      expect(CADENCE_TEXT[file.cadence]).toBeTruthy();
      expect(file.description.trim()).not.toBe("");
      if (file.format === "json") expect(file.loadHeader, file.name).toBeTypeOf("function");
    }
  });
});
