import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { LoadProgress } from "./load";

/**
 * `loadFile` keeps a module-scope cache, so each scenario re-imports the
 * module after `vi.resetModules()` to start from an empty one.
 */

const COLUMNS = {
  header: {
    generated_at: "2026-10-21T12:00:00Z",
    source_gameweek: null,
    scoring_config: "scoring.yaml",
    model_git_sha: null,
    normalization_basis: "x",
    rows: 0,
  },
  columns: [],
};

const CHUNK = 16;

function jsonResponse(body: unknown, headers: Record<string, string> = {}) {
  const bytes = new TextEncoder().encode(JSON.stringify(body));
  const stream = new ReadableStream<Uint8Array>({
    start(controller) {
      for (let i = 0; i < bytes.length; i += CHUNK) {
        controller.enqueue(bytes.slice(i, i + CHUNK));
      }
      controller.close();
    },
  });
  return new Response(stream, { status: 200, headers });
}

beforeEach(() => {
  vi.resetModules();
});

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("loadFile cache", () => {
  it("does not fetch a file twice", async () => {
    const fetchMock = vi.fn(async () => jsonResponse(COLUMNS));
    vi.stubGlobal("fetch", fetchMock);
    const { loadColumns } = await import("./load");

    const first = await loadColumns();
    const second = await loadColumns();

    expect(fetchMock).toHaveBeenCalledTimes(1);
    expect(second).toBe(first);
  });

  it("shares one request between concurrent callers", async () => {
    const fetchMock = vi.fn(async () => jsonResponse(COLUMNS));
    vi.stubGlobal("fetch", fetchMock);
    const { loadColumns } = await import("./load");

    await Promise.all([loadColumns(), loadColumns()]);

    expect(fetchMock).toHaveBeenCalledTimes(1);
  });

  it("retries after a failed load", async () => {
    const fetchMock = vi
      .fn()
      .mockResolvedValueOnce(new Response("nope", { status: 500, statusText: "boom" }))
      .mockImplementation(async () => jsonResponse(COLUMNS));
    vi.stubGlobal("fetch", fetchMock);
    const { loadColumns } = await import("./load");

    await expect(loadColumns()).rejects.toThrow("500");
    await expect(loadColumns()).resolves.toMatchObject({ columns: [] });
    expect(fetchMock).toHaveBeenCalledTimes(2);
  });
});

describe("readWithProgress", () => {
  it("reports a null total for an encoded response", async () => {
    const { readWithProgress } = await import("./load");
    const seen: LoadProgress[] = [];
    await readWithProgress(
      jsonResponse(COLUMNS, { "content-encoding": "gzip", "content-length": "10" }),
      (p) => seen.push(p),
    );
    expect(seen.length).toBeGreaterThan(0);
    expect(seen.every((p) => p.total === null)).toBe(true);
  });

  it("reports content-length when the response is not encoded", async () => {
    const { readWithProgress } = await import("./load");
    const length = JSON.stringify(COLUMNS).length;
    const seen: LoadProgress[] = [];
    await readWithProgress(jsonResponse(COLUMNS, { "content-length": String(length) }), (p) =>
      seen.push(p),
    );
    expect(seen.length).toBeGreaterThan(0);
    expect(seen.every((p) => p.total === length)).toBe(true);
  });

  it("treats identity encoding with no length as an unknown total", async () => {
    const { readWithProgress } = await import("./load");
    const seen: LoadProgress[] = [];
    await readWithProgress(jsonResponse(COLUMNS, { "content-encoding": "identity" }), (p) =>
      seen.push(p),
    );
    expect(seen.every((p) => p.total === null)).toBe(true);
  });

  it("counts received bytes and returns them all", async () => {
    const { readWithProgress } = await import("./load");
    const expected = new TextEncoder().encode(JSON.stringify(COLUMNS));
    const seen: number[] = [];
    const bytes = await readWithProgress(jsonResponse(COLUMNS), (p) => seen.push(p.received));
    expect(Array.from(bytes)).toEqual(Array.from(expected));
    expect(seen.at(-1)).toBe(expected.length);
    expect(seen).toEqual([...seen].sort((a, b) => a - b));
    expect(seen.length).toBe(Math.ceil(expected.length / CHUNK));
  });
});
