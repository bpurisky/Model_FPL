import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

/**
 * `data/optimizer.ts` is the client for the one live backend this app
 * calls (see its own module docstring for why). `OPTIMIZER_API_URL` is
 * read from `import.meta.env` once at module load, so each scenario here
 * stubs the env and `vi.resetModules()`s before re-importing — the only
 * way to exercise both the "no backend configured" and "backend
 * configured" code paths in one file.
 */

const VALID_BODY = {
  entry_id: 123,
  data_gw: 3,
  history_gws: 3,
  teams_with_played_data: 20,
  teams_total: 20,
  caveat: "provisional early-season read",
  free_transfers: 1,
  bank: 5,
  horizon: [4, 5, 6],
  transfers: [],
  hits_taken: 0,
  bank_after: 5,
  template_risk: [],
  starting_xi: { "4": [] },
  bench_order: [],
  squad_size: 15,
  unchanged_from_current: 15,
};

function respond(status: number, ok: boolean, body: unknown) {
  return vi.fn().mockResolvedValue({
    ok,
    status,
    statusText: "status text",
    json: async () => body,
  });
}

describe("no backend configured", () => {
  beforeEach(() => {
    vi.stubEnv("VITE_OPTIMIZER_API_URL", "");
    vi.resetModules();
  });

  afterEach(() => {
    vi.unstubAllEnvs();
    vi.unstubAllGlobals();
  });

  it("rejects without ever calling fetch", async () => {
    const fetchSpy = vi.fn();
    vi.stubGlobal("fetch", fetchSpy);

    const { fetchRecommendation, OptimizerError } = await import("./optimizer");
    await expect(fetchRecommendation(123)).rejects.toBeInstanceOf(OptimizerError);
    expect(fetchSpy).not.toHaveBeenCalled();
  });
});

describe("backend configured", () => {
  beforeEach(() => {
    vi.stubEnv("VITE_OPTIMIZER_API_URL", "https://optimizer.example/");
    vi.resetModules();
  });

  afterEach(() => {
    vi.unstubAllEnvs();
    vi.unstubAllGlobals();
  });

  it("strips a trailing slash and posts entry_id to /api/recommend", async () => {
    const fetchSpy = respond(200, true, VALID_BODY);
    vi.stubGlobal("fetch", fetchSpy);

    const { fetchRecommendation } = await import("./optimizer");
    await fetchRecommendation(123);

    const [url, init] = fetchSpy.mock.calls[0]!;
    expect(url).toBe("https://optimizer.example/api/recommend");
    expect(JSON.parse((init as RequestInit).body as string)).toEqual({ entry_id: 123 });
  });

  it("only includes optional fields when given", async () => {
    const fetchSpy = respond(200, true, VALID_BODY);
    vi.stubGlobal("fetch", fetchSpy);

    const { fetchRecommendation } = await import("./optimizer");
    await fetchRecommendation(123, { horizon: [4, 5], maxTransfers: 1 });

    const [, init] = fetchSpy.mock.calls[0]!;
    expect(JSON.parse((init as RequestInit).body as string)).toEqual({
      entry_id: 123,
      horizon: [4, 5],
      max_transfers: 1,
    });
  });

  it("parses a valid response through the zod contract", async () => {
    vi.stubGlobal("fetch", respond(200, true, VALID_BODY));

    const { fetchRecommendation } = await import("./optimizer");
    const result = await fetchRecommendation(123);
    expect(result.entry_id).toBe(123);
    expect(result.horizon).toEqual([4, 5, 6]);
  });

  it("surfaces the backend's own detail string and status on a non-2xx response", async () => {
    vi.stubGlobal("fetch", respond(404, false, { detail: "entry 999 not found on the FPL API" }));

    const { fetchRecommendation, OptimizerError } = await import("./optimizer");
    try {
      await fetchRecommendation(999);
      expect.unreachable("should have thrown");
    } catch (error) {
      expect(error).toBeInstanceOf(OptimizerError);
      expect((error as InstanceType<typeof OptimizerError>).status).toBe(404);
      expect((error as Error).message).toContain("999");
    }
  });

  it("rejects a response that fails the zod contract rather than rendering it", async () => {
    vi.stubGlobal("fetch", respond(200, true, { nonsense: true }));

    const { fetchRecommendation } = await import("./optimizer");
    await expect(fetchRecommendation(123)).rejects.toThrow(/Unexpected response shape/);
  });

  it("wraps a network failure without leaking the raw fetch error", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockRejectedValue(new TypeError("Failed to fetch")),
    );

    const { fetchRecommendation, OptimizerError } = await import("./optimizer");
    await expect(fetchRecommendation(123)).rejects.toBeInstanceOf(OptimizerError);
  });
});
