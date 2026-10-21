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

  it("parses team rating and chip comparison when present", async () => {
    const body = {
      ...VALID_BODY,
      current_xi_xp: 120.5,
      recommended_xp: 131.2,
      optimum_xp: 150,
      team_rating: 80.3,
      chips: [{ chip: "bboost", gw: 4, gain: 7.1, available: true }],
      chip_best_gw: { bboost: 4 },
    };
    vi.stubGlobal("fetch", respond(200, true, body));

    const { fetchRecommendation } = await import("./optimizer");
    const result = await fetchRecommendation(123);
    expect(result.team_rating).toBe(80.3);
    expect(result.chips?.[0]?.chip).toBe("bboost");
  });

  it("rejects an unknown chip id", async () => {
    vi.stubGlobal("fetch", respond(200, true, { ...VALID_BODY, chips: [{ chip: "magic", gw: 4, gain: 1 }] }));

    const { fetchRecommendation } = await import("./optimizer");
    await expect(fetchRecommendation(123)).rejects.toThrow(/Unexpected response shape/);
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

describe("parseEntryId", () => {
  it("accepts a bare ID and the Points page link it is easiest to copy", async () => {
    const { parseEntryId } = await import("./optimizer");
    expect(parseEntryId("2986528")).toBe(2986528);
    expect(parseEntryId("  2986528 ")).toBe(2986528);
    expect(parseEntryId("https://fantasy.premierleague.com/entry/2986528/event/7")).toBe(2986528);
    expect(parseEntryId("fantasy.premierleague.com/entry/2986528/history")).toBe(2986528);
  });

  it("rejects anything that does not yield a positive integer", async () => {
    const { parseEntryId } = await import("./optimizer");
    for (const raw of ["", "abc", "0", "-5", "12.5", "https://fantasy.premierleague.com/leagues"]) {
      expect(parseEntryId(raw)).toBeUndefined();
    }
  });
});

describe("waking and timing out", () => {
  beforeEach(() => {
    vi.stubEnv("VITE_OPTIMIZER_API_URL", "https://optimizer.example");
    vi.resetModules();
  });

  afterEach(() => {
    vi.unstubAllEnvs();
    vi.unstubAllGlobals();
  });

  it("warmUp pings /api/health and never throws", async () => {
    const fetchSpy = vi.fn().mockResolvedValue({ ok: true });
    vi.stubGlobal("fetch", fetchSpy);
    const { warmUp } = await import("./optimizer");

    expect(await warmUp()).toBe(true);
    expect(fetchSpy.mock.calls[0]![0]).toBe("https://optimizer.example/api/health");

    vi.stubGlobal("fetch", vi.fn().mockRejectedValue(new TypeError("Failed to fetch")));
    expect(await warmUp()).toBe(false);
  });

  it("gives up on a request that never answers, with a timeout status", async () => {
    // A fetch that only settles when its signal aborts -- the stalled case.
    vi.stubGlobal(
      "fetch",
      vi.fn((_url: string, init: RequestInit) =>
        new Promise((_resolve, reject) => {
          init.signal?.addEventListener("abort", () => reject(new DOMException("aborted", "AbortError")));
        }),
      ),
    );
    const { fetchRecommendation, OptimizerError, TIMEOUT_STATUS } = await import("./optimizer");

    const error = await fetchRecommendation(123, { timeoutMs: 20 }).catch((e: unknown) => e);
    expect(error).toBeInstanceOf(OptimizerError);
    expect((error as InstanceType<typeof OptimizerError>).status).toBe(TIMEOUT_STATUS);
    expect((error as Error).message).toMatch(/didn't answer/);
  });
});
