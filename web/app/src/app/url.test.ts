import { describe, expect, it } from "vitest";
import { DEFAULT_STATE, parseUrl, toSearch, VIEWS, type AppUrlState } from "./url";

/**
 * §5.5: "Every view's selection encodes into the query string, including
 * the full Graph Builder encoding spec, so a finding can be linked."
 *
 * A link is only worth anything if it comes back as what it left as, so
 * the property under test is the round trip rather than the format. The
 * format is checked separately in the readability tests below, because
 * §5.5 calls the link itself the deliverable — a reader should be able to
 * edit `pos=MID` in the address bar and get what they expect.
 */

const FULL: AppUrlState = {
  view: "graph",
  position: "MID",
  normalized: true,
  filters: {
    seasons: ["2024-25", "2025-26"],
    positions: ["MID", "FWD"],
    teams: ["ARS", "LIV"],
    elements: [430],
    priceMin: 45,
    priceMax: 130,
    minutesFloor: 450,
    gwMin: 3,
    gwMax: 38,
  },
  selection: [233, 401],
  encoding: {
    x: "xg_per90",
    y: "total_points",
    color: "team",
    wrap: "position",
    aggregate: "median",
  },
  entry: 2986528,
};

describe("round tripping", () => {
  it("returns a fully populated state unchanged", () => {
    expect(parseUrl(toSearch(FULL))).toEqual(FULL);
  });

  it("returns the default state unchanged, and encodes it as nothing", () => {
    expect(toSearch(DEFAULT_STATE)).toBe("");
    expect(parseUrl("")).toEqual(DEFAULT_STATE);
  });

  it("round trips every view", () => {
    for (const view of VIEWS) {
      const state = { ...DEFAULT_STATE, view };
      expect(parseUrl(toSearch(state)).view).toBe(view);
    }
  });

  it("round trips one-sided ranges", () => {
    const cases: [number | null, number | null][] = [
      [null, null],
      [5, null],
      [null, 20],
      [5, 20],
    ];
    for (const [lo, hi] of cases) {
      const state: AppUrlState = {
        ...DEFAULT_STATE,
        filters: { ...DEFAULT_STATE.filters, gwMin: lo, gwMax: hi },
      };
      const back = parseUrl(toSearch(state));
      expect([back.filters.gwMin, back.filters.gwMax]).toEqual([lo, hi]);
    }
  });

  it("is idempotent — encoding a parsed state reproduces the search", () => {
    const once = toSearch(FULL);
    expect(toSearch(parseUrl(once))).toBe(once);
  });
});

describe("the link stays readable", () => {
  it("omits everything sitting at its default", () => {
    // The point of the omission: a link to the hero surface with nothing
    // selected must not be forty parameters spelling out that nothing is
    // set.
    expect(toSearch({ ...DEFAULT_STATE, view: "correlations" })).toBe("");
    expect(toSearch({ ...DEFAULT_STATE, view: "graph" })).toBe("?view=graph");
  });

  it("writes channels under their own names", () => {
    const search = toSearch({
      ...DEFAULT_STATE,
      view: "graph",
      encoding: { x: "minutes", y: "total_points", color: null, wrap: null, aggregate: "mean" },
    });
    expect(search).toContain("x=minutes");
    expect(search).toContain("y=total_points");
    // `mean` is the default and an empty builder should not carry it.
    expect(search).not.toContain("agg=");
  });

  it("carries the element filter §5.5.4's bridge sets", () => {
    // "Explain this" opens the builder with the player pre-filtered, and
    // that has to survive into the link like everything else.
    const search = toSearch({
      ...DEFAULT_STATE,
      filters: { ...DEFAULT_STATE.filters, elements: [430, 64] },
    });
    expect(search).toContain("el=430%2C64");
    expect(parseUrl(search).filters.elements).toEqual([430, 64]);
  });

  it("carries a non-default aggregate once a channel is filled", () => {
    const search = toSearch({
      ...DEFAULT_STATE,
      encoding: { x: "minutes", y: null, color: null, wrap: null, aggregate: "sum" },
    });
    expect(search).toContain("agg=sum");
  });
});

describe("URLs from other builds", () => {
  it("falls back to the hero surface for an unknown view", () => {
    expect(parseUrl("?view=nonesuch").view).toBe("correlations");
  });

  it("ignores an unknown aggregate rather than failing", () => {
    // A URL from a newer build must degrade to a usable view. Accepting
    // the name would put a string §5.6.2 does not permit into a query.
    expect(parseUrl("?agg=loess").encoding.aggregate).toBe("mean");
  });

  it("drops parameters it does not recognise", () => {
    expect(parseUrl("?view=graph&overlay=xg_per90")).toEqual({
      ...DEFAULT_STATE,
      view: "graph",
    });
  });

  it("ignores non-numeric ranges and selections", () => {
    const state = parseUrl("?gw=a-b&sel=x,7,y&mins=nope");
    expect([state.filters.gwMin, state.filters.gwMax]).toEqual([null, null]);
    expect(state.selection).toEqual([7]);
    expect(state.filters.minutesFloor).toBe(null);
  });

  it("does not resurrect a channel that no longer exists", () => {
    // `size` was never one of §5.4.2's four zones. A URL naming it must
    // not create a fifth.
    const state = parseUrl("?x=minutes&size=value");
    expect(Object.keys(state.encoding).sort()).toEqual([
      "aggregate",
      "color",
      "wrap",
      "x",
      "y",
    ]);
  });
});
