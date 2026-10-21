import { describe, expect, it } from "vitest";
import type { PriceTrackRecord, PriceWatchPlayer } from "../data/schema";
import {
  barFill,
  filterPlayers,
  NO_PRICE_FILTERS,
  progressPercent,
  reasonLabel,
  trackRecordHeadline,
} from "./PriceWatch";

const player = (over: Partial<PriceWatchPlayer>): PriceWatchPlayer => ({
  element_id: 1,
  name: "Salah",
  team: "Liverpool",
  position: "MID",
  now_cost: 130,
  net_transfers_24h: 1000,
  progress: 0.5,
  eligible: true,
  ineligible_reason: null,
  call: "stable",
  ...over,
});

const record = (over: Partial<PriceTrackRecord>): PriceTrackRecord => ({
  status: "insufficient data",
  n_calls: 7,
  n_cycles: 3,
  min_calls: 20,
  hit_rate: null,
  ci_low: null,
  ci_high: null,
  span: null,
  in_sample: null,
  recent_cycles: [],
  ...over,
});

describe("the honesty headline", () => {
  it("says not enough scored calls yet, with n over the minimum, rather than a rate", () => {
    const headline = trackRecordHeadline(record({ hit_rate: 1.0, n_calls: 7 }));
    expect(headline.scored).toBe(false);
    expect(headline.text).toContain("Not enough scored calls yet (7/20)");
    expect(headline.text).not.toContain("100%");
  });

  it("reports the out-of-sample hit rate with its interval and n once scored", () => {
    const headline = trackRecordHeadline(
      record({ status: "PASS", n_calls: 25, hit_rate: 0.6, ci_low: 0.41, ci_high: 0.79 }),
    );
    expect(headline.scored).toBe(true);
    expect(headline.text).toBe("Out-of-sample hit rate 60% (95% CI 41% to 79%) over 25 rise calls.");
  });
});

describe("progress bar", () => {
  it("clamps to 0..1 and treats unknown as empty", () => {
    expect(barFill(-0.4)).toBe(0);
    expect(barFill(0.37)).toBe(0.37);
    expect(barFill(2.5)).toBe(1);
    expect(barFill(null)).toBe(0);
  });

  it("prints the true percentage even beyond the bar", () => {
    expect(progressPercent(2.5)).toBe("250%");
    expect(progressPercent(-0.4)).toBe("-40%");
    expect(progressPercent(null)).toBe("n/a");
  });
});

describe("reasons", () => {
  it("names the 72h rule and the history gap", () => {
    expect(reasonLabel("moved_within_72h", 72)).toBe("price moved in the last 72h");
    expect(reasonLabel("insufficient_history", 72)).toBe("not enough snapshot history");
    expect(reasonLabel(null, 72)).toBe("");
  });
});

describe("filtering", () => {
  const players = [
    player({ element_id: 1, name: "Salah", team: "Liverpool", position: "MID" }),
    player({ element_id: 2, name: "Haaland", team: "Man City", position: "FWD" }),
    player({ element_id: 3, name: "Foden", team: "Man City", position: "MID" }),
  ];

  it("keeps everything, in order, with no filters", () => {
    expect(filterPlayers(players, NO_PRICE_FILTERS).map((p) => p.element_id)).toEqual([1, 2, 3]);
  });

  it("combines position, team and a case-insensitive name search", () => {
    expect(filterPlayers(players, { ...NO_PRICE_FILTERS, team: "Man City", position: "MID" }).map((p) => p.name)).toEqual(["Foden"]);
    expect(filterPlayers(players, { ...NO_PRICE_FILTERS, query: " HAAL " }).map((p) => p.name)).toEqual(["Haaland"]);
  });
});
