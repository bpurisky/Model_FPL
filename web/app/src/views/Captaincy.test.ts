import { describe, expect, it } from "vitest";
import { CaptaincyFile } from "../data/schema";
import type { CaptaincyGw, CaptaincyTrackRecord } from "../data/schema";
import {
  calibrationLine,
  formatOpponents,
  haulFill,
  pickVersusMostOwned,
  recordHeadline,
} from "./Captaincy";

const gw = (over: Partial<CaptaincyGw>): CaptaincyGw => ({
  gw: 8,
  pick: { element_id: 1, name: "Haaland", team: "Man City", xp: 7, p_haul: 0.2 },
  pick_points: 6,
  pick_points_x2: 12,
  pick_hauled: false,
  most_owned: { element_id: 2, name: "Salah", ownership: 40 },
  most_owned_points: 5,
  most_owned_points_x2: 10,
  best_of_shortlist: { element_id: 3, name: "Saka", ownership: null },
  best_of_shortlist_points: 12,
  best_of_shortlist_points_x2: 24,
  shortlist_size: 5,
  ...over,
});

const record = (over: Partial<CaptaincyTrackRecord>): CaptaincyTrackRecord => ({
  n_gameweeks: 0,
  mean_pick_points_x2: null,
  mean_most_owned_points_x2: null,
  mean_best_of_shortlist_points_x2: null,
  pick_haul_rate: null,
  calibration: { n: 0, brier: null, brier_climatology: null, mean_p_haul: null, haul_rate: null },
  gameweeks: [],
  skipped: [],
  ...over,
});

describe("formatOpponents", () => {
  it("shows home and away, sums a double and names a blank", () => {
    expect(formatOpponents([{ opponent: "ARS", home: true, difficulty: 4 }])).toBe("ARS (H)");
    expect(
      formatOpponents([
        { opponent: "ARS", home: true, difficulty: 4 },
        { opponent: "MCI", home: false, difficulty: 5 },
      ]),
    ).toBe("ARS (H) + MCI (A)");
    expect(formatOpponents([])).toBe("no fixture");
  });
});

describe("haulFill", () => {
  it("scales to a ceiling and clamps", () => {
    expect(haulFill(0.2)).toBeCloseTo(0.5);
    expect(haulFill(0.9)).toBe(1);
    expect(haulFill(-1)).toBe(0);
    expect(haulFill(Number.NaN)).toBe(0);
  });
});

describe("the record headline", () => {
  it("says nothing is scored yet rather than inventing a number", () => {
    expect(recordHeadline(record({}))).toMatch(/No pick has been scored yet/);
  });

  it("compares the pick with both baselines once something is scored", () => {
    const text = recordHeadline(
      record({
        n_gameweeks: 3,
        mean_pick_points_x2: 14.0,
        mean_most_owned_points_x2: 11.3,
        mean_best_of_shortlist_points_x2: 22,
      }),
    );
    expect(text).toContain("3 scored gameweeks");
    expect(text).toContain("14.0");
    expect(text).toContain("11.3");
    expect(text).toContain("22.0");
  });
});

describe("calibrationLine", () => {
  it("is absent until something is scored", () => {
    expect(calibrationLine(record({}))).toBeNull();
  });

  it("says when the Brier score does not beat the base rate", () => {
    const base = { n: 100, mean_p_haul: 0.1, haul_rate: 0.12 };
    const worse = calibrationLine(record({ calibration: { ...base, brier: 0.12, brier_climatology: 0.1 } }));
    const better = calibrationLine(record({ calibration: { ...base, brier: 0.09, brier_climatology: 0.1 } }));
    expect(worse).toContain("no better than");
    expect(better).toContain("better than");
    expect(better).not.toContain("no better");
  });
});

describe("pickVersusMostOwned", () => {
  it("counts wins, ties and losses and ignores a missing baseline", () => {
    const rows = [
      gw({ pick_points_x2: 12, most_owned_points_x2: 10 }),
      gw({ pick_points_x2: 4, most_owned_points_x2: 4 }),
      gw({ pick_points_x2: 2, most_owned_points_x2: 18 }),
      gw({ pick_points_x2: 8, most_owned_points_x2: null }),
    ];
    expect(pickVersusMostOwned(rows)).toEqual({ won: 1, tied: 1, lost: 1 });
  });
});

describe("the contract", () => {
  it("accepts the committed shape, including an empty record", () => {
    const parsed = CaptaincyFile.safeParse({
      header: {
        contract_version: 1,
        generated_at: "2026-10-21T00:00:00Z",
        source_gameweek: 7,
        scoring_config: "scoring_2026_27.yaml",
        model_git_sha: null,
        normalization_basis: "within_position_season_to_date",
        rows: 1,
        current_season: "2026-27",
      },
      season: "2026-27",
      gameweek: 8,
      haul_threshold: 10,
      method: "m",
      candidates: [
        {
          element_id: 1,
          name: "Haaland",
          team: "Man City",
          position: "FWD",
          opponents: [{ opponent: "AVL", home: false, difficulty: 3 }],
          xp: 6.5,
          p_haul: 0.2,
          p_low: 0.1,
          ownership: null,
        },
      ],
      track_record: record({}),
    });
    expect(parsed.success).toBe(true);
  });
});
