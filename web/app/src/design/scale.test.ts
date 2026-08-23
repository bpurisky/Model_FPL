import { describe, expect, it } from "vitest";
import { cellTextColor, divergingColor, formatRho, legendStops } from "./scale";

describe("the diverging scale", () => {
  it("interpolates in oklch, not sRGB", () => {
    // §5.8.8: sRGB interpolation between teal and rose passes through a
    // desaturated grey-mud midpoint that makes near-zero correlations look
    // like rendering artifacts. This is the assertion that keeps the
    // implementation honest if someone reaches for a hex ramp.
    expect(divergingColor(0.5)).toContain("in oklch");
  });

  it("sends the poles to the tokens rather than to literal colours", () => {
    expect(divergingColor(0.9)).toContain("var(--rho-pos)");
    expect(divergingColor(-0.9)).toContain("var(--rho-neg)");
  });

  it("mixes toward the panel so a zero correlation reads as absence", () => {
    // Zero correlation is the absence of a relationship and should read as
    // absence, not as a colour someone chose.
    expect(divergingColor(0)).toContain("0%");
    expect(divergingColor(0)).toContain("var(--panel)");
  });

  it("scales magnitude monotonically", () => {
    const percent = (value: number) => Number(/(\d+)%/.exec(divergingColor(value))![1]);
    expect(percent(0.25)).toBeLessThan(percent(0.5));
    expect(percent(0.5)).toBeLessThan(percent(1));
    expect(percent(1)).toBe(100);
  });

  it("swaps the poles where low is better", () => {
    // §5.8.2's v2 extension: one scale meaning "far from the middle"
    // everywhere, rather than meaning "good" in one view and "bad" in
    // another. xGC is a stat to be low on.
    expect(divergingColor(0.6, "lower_is_better")).toContain("var(--rho-neg)");
    expect(divergingColor(0.6, "higher_is_better")).toContain("var(--rho-pos)");
  });

  it("renders a null as transparent rather than as a midpoint colour", () => {
    // A cell with no correlation defined must not be paintable as though
    // it measured zero (§5.3.3).
    expect(divergingColor(null)).toBe("transparent");
  });

  it("gives a legend an odd number of stops so zero has a true midpoint", () => {
    const stops = legendStops();
    expect(stops.length % 2).toBe(1);
    expect(stops[(stops.length - 1) / 2]!.value).toBeCloseTo(0);
  });
});

describe("formatRho", () => {
  it("always signs and always shows two decimals so a column aligns", () => {
    // §5.8.3 calls proportional figures in a stats table a defect rather
    // than a preference; ragged decimals are the same failure.
    expect(formatRho(0.7)).toBe("+0.70");
    expect(formatRho(-0.65)).toBe("−0.65");
    expect(formatRho(0)).toBe("+0.00");
  });

  it("renders a null as an em dash, never as zero", () => {
    // "No correlation is defined here" and "the correlation measured zero"
    // are different statements (§5.3.3).
    expect(formatRho(null)).toBe("—");
    expect(formatRho(Number.NaN)).toBe("—");
  });
});

describe("cellTextColor", () => {
  it("flips to the ground colour once the pole dominates the cell", () => {
    expect(cellTextColor(0.2)).toBe("var(--paper)");
    expect(cellTextColor(0.9)).toBe("var(--ground)");
    expect(cellTextColor(-0.9)).toBe("var(--ground)");
  });
});
