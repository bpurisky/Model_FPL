import { describe, expect, it } from "vitest";
import { criterionTone } from "./PaperTradeResults";

/**
 * §6.5's five criteria report five distinct status strings, not a
 * boolean. `criterionTone` collapses them to the three badges the launch
 * gate panel renders — this is the classification the rest of that panel
 * hangs off, so it is tested directly rather than through the DOM.
 */

describe("criterion tone", () => {
  it("calls PASS a pass", () => {
    expect(criterionTone("PASS")).toBe("pass");
  });

  it("calls FAIL a fail", () => {
    expect(criterionTone("FAIL")).toBe("fail");
  });

  it("calls insufficient data pending", () => {
    expect(criterionTone("insufficient data")).toBe("pending");
  });

  it("calls not tracked pending", () => {
    expect(criterionTone("not tracked")).toBe("pending");
  });

  it("calls not wired to live baselines yet pending", () => {
    // The one status string that isn't "we don't have enough evidence" --
    // it's "this comparison doesn't exist yet" -- but it still isn't a
    // verdict, so it collapses to the same bucket as the other two.
    expect(criterionTone("not wired to live baselines yet")).toBe("pending");
  });

  it("distinguishes all three tones across the five real status strings", () => {
    const tones = [
      criterionTone("PASS"),
      criterionTone("FAIL"),
      criterionTone("insufficient data"),
      criterionTone("not tracked"),
      criterionTone("not wired to live baselines yet"),
    ];
    expect(new Set(tones).size).toBe(3);
  });
});
