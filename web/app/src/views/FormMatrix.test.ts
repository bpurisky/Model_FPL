import { describe, expect, it } from "vitest";
import { cellState } from "./FormMatrix";

/**
 * §5.4.3's load-bearing rule, and §5.14.9's acceptance criterion:
 *
 * > "Blank vs zero vs did-not-play are three visually distinct states. A
 * > player who played 90 minutes and scored zero points and a player who
 * > was not in the squad are not the same fact and must not share a cell
 * > treatment."
 *
 * The states are what the rest of the surface hangs off — the class it
 * renders, the tooltip it writes, whether the diverging scale touches the
 * cell at all — so they are tested here rather than through the DOM.
 */

const cell = (minutes: number, value: number | null, fixtures = 1) => ({
  minutes,
  value,
  fixtures,
});

describe("the four cell states", () => {
  it("calls a missing row blank", () => {
    // No row: the club had no fixture, or he was not in the squad.
    // Nothing happened to measure.
    expect(cellState(undefined)).toBe("blank");
  });

  it("calls a fixture with no minutes did-not-play", () => {
    // Something happened and he was not part of it, which is a fact
    // about the player rather than about the fixture list.
    expect(cellState(cell(0, 0))).toBe("dnp");
    expect(cellState(cell(0, null))).toBe("dnp");
  });

  it("calls minutes with a null metric absent, not zero", () => {
    // §5.3.3: most often a rate below the eligibility floor. Unknown is
    // not a small number.
    expect(cellState(cell(90, null))).toBe("absent");
  });

  it("calls a real zero a value", () => {
    // The case the spec names explicitly: 90 minutes and zero points is
    // a measurement, and it must not look like an absence.
    expect(cellState(cell(90, 0))).toBe("value");
  });

  it("distinguishes all five from one another", () => {
    const states = [
      cellState(undefined),
      cellState(undefined, false),
      cellState(cell(0, 0)),
      cellState(cell(90, null)),
      cellState(cell(90, 0)),
    ];
    expect(new Set(states).size).toBe(5);
  });

  it("tells a blank gameweek from a player left out, when a schedule says so", () => {
    /*
     * The distinction §5.4.3 asks for and the panel alone cannot make: a
     * missing row is either the club having no fixture or the player not
     * being in the squad, and a reader draws opposite conclusions from
     * the two. Only the schedule knows which.
     */
    expect(cellState(undefined, false)).toBe("noFixture");
    expect(cellState(undefined, true)).toBe("blank");
  });

  it("falls back to an undifferentiated blank with no schedule", () => {
    // The archive has no schedule loaded, and inventing one would be
    // worse than admitting the two cases cannot be separated.
    expect(cellState(undefined, null)).toBe("blank");
    expect(cellState(undefined)).toBe("blank");
  });

  it("never lets the schedule override a row that exists", () => {
    // A club marked blank while the player has a row is a contradiction
    // in the data, and the row is the stronger evidence: something was
    // measured.
    expect(cellState(cell(90, 3), false)).toBe("value");
    expect(cellState(cell(0, null), false)).toBe("dnp");
  });

  it("never reports a zero-minute row as a value, whatever the metric says", () => {
    // A per-90 rate carried on a row with no minutes would be a division
    // by zero upstream; if one ever arrives, it is still not a
    // performance.
    for (const value of [0, 1.5, -2, null]) {
      expect(cellState(cell(0, value))).toBe("dnp");
    }
  });

  it("treats a negative score as a value, because it is one", () => {
    // Red cards and own goals produce negative points, and they are the
    // most informative cells on the row.
    expect(cellState(cell(90, -1))).toBe("value");
  });
});
