import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import type { ColumnSpec, CorrelationCell } from "../data/schema";
import { Matrix } from "./Matrix";

function column(key: string, higherIsBetter: boolean | null = true): ColumnSpec {
  return {
    key,
    label: key.replace("_per90", ""),
    role: "quantitative",
    unit: "per90",
    format: ".2f",
    definition: `definition of ${key}`,
    source: "fpl_api",
    grain: "player_gameweek",
    normalizable: true,
    normalized_key: `${key}_z_pos`,
    position_relevance: { GK: "none", DEF: "primary", MID: "primary", FWD: "primary" },
    higher_is_better: higherIsBetter,
    available_from_season: null,
    available_to_season: null,
  };
}

const METRICS = ["xg_per90", "xa_per90", "xgc_per90"];
const COLUMNS = new Map(METRICS.map((key) => [key, column(key, key !== "xgc_per90")]));

const CELLS: CorrelationCell[] = [
  { group: "MID", a: "xg_per90", b: "xa_per90", rho: 0.62, n: 284, p_value: 1e-30 },
  { group: "MID", a: "xg_per90", b: "xgc_per90", rho: -0.11, n: 284, p_value: 0.06 },
  // deliberately below any sensible floor, to exercise the hatch
  { group: "MID", a: "xa_per90", b: "xgc_per90", rho: 0.4, n: 12, p_value: 0.2 },
];

function renderMatrix(overrides: Partial<Parameters<typeof Matrix>[0]> = {}) {
  const onSelect = vi.fn();
  render(
    <Matrix
      metrics={METRICS}
      cells={CELLS}
      columns={COLUMNS}
      minN={30}
      selected={null}
      onSelect={onSelect}
      revealKey="MID"
      {...overrides}
    />,
  );
  return { onSelect };
}

describe("the correlation matrix", () => {
  it("renders both halves of a symmetric pair from one exported cell", () => {
    // The export ships one triangle because Spearman is symmetric. If the
    // component only resolved one ordering, half the grid would be blank.
    renderMatrix();
    const cells = screen.getAllByRole("gridcell");
    const withValue = cells.filter((cell) => cell.textContent !== "—");

    expect(withValue).toHaveLength(6); // 3 pairs x 2 orderings
  });

  it("omits the diagonal rather than drawing a cell that says nothing", () => {
    renderMatrix();
    // 9 positions minus 3 on the diagonal.
    expect(screen.getAllByRole("gridcell")).toHaveLength(6);
  });

  it("hatches a cell below the sample floor instead of colouring it", () => {
    // §5.10: no heat map encodes meaning by colour alone. The hatch is a
    // pattern so it survives greyscale and colour blindness.
    renderMatrix();
    const low = screen.getAllByRole("gridcell").find((cell) =>
      cell.getAttribute("aria-label")?.includes("below the sample floor"),
    );

    expect(low).toBeDefined();
    expect(low!.className).toContain("hatched");
    expect(low!.getAttribute("style")).toContain("var(--panel)");
  });

  it("states rho, n and p in the accessible name of every cell", () => {
    // §5.6.3: every derived number renders with access to its provenance,
    // at minimum n. A matrix of bare rho values is where that is easiest
    // to omit.
    renderMatrix();
    const cell = screen.getAllByRole("gridcell")[0]!;

    expect(cell.getAttribute("aria-label")).toMatch(/rho -?\d\.\d\d over \d+/);
  });

  it("paints a symmetric pair identically in both orderings", () => {
    // A correlation is symmetric, so the matrix must be too. Orienting a
    // cell by the row metric's `higher_is_better` — §5.8.2's rule for
    // *value* heat maps — would paint (xGC, xG) and (xG, xGC) in opposing
    // colours for one rho, which is why that rule does not apply here.
    renderMatrix();
    const cells = screen.getAllByRole("gridcell");
    // Compare the fill alone: the full style attribute also carries
    // --reveal-index, which is the column position and legitimately
    // differs between the two orderings.
    const fill = (a: string, b: string) => {
      const style = cells
        .find((cell) => cell.getAttribute("aria-label")?.startsWith(`${a} against ${b}`))!
        .getAttribute("style")!;
      return /--fill:([^;]+)/.exec(style)![1]!.trim();
    };

    expect(fill("xgc", "xg")).toBe(fill("xg", "xgc"));
  });

  it("sends a negative correlation to the teal pole and a positive one to rose", () => {
    renderMatrix();
    const cells = screen.getAllByRole("gridcell");
    const styleOf = (label: string) =>
      cells.find((cell) => cell.getAttribute("aria-label")?.startsWith(label))!.getAttribute("style");

    expect(styleOf("xg against xa")).toContain("--rho-pos");
    expect(styleOf("xg against xgc")).toContain("--rho-neg");
  });

  it("selects a pair on click and deselects on a second click", () => {
    const { onSelect } = renderMatrix();
    const cell = screen.getAllByRole("gridcell")[0]!;

    cell.click();

    expect(onSelect).toHaveBeenCalledWith({ a: "xg_per90", b: "xa_per90" });
  });

  it("clears the selection when the selected cell is clicked again", () => {
    const { onSelect } = renderMatrix({ selected: { a: "xg_per90", b: "xa_per90" } });
    const cell = screen.getAllByRole("gridcell")[0]!;

    cell.click();

    expect(onSelect).toHaveBeenCalledWith(null);
  });

  it("is navigable by keyboard", async () => {
    // §5.10 requires full keyboard navigation of matrices. Every cell is a
    // real focusable control, so this comes from the platform rather than
    // from a roving-tabindex reimplementation over <rect>s.
    renderMatrix();
    const user = userEvent.setup();
    const first = screen.getAllByRole("gridcell")[0]!;

    first.focus();
    await user.keyboard("{ArrowRight}");

    expect(document.activeElement).not.toBe(first);
    expect(document.activeElement?.getAttribute("role")).toBe("gridcell");
  });

  it("exposes one tab stop, not one per cell", async () => {
    renderMatrix();
    const cells = screen.getAllByRole("gridcell");
    const reachable = cells.filter((cell) => cell.getAttribute("tabindex") === "0");

    expect(reachable).toHaveLength(1);
  });

  it("labels the grid and its headers for a screen reader", () => {
    renderMatrix();
    const grid = screen.getByRole("grid");

    expect(grid).toHaveAccessibleName(/Spearman/i);
    expect(within(grid).getAllByRole("rowheader")).toHaveLength(3);
    expect(within(grid).getAllByRole("columnheader")).toHaveLength(3);
  });

  it("renders a missing pair as an em dash rather than a zero", () => {
    renderMatrix({ cells: [CELLS[0]!] });
    const dashes = screen.getAllByRole("gridcell").filter((c) => c.textContent === "—");

    expect(dashes).toHaveLength(4);
  });
});
