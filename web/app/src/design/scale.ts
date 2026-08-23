/**
 * The diverging scale (§5.8.2, §5.8.8).
 *
 * "The diverging scale is the brand — the accent colors are not decoration
 * applied to charts, they are the chart's own encoding promoted to
 * identity." So this is the one piece of visual logic that is genuinely
 * load-bearing, and it is computed rather than picked from a ramp of
 * hand-chosen stops.
 *
 * Interpolation happens in OKLCH via `color-mix`, which is what keeps
 * perceived lightness constant across the ramp: sRGB interpolation
 * between teal and rose passes through a desaturated grey-mud midpoint
 * that makes near-zero correlations look like rendering artifacts rather
 * than like the finding they are.
 *
 * The midpoint is `--panel` rather than a neutral grey, so a cell at rho
 * zero disappears into the surface it sits on. That is deliberate: zero
 * correlation is the absence of a relationship, and it should read as
 * absence rather than as a colour someone chose.
 */

/** The value a cell's magnitude is normalized against. */
export const SCALE_MAX = 1;

export type Direction = "higher_is_better" | "lower_is_better" | "neutral";

/**
 * A CSS colour for a correlation.
 *
 * `direction` swaps the poles for a metric where low is good (§5.8.2's v2
 * extension), so one scale means "far from the middle" everywhere in the
 * app rather than meaning "good" in one view and "bad" in another. The
 * legend states the direction explicitly; the colour never has to be
 * guessed at.
 */
export function divergingColor(value: number | null, direction: Direction = "neutral"): string {
  if (value === null || Number.isNaN(value)) return "transparent";

  const oriented = direction === "lower_is_better" ? -value : value;
  const magnitude = Math.min(Math.abs(oriented) / SCALE_MAX, 1);
  const pole = oriented >= 0 ? "var(--rho-pos)" : "var(--rho-neg)";

  // `color-mix` in oklch is the interpolation; the percentage is the
  // magnitude. Rounded to whole percent so identical values produce
  // byte-identical colours and the browser can cache the parse.
  return `color-mix(in oklch, ${pole} ${Math.round(magnitude * 100)}%, var(--panel))`;
}

/**
 * Text colour for a cell, chosen so the numeral stays legible against
 * whatever the cell became. Below roughly half magnitude the cell is
 * closer to `--panel` than to a pole, so the normal paper colour reads;
 * above it the pole dominates and the numeral needs the ground behind it.
 */
export function cellTextColor(value: number | null): string {
  if (value === null || Number.isNaN(value)) return "var(--muted)";
  return Math.abs(value) > 0.55 ? "var(--ground)" : "var(--paper)";
}

/**
 * Stops for the legend, from one pole to the other. Odd count so there is
 * a true midpoint rather than a seam where zero should be.
 */
export function legendStops(count = 9): { value: number; color: string }[] {
  return Array.from({ length: count }, (_, index) => {
    const value = -SCALE_MAX + (2 * SCALE_MAX * index) / (count - 1);
    return { value, color: divergingColor(value) };
  });
}

/**
 * Formats a correlation for display. Always signed, always two decimals,
 * because a column of `.7` next to `-.65` is a column that does not align
 * on the decimal — which §5.8.3 calls a defect rather than a preference.
 *
 * A null renders as an em dash, never as `0.00`: no correlation being
 * defined is a different statement from one that measured zero.
 */
export function formatRho(value: number | null): string {
  if (value === null || Number.isNaN(value)) return "—";
  return (value < 0 ? "−" : "+") + Math.abs(value).toFixed(2);
}
