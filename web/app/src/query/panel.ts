/**
 * The named query helpers (§5.1.2).
 *
 * > "The view layer never touches a raw file path, never sees an
 * > unvalidated payload, and never writes SQL directly — it calls named
 * > helpers in `query/`."
 *
 * So this is the only module in `src/` that composes SQL, and it composes
 * it under two rules that are not negotiable:
 *
 * **Every identifier is checked against `columns.json` before it reaches
 * a query string.** DuckDB-WASM runs in the user's own browser over a
 * static file, so the stakes are lower than a server — but the app builds
 * SQL out of state that comes back from a URL (§5.5), and a URL is
 * user-supplied input whoever typed it. `column()` below refuses anything
 * the registry does not name, which also catches the ordinary bug of a
 * renamed column reaching the builder as a silent empty result.
 *
 * **No reduction is expressed in SQL.** The queries here group and
 * collect (`list(...)`), and `reduce.ts` does the arithmetic. That is a
 * deliberate cost — moving arrays out of the engine rather than scalars
 * — paid so that §5.6.2's seven operations have one implementation, in
 * TypeScript, covered by `reduce.golden.test.ts` against Python. An
 * `avg()` in a query string here would be a second implementation that
 * no test in CI ever runs.
 */

import type { ColumnSpec } from "../data/schema";
import type { Session } from "./session";

/** §5.4.2's global filter bar, as state. Empty arrays mean "no filter". */
export interface PanelFilters {
  seasons: string[];
  positions: string[];
  teams: string[];
  /** Tenths of a million, matching the panel's own `value` column. */
  priceMin: number | null;
  priceMax: number | null;
  /** Season-to-date minutes a player must have reached in the row. */
  minutesFloor: number | null;
  gwMin: number | null;
  gwMax: number | null;
}

export const NO_FILTERS: PanelFilters = {
  seasons: [],
  positions: [],
  teams: [],
  priceMin: null,
  priceMax: null,
  minutesFloor: null,
  gwMin: null,
  gwMax: null,
};

/** One group, with the raw values behind it for `reduce.ts` to reduce. */
export interface GroupedRow {
  key: Record<string, string | number | null>;
  values: Record<string, (number | null)[]>;
}

export interface GroupedQuery {
  /** Registry keys to group by. `element_id` implies the player grain. */
  groupBy: string[];
  /** Registry keys to collect. Reduced by the caller, never here. */
  collect: string[];
  filters: PanelFilters;
  /** Read `{key}_z_pos` in place of `{key}` where the registry offers it. */
  normalized: boolean;
  /**
   * Refuse to return more than this many groups. A categorical channel
   * carrying `name` would otherwise ask for 800 bars, which is not a
   * chart. The builder reports the cap rather than truncating silently.
   */
  limit?: number;
}

export class QueryError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "QueryError";
  }
}

/** The default group cap. Above this, a mark stops being readable. */
export const GROUP_LIMIT = 2000;

/**
 * Every name that may appear as an identifier in a query.
 *
 * The registry's own keys, **plus the companion columns they declare**.
 * §5.7.2's `{key}_z_pos` columns are deliberately not separate registry
 * entries — `columns.py:companion_keys` derives them by convention and
 * each base column points at its own through `normalized_key`, which
 * `test_normalizable_columns_declare_their_companion_key` enforces. So
 * the allowlist has to be built the same way the export builds them, or
 * the §5.7.3 toggle resolves to a column the query is not allowed to
 * name.
 */
function allowedIdentifiers(columns: ReadonlyMap<string, ColumnSpec>): Set<string> {
  const allowed = new Set<string>();
  for (const spec of columns.values()) {
    allowed.add(spec.key);
    if (spec.normalized_key) allowed.add(spec.normalized_key);
  }
  return allowed;
}

/**
 * A registry-checked, quoted identifier.
 *
 * The allowlist comes from the registry, so a name that is not a real
 * exported column cannot reach a query string — which covers both the
 * injection case and the far likelier one of a stale URL naming a column
 * that was renamed upstream.
 */
function column(key: string, columns: ReadonlyMap<string, ColumnSpec>): string {
  if (!allowedIdentifiers(columns).has(key)) {
    throw new QueryError(`${key} is not a column in the registry`);
  }
  // Belt and braces: the registry is committed data, but a quoted
  // identifier containing a quote would still be a way out of the quotes.
  if (!/^[a-z_][a-z0-9_]*$/i.test(key)) {
    throw new QueryError(`${key} is not a usable SQL identifier`);
  }
  return `"${key}"`;
}

/**
 * The column to actually read for a key, honouring the §5.7.3 toggle.
 *
 * A key with no `normalized_key` reads raw whatever the toggle says —
 * `minutes` has no within-position z-score and inventing one in the
 * browser is exactly what §5.6 forbids. The caller is told which columns
 * were normalized so the surface can state its basis (§5.7.4) rather
 * than implying every number on screen moved.
 */
export function resolveColumn(
  key: string,
  columns: ReadonlyMap<string, ColumnSpec>,
  normalized: boolean,
): { key: string; normalized: boolean } {
  const spec = columns.get(key);
  if (!spec) throw new QueryError(`${key} is not a column in the registry`);
  if (normalized && spec.normalizable && spec.normalized_key) {
    return { key: spec.normalized_key, normalized: true };
  }
  return { key: spec.key, normalized: false };
}

/** A SQL string literal. Values reaching here come from the URL. */
function literal(value: string): string {
  return `'${value.replace(/'/g, "''")}'`;
}

function integer(value: number, what: string): string {
  if (!Number.isFinite(value)) throw new QueryError(`${what} is not a number`);
  return String(Math.trunc(value));
}

/** §5.4.2's filter bar, as a WHERE clause. */
export function wherePredicates(filters: PanelFilters): string[] {
  const clauses: string[] = [];

  const inList = (col: string, values: string[]) => {
    if (values.length === 0) return;
    clauses.push(`"${col}" IN (${values.map(literal).join(", ")})`);
  };

  inList("season", filters.seasons);
  inList("position", filters.positions);
  inList("team", filters.teams);

  if (filters.priceMin !== null) clauses.push(`"value" >= ${integer(filters.priceMin, "priceMin")}`);
  if (filters.priceMax !== null) clauses.push(`"value" <= ${integer(filters.priceMax, "priceMax")}`);
  if (filters.gwMin !== null) clauses.push(`"gw" >= ${integer(filters.gwMin, "gwMin")}`);
  if (filters.gwMax !== null) clauses.push(`"gw" <= ${integer(filters.gwMax, "gwMax")}`);
  if (filters.minutesFloor !== null) {
    // `cum_minutes` rather than `minutes`: the filter means "players who
    // have played this much", not "gameweeks in which they played this
    // much", and the second reading would drop every rotation week from a
    // regular starter's series and leave a chart of their best days.
    clauses.push(`"cum_minutes" >= ${integer(filters.minutesFloor, "minutesFloor")}`);
  }

  return clauses;
}

/**
 * Group the panel and hand back the values behind each group.
 *
 * `list(...)` rather than an aggregate is the whole point — see the
 * module docstring. The arrays are per-group and the groups are capped,
 * so what crosses the boundary is bounded by the chart's own readability
 * rather than by the panel's 85,000 rows.
 */
export async function grouped(
  session: Session,
  query: GroupedQuery,
  columns: ReadonlyMap<string, ColumnSpec>,
): Promise<{ rows: GroupedRow[]; truncated: boolean; normalizedKeys: string[] }> {
  if (query.groupBy.length === 0) {
    throw new QueryError("a grouped query needs at least one group key");
  }

  const limit = query.limit ?? GROUP_LIMIT;
  const normalizedKeys: string[] = [];

  const keySelect = query.groupBy.map((key) => `${column(key, columns)} AS ${column(key, columns)}`);

  const valueSelect = query.collect.map((key) => {
    const resolved = resolveColumn(key, columns, query.normalized);
    if (resolved.normalized) normalizedKeys.push(key);
    // Aliased back to the *requested* key so the caller reads its own
    // vocabulary and does not have to know whether the toggle was on.
    return `list(${column(resolved.key, columns)}) AS "${key}"`;
  });

  const where = wherePredicates(query.filters);
  const sql = [
    `SELECT ${[...keySelect, ...valueSelect].join(", ")}`,
    `FROM panel`,
    where.length > 0 ? `WHERE ${where.join(" AND ")}` : "",
    `GROUP BY ${query.groupBy.map((key) => column(key, columns)).join(", ")}`,
    `ORDER BY ${query.groupBy.map((key) => column(key, columns)).join(", ")}`,
    // One over the cap, so the caller can tell "exactly at the limit"
    // from "more than we will draw".
    `LIMIT ${limit + 1}`,
  ]
    .filter(Boolean)
    .join(" ");

  const raw = await session.run(sql);
  const truncated = raw.length > limit;
  const rows = (truncated ? raw.slice(0, limit) : raw).map((row) => {
    const key: GroupedRow["key"] = {};
    for (const name of query.groupBy) key[name] = normalizeScalar(row[name]);

    const values: GroupedRow["values"] = {};
    for (const name of query.collect) values[name] = normalizeList(row[name]);

    return { key, values };
  });

  return { rows, truncated, normalizedKeys };
}

/**
 * What the filter bar offers, read from the data rather than hard-coded.
 *
 * A team list in the source would be wrong three times a season on
 * promotion and relegation, and a gameweek range in the source would be
 * wrong every week.
 */
export interface PanelFacets {
  seasons: string[];
  teams: string[];
  positions: string[];
  gwMin: number;
  gwMax: number;
  priceMin: number;
  priceMax: number;
  rows: number;
}

export async function facets(session: Session): Promise<PanelFacets> {
  const [seasons, teams, positions, extent] = await Promise.all([
    session.run(`SELECT DISTINCT "season" AS v FROM panel ORDER BY v`),
    session.run(`SELECT DISTINCT "team" AS v FROM panel ORDER BY v`),
    session.run(`SELECT DISTINCT "position" AS v FROM panel ORDER BY v`),
    session.run(
      `SELECT min("gw") AS gw_min, max("gw") AS gw_max, ` +
        `min("value") AS price_min, max("value") AS price_max, count(*) AS rows FROM panel`,
    ),
  ]);

  const bounds = extent[0] ?? {};
  return {
    seasons: seasons.map((row) => String(row["v"])),
    teams: teams.map((row) => String(row["v"])),
    /*
     * Pitch order, not alphabetical. SQL's ORDER BY gives DEF, FWD, GK,
     * MID, which is the order of no football thought anyone has ever had
     * — every squad list, every FPL screen and every table in this repo
     * reads goalkeeper outward.
     */
    positions: positions.map((row) => String(row["v"])).sort(byPitchOrder),
    gwMin: Number(bounds["gw_min"] ?? 1),
    gwMax: Number(bounds["gw_max"] ?? 38),
    priceMin: Number(bounds["price_min"] ?? 0),
    priceMax: Number(bounds["price_max"] ?? 0),
    rows: Number(bounds["rows"] ?? 0),
  };
}

const PITCH_ORDER = ["GK", "DEF", "MID", "FWD"];

function byPitchOrder(left: string, right: string): number {
  const rank = (value: string) => {
    const index = PITCH_ORDER.indexOf(value);
    // An unknown position sorts last rather than first, so a new one
    // appearing upstream is visible instead of silently leading.
    return index === -1 ? PITCH_ORDER.length : index;
  };
  return rank(left) - rank(right) || left.localeCompare(right);
}

/**
 * Arrow hands back its own wrappers, and BigInt for 64-bit integers.
 * Both have to become plain JS before anything downstream compares them.
 */
function normalizeScalar(value: unknown): string | number | null {
  if (value === null || value === undefined) return null;
  if (typeof value === "bigint") return Number(value);
  if (typeof value === "number" || typeof value === "string") return value;
  if (typeof value === "boolean") return String(value);
  return String(value);
}

/**
 * An Arrow list column as plain JS, **by iteration and never by
 * `toArray()`**.
 *
 * This is not a style preference and the difference is not subtle.
 * `Vector.toArray()` hands back the underlying typed array and ignores
 * the validity bitmap, so every null slot reads as whatever was in that
 * memory. Measured against `xg_per90` for one gameweek of 2024-25: 309 of
 * 616 values are null, and `toArray()` returned `2.5465051432e-313` and
 * `1.9e-322` for them — uninitialised memory, silently, as Float64.
 *
 * Those would not have looked like an error. They would have looked like
 * a player with a very small xG, reduced into a mean, and drawn. Which is
 * §5.3.3's whole point arriving from an unexpected direction: "a null
 * z-score is 'this player is below the minutes floor'", and there is no
 * number that says that.
 *
 * Iterating the vector reads the bitmap and yields real `null`s. It is
 * slower and it is the only correct option.
 */
function normalizeList(value: unknown): (number | null)[] {
  if (value === null || value === undefined) return [];

  const iterable =
    Array.isArray(value) ? value
    : typeof (value as { [Symbol.iterator]?: unknown })[Symbol.iterator] === "function"
      ? (value as Iterable<unknown>)
      : [];

  const out: (number | null)[] = [];
  for (const entry of iterable) {
    if (entry === null || entry === undefined) {
      out.push(null);
      continue;
    }
    if (typeof entry === "bigint") {
      out.push(Number(entry));
      continue;
    }
    const asNumber = Number(entry);
    out.push(Number.isFinite(asNumber) ? asNumber : null);
  }
  return out;
}
