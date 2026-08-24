/**
 * The named query helpers (§5.1.2).
 *
 * > "The view layer never touches a raw file path, never sees an
 * > unvalidated payload, and never writes SQL directly — it calls named
 * > helpers in `query/`."
 *
 * There is no SQL left to write. §5.16 deviation D10 replaced DuckDB-WASM
 * with a parquet reader and this module, because the engine cost 4.6 MB
 * brotli against §5.9's 1.2 MB budget while doing only `WHERE`,
 * `GROUP BY` and `list(...)` — the reductions having already moved to
 * `reduce.ts` so that §5.6.2's seven operations could have one
 * implementation covered by a golden test. What follows is that `WHERE`
 * and `GROUP BY`, over decoded columns.
 *
 * Two rules survive the change unchanged, because they were never about
 * SQL:
 *
 * **Every column name is checked against `columns.json` before it is
 * used.** The app builds queries out of state that comes back from a URL
 * (§5.5), and a URL is user-supplied input whoever typed it. There is no
 * injection surface any more, but the check also catches the ordinary bug
 * of a renamed column reaching the builder as a silently empty result,
 * which is the failure it was really there for.
 *
 * **No reduction happens here.** This groups and collects; `reduce.ts`
 * does the arithmetic. A `mean` computed in this file would be a second
 * implementation that `reduce.golden.test.ts` never runs.
 */

import type { ColumnSpec } from "../data/schema";
import type { PanelColumn, Session } from "./session";

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
  /** Registry keys to group by. */
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

/** The panel columns the filter bar reads, whatever the encoding asks for. */
const FILTER_COLUMNS = ["season", "position", "team", "value", "gw", "cum_minutes"] as const;

/**
 * Every name that may be read as a column.
 *
 * The registry's own keys, **plus the companion columns they declare**.
 * §5.7.2's `{key}_z_pos` columns are deliberately not separate registry
 * entries — `columns.py:companion_keys` derives them by convention and
 * each base column points at its own through `normalized_key`, which
 * `test_normalizable_columns_declare_their_companion_key` enforces. So
 * the allowlist has to be built the same way the export builds them, or
 * the §5.7.3 toggle resolves to a column the query refuses to name.
 */
function allowedNames(columns: ReadonlyMap<string, ColumnSpec>): Set<string> {
  const allowed = new Set<string>(FILTER_COLUMNS);
  for (const spec of columns.values()) {
    allowed.add(spec.key);
    if (spec.normalized_key) allowed.add(spec.normalized_key);
  }
  return allowed;
}

function checkName(key: string, columns: ReadonlyMap<string, ColumnSpec>): string {
  if (!allowedNames(columns).has(key)) {
    throw new QueryError(`${key} is not a column in the registry`);
  }
  return key;
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

function bound(value: number, what: string): number {
  if (!Number.isFinite(value)) throw new QueryError(`${what} is not a number`);
  return Math.trunc(value);
}

/** Which panel columns a set of filters needs in order to be evaluated. */
export function filterColumns(filters: PanelFilters): string[] {
  const needed: string[] = [];
  if (filters.seasons.length) needed.push("season");
  if (filters.positions.length) needed.push("position");
  if (filters.teams.length) needed.push("team");
  if (filters.priceMin !== null || filters.priceMax !== null) needed.push("value");
  if (filters.gwMin !== null || filters.gwMax !== null) needed.push("gw");
  if (filters.minutesFloor !== null) needed.push("cum_minutes");
  return needed;
}

/**
 * §5.4.2's filter bar, as a row mask.
 *
 * A mask rather than a filtered copy: the panel is 85,000 rows and every
 * channel reads its own column, so materialising a filtered frame per
 * query would copy far more than it saves. The mask is computed once and
 * every column is walked through it.
 */
export function rowMask(
  filters: PanelFilters,
  values: ReadonlyMap<string, PanelColumn>,
  rows: number,
): Uint8Array {
  const mask = new Uint8Array(rows).fill(1);

  const restrict = (name: string, keep: (value: unknown) => boolean) => {
    const column = values.get(name);
    if (!column) throw new QueryError(`${name} was not loaded for the filter`);
    for (let i = 0; i < rows; i += 1) {
      if (mask[i] && !keep(column[i])) mask[i] = 0;
    }
  };

  const inSet = (name: string, allowed: string[]) => {
    if (allowed.length === 0) return;
    const set = new Set(allowed);
    restrict(name, (value) => value !== null && set.has(String(value)));
  };

  inSet("season", filters.seasons);
  inSet("position", filters.positions);
  inSet("team", filters.teams);

  if (filters.priceMin !== null) {
    const min = bound(filters.priceMin, "priceMin");
    restrict("value", (value) => typeof value === "number" && value >= min);
  }
  if (filters.priceMax !== null) {
    const max = bound(filters.priceMax, "priceMax");
    restrict("value", (value) => typeof value === "number" && value <= max);
  }
  if (filters.gwMin !== null) {
    const min = bound(filters.gwMin, "gwMin");
    restrict("gw", (value) => typeof value === "number" && value >= min);
  }
  if (filters.gwMax !== null) {
    const max = bound(filters.gwMax, "gwMax");
    restrict("gw", (value) => typeof value === "number" && value <= max);
  }
  if (filters.minutesFloor !== null) {
    const min = bound(filters.minutesFloor, "minutesFloor");
    /*
     * `cum_minutes` rather than `minutes`: the filter means "players who
     * have played this much", not "gameweeks in which they played this
     * much", and the second reading would drop every rotation week from
     * a regular starter's series and leave a chart of their best days.
     */
    restrict("cum_minutes", (value) => typeof value === "number" && value >= min);
  }

  return mask;
}

/**
 * Group the panel and hand back the values behind each group.
 *
 * Collecting the values rather than reducing them is the whole point —
 * see the module docstring. The groups are capped, so what crosses into
 * the view layer is bounded by the chart's own readability rather than by
 * the panel's row count.
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

  // Validate every name *before* any decoding, so a bad key costs
  // nothing and fails with the name in the message.
  const groupNames = query.groupBy.map((key) => checkName(key, columns));
  const collectNames = query.collect.map((key) => {
    const resolved = resolveColumn(key, columns, query.normalized);
    if (resolved.normalized) normalizedKeys.push(key);
    return { requested: key, actual: checkName(resolved.key, columns) };
  });

  const needed = [
    ...new Set([
      ...filterColumns(query.filters),
      ...groupNames,
      ...collectNames.map((entry) => entry.actual),
    ]),
  ];

  const loaded = new Map<string, PanelColumn>();
  await Promise.all(
    needed.map(async (name) => {
      loaded.set(name, await session.column(name));
    }),
  );

  const rows = session.rows;
  const mask = rowMask(query.filters, loaded, rows);

  const groups = new Map<string, GroupedRow>();
  let truncated = false;

  for (let i = 0; i < rows; i += 1) {
    if (!mask[i]) continue;

    // U+001F, a unit separator, so a team called "A|B" cannot collide
    // with the pair ("A", "B").
    let signature = "";
    for (const name of groupNames) {
      signature += `${String(loaded.get(name)![i] ?? "")}`;
    }

    let group = groups.get(signature);
    if (!group) {
      if (groups.size >= limit) {
        truncated = true;
        continue;
      }
      const key: GroupedRow["key"] = {};
      for (const name of groupNames) {
        const value = loaded.get(name)![i];
        key[name] = value === null ? null : typeof value === "number" ? value : String(value);
      }
      const values: GroupedRow["values"] = {};
      for (const entry of collectNames) values[entry.requested] = [];
      group = { key, values };
      groups.set(signature, group);
    }

    for (const entry of collectNames) {
      const value = loaded.get(entry.actual)![i];
      // Nulls are kept, because `reduce.ts` distinguishes "no rows" from
      // "rows with no value" and both are real answers (§5.3.3).
      group.values[entry.requested]!.push(typeof value === "number" ? value : null);
    }
  }

  return { rows: sortGroups([...groups.values()], groupNames), truncated, normalizedKeys };
}

/**
 * Groups in key order, so a chart's categories do not reshuffle when a
 * filter changes. Insertion order here is row order, which is stable but
 * arbitrary; ordering explicitly is what makes the axis reproducible.
 */
function sortGroups(rows: GroupedRow[], groupNames: string[]): GroupedRow[] {
  return rows.sort((left, right) => {
    for (const name of groupNames) {
      const a = left.key[name];
      const b = right.key[name];
      if (a === b) continue;
      if (a === null) return -1;
      if (b === null) return 1;
      if (typeof a === "number" && typeof b === "number") return a - b;
      const compared = String(a).localeCompare(String(b), undefined, { numeric: true });
      if (compared !== 0) return compared;
    }
    return 0;
  });
}

/** Rows that passed the filters, plus the columns to read them from. */
export interface Selection {
  /** Panel row indices that survived the filters, in file order. */
  index: number[];
  /** The requested columns, keyed by the name the caller asked for. */
  values: ReadonlyMap<string, PanelColumn>;
  /** Which requested keys were served by a `_z_pos` companion (§5.7.4). */
  normalizedKeys: string[];
}

/**
 * Filtered rows at panel grain, without grouping.
 *
 * `grouped` is the wrong shape for a surface whose cells *are* the rows:
 * the Form Matrix is one cell per player-gameweek, which is 700 players
 * by 38 gameweeks and would ask `grouped` for 26,000 groups of one value
 * each — all the cost of grouping and none of the point.
 *
 * Returns indices rather than materialised row objects. The caller walks
 * them and reads whichever columns it needs, so nothing allocates 26,000
 * objects to throw most of them away.
 *
 * No reduction happens here either. This is `WHERE` and a projection.
 */
export async function select(
  session: Session,
  query: Omit<GroupedQuery, "groupBy" | "collect"> & { columns: string[] },
  columns: ReadonlyMap<string, ColumnSpec>,
): Promise<Selection> {
  const normalizedKeys: string[] = [];

  // Validated before anything is decoded, so a bad key costs nothing.
  const requested = query.columns.map((key) => {
    const spec = columns.get(key);
    if (!spec) {
      // Structural panel columns the registry does not describe as
      // metrics still have to be readable — `cum_minutes` is how the
      // minutes filter works at all.
      return { requested: key, actual: checkName(key, columns) };
    }
    const resolved = resolveColumn(key, columns, query.normalized);
    if (resolved.normalized) normalizedKeys.push(key);
    return { requested: key, actual: checkName(resolved.key, columns) };
  });

  const needed = [
    ...new Set([...filterColumns(query.filters), ...requested.map((entry) => entry.actual)]),
  ];

  const loaded = new Map<string, PanelColumn>();
  await Promise.all(
    needed.map(async (name) => {
      loaded.set(name, await session.column(name));
    }),
  );

  const mask = rowMask(query.filters, loaded, session.rows);
  const index: number[] = [];
  const limit = query.limit ?? Number.POSITIVE_INFINITY;
  for (let i = 0; i < session.rows && index.length < limit; i += 1) {
    if (mask[i]) index.push(i);
  }

  const values = new Map<string, PanelColumn>();
  for (const entry of requested) values.set(entry.requested, loaded.get(entry.actual)!);

  return { index, values, normalizedKeys };
}

/**
 * What the filter bar offers, read from the data rather than hard-coded.
 *
 * A team list in the source would be wrong three times a season on
 * promotion and relegation, and a gameweek range in the source would be
 * wrong every week.
 */
/**
 * One season the filter bar can offer, and what is actually behind it.
 *
 * `rows: 0` is a real and expected state, not a bug. The panel can only
 * report the seasons it carries, and the season everyone actually cares
 * about is the one that has barely started — invisible to the panel until
 * its first gameweek is recorded. So the current season is unioned in
 * from the export header (§5.3.1's `current_season`) whether or not it has
 * data, and the count travels with it so the UI can say "no gameweeks
 * yet" rather than offering a chip that silently draws nothing.
 */
export interface SeasonFacet {
  season: string;
  /** Panel rows carried for this season. Zero is an answer. */
  rows: number;
  /** Distinct gameweeks with at least one row. */
  gameweeks: number;
  /** Whether this is the season the pipeline is currently collecting. */
  current: boolean;
}

export interface PanelFacets {
  seasons: SeasonFacet[];
  teams: string[];
  positions: string[];
  gwMin: number;
  gwMax: number;
  priceMin: number;
  priceMax: number;
  rows: number;
}

export async function facets(
  session: Session,
  currentSeason?: string | null,
): Promise<PanelFacets> {
  const [season, team, position, gw, value] = await Promise.all([
    session.column("season"),
    session.column("team"),
    session.column("position"),
    session.column("gw"),
    session.column("value"),
  ]);

  const distinct = (column: PanelColumn): string[] => {
    const seen = new Set<string>();
    for (const entry of column) if (entry !== null) seen.add(String(entry));
    return [...seen].sort();
  };

  const numericExtent = (column: PanelColumn): [number, number] => {
    let lo = Infinity;
    let hi = -Infinity;
    for (const entry of column) {
      if (typeof entry !== "number") continue;
      if (entry < lo) lo = entry;
      if (entry > hi) hi = entry;
    }
    return Number.isFinite(lo) ? [lo, hi] : [0, 0];
  };

  const [gwMin, gwMax] = numericExtent(gw);
  const [priceMin, priceMax] = numericExtent(value);

  return {
    seasons: seasonFacets(season, gw, currentSeason ?? null),
    teams: distinct(team),
    /*
     * Pitch order, not alphabetical. Sorted output gives DEF, FWD, GK,
     * MID, which is the order of no football thought anyone has ever had
     * — every squad list, every FPL screen and every table in this repo
     * reads goalkeeper outward.
     */
    positions: distinct(position).sort(byPitchOrder),
    gwMin,
    gwMax,
    priceMin,
    priceMax,
    rows: session.rows,
  };
}

/**
 * Seasons in chronological order, each with what the panel holds for it,
 * and the current season present whether or not that is anything.
 *
 * FPL season labels sort lexically in chronological order ("2023-24" <
 * "2026-27"), which is why a plain sort is correct here and will stay
 * correct until the year 10000.
 */
function seasonFacets(
  season: PanelColumn,
  gw: PanelColumn,
  currentSeason: string | null,
): SeasonFacet[] {
  const gameweeks = new Map<string, Set<number>>();
  const rows = new Map<string, number>();

  for (let i = 0; i < season.length; i += 1) {
    const value = season[i];
    if (value === null) continue;
    const key = String(value);
    rows.set(key, (rows.get(key) ?? 0) + 1);

    const week = gw[i];
    if (typeof week === "number") {
      let weeks = gameweeks.get(key);
      if (!weeks) {
        weeks = new Set();
        gameweeks.set(key, weeks);
      }
      weeks.add(week);
    }
  }

  // The union: a current season with no completed gameweek yet is still a
  // season the reader may want to select, and saying so is more useful
  // than pretending it does not exist.
  if (currentSeason && !rows.has(currentSeason)) rows.set(currentSeason, 0);

  return [...rows.keys()]
    .sort()
    .map((name) => ({
      season: name,
      rows: rows.get(name) ?? 0,
      gameweeks: gameweeks.get(name)?.size ?? 0,
      current: name === currentSeason,
    }));
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
