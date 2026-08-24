/**
 * §5.4.3 — the player × gameweek heat map.
 *
 * > "This surface is where a slump or a hot streak becomes visible in one
 * > glance, which no line chart of the same data achieves."
 *
 * Technically expressible in the Graph Builder as a rect mark, and it
 * gets its own route anyway, because §5.4.3 says so and because the
 * defaults are the product: rows sorted by what the reader is looking
 * for, gameweeks in order, and the three-way cell distinction below —
 * none of which survives being reassembled from five drag operations.
 *
 * **The load-bearing rule is the one about cell states.** §5.4.3:
 *
 * > "Blank vs zero vs did-not-play are three visually distinct states. A
 * > player who played 90 minutes and scored zero points and a player who
 * > was not in the squad are not the same fact and must not share a cell
 * > treatment."
 *
 * So `cellState` below returns four, not one — the fourth being a row
 * that exists with a null metric, which is §5.3.3's "unknown" and is not
 * the same claim as zero either. Every one of them renders differently
 * and none of them renders as a zero-coloured cell.
 *
 * §5.16 deviation D11: **the default colour metric is raw `total_points`,
 * not the position-normalized total points §5.4.3 asks for.** There is no
 * such column to read. §5.7.2 emits `_z_pos` companions only for metrics
 * the registry flags `normalizable`, and `columns.py` does not flag the
 * gameweek point total — it is a count for one match, not a rate over a
 * season, and there is no exported population to score it against.
 * Computing one here is exactly what §5.6 forbids. The toggle still works
 * for every rate metric, which do carry companions, and the basis line
 * says which is which. Points are also the one metric where raw
 * cross-position comparison is legitimate: six points is six points
 * whoever scored them, which is why §5.7.5's caution is scoped to
 * `normalizable` columns and correctly stays silent here.
 */

import { useEffect, useMemo, useState } from "react";
import { useApp } from "../app/state";
import { BucketBadge } from "../components/BucketBadge";
import { FilterBar } from "../components/FilterBar";
import { Provenance } from "../components/Provenance";
import { loadColumns, type LoadProgress } from "../data/load";
import { useBoard } from "../data/useBoard";
import { byTeam, gameweekShape } from "../data/fixtures";
import { useFixtures } from "../data/fixtures";
import type { BoardFile, ColumnsFile, ColumnSpec } from "../data/schema";
import { divergingColor, type Direction } from "../design/scale";
import { reduce } from "../query/reduce";
import { facets as loadFacets, select, type PanelFacets } from "../query/panel";
import { openSession, PanelMissingError, type Session } from "../query/session";
import styles from "./FormMatrix.module.css";
import { count, noun } from "../design/text";

type Engine =
  | { status: "opening"; progress: LoadProgress | null }
  | { status: "ready"; session: Session; facets: PanelFacets; columns: ColumnsFile }
  | { status: "absent" }
  | { status: "error"; error: Error };

/** How many rows the grid draws. More than this stops being one glance. */
const ROW_CHOICES = [20, 40, 80] as const;

/** §5.4.3's default colour metric — see D11 in the module docstring. */
const DEFAULT_METRIC = "total_points";

interface Cell {
  value: number | null;
  minutes: number;
  fixtures: number;
}

interface Row {
  id: number;
  name: string;
  team: string;
  position: string;
  cells: Map<number, Cell>;
  /** The sort value: the metric reduced over the visible gameweeks. */
  score: number | null;
  /** Gameweeks with a fixture, for the row summary. */
  played: number;
}

export function FormMatrix() {
  const { state, dispatch } = useApp();
  const [engine, setEngine] = useState<Engine>({ status: "opening", progress: null });
  const [metric, setMetric] = useState(DEFAULT_METRIC);
  const [rowCount, setRowCount] = useState<number>(40);
  const [data, setData] = useState<{ rows: Row[]; gameweeks: number[]; total: number } | null>(null);
  const [working, setWorking] = useState(false);
  const [failure, setFailure] = useState<Error | null>(null);
  const [toggleTouched, setToggleTouched] = useState(false);
  /* §5.5.4's reverse path, for rows the reader has selected. */
  const board = useBoard();
  /*
   * The schedule, so a gameweek with no fixture is distinguishable from a
   * gameweek the player sat out (§5.4.3). Scoped to the season on screen
   * by `byTeam`, so the archive is never annotated with a schedule that
   * belongs to a different year.
   */
  const fixtures = useFixtures();

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const columns = await loadColumns();
        const session = await openSession((progress) =>
          cancelled ? undefined : setEngine({ status: "opening", progress }),
        );
        /*
         * §5.3.1's `current_season`, stamped on every export header. The
         * panel can only report the seasons it *carries*, and the season
         * the reader cares about most is the one that has barely started
         * — so the filter learns it from the header instead and offers it
         * whether or not any of its gameweeks have landed.
         */
        const panelFacets = await loadFacets(session, columns.header.current_season ?? null);
        if (!cancelled) setEngine({ status: "ready", session, facets: panelFacets, columns });
      } catch (error) {
        if (cancelled) return;
        setEngine(
          error instanceof PanelMissingError
            ? { status: "absent" }
            : { status: "error", error: error as Error },
        );
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  const columnIndex = useMemo(() => {
    if (engine.status !== "ready") return new Map<string, ColumnSpec>();
    return new Map(engine.columns.columns.map((column) => [column.key, column]));
  }, [engine]);

  /*
   * §5.7.3: "Form Matrix — normalized. Comparing players down the rows is
   * the entire point." A default rather than a lock; the toggle is
   * app-level state (§5.5.3) and stays where the reader put it.
   */
  const normalized = toggleTouched ? state.normalized : true;
  const spec = columnIndex.get(metric);
  const companion = Boolean(spec?.normalizable && spec.normalized_key);
  const readingNormalized = normalized && companion;

  const session = engine.status === "ready" ? engine.session : null;
  const allSeasons = engine.status === "ready" ? engine.facets.seasons : [];

  /*
   * **This surface is single-season by construction**, and that is a fact
   * about the data rather than a simplification.
   *
   * FPL reassigns `element_id` at every rollover, so id 1 in 2023-24 and
   * id 1 in 2025-26 are two different footballers. A matrix keyed on the
   * id alone across seasons would draw them as one row — and since both
   * seasons number their gameweeks 1 to 38, their cells would overwrite
   * each other rather than sit side by side. The result would look
   * completely ordinary and be entirely wrong.
   *
   * The columns are gameweeks of a season anyway, so the honest reading
   * of "no season filtered" is the latest one rather than all of them.
   */
  /*
   * Which season to show when the filter does not name exactly one.
   *
   * The current season first — it is the one the reader came for — but
   * only once it has recorded something. Opening on an empty grid in
   * August would be honest and useless. Until then it falls back to the
   * most recent season that has data, and the note below says so, so the
   * fallback is never mistaken for the current season being empty of
   * *players* rather than empty of *gameweeks*.
   */
  const populated = allSeasons.filter((entry) => entry.rows > 0);
  const preferred =
    allSeasons.find((entry) => entry.current && entry.rows > 0) ??
    populated[populated.length - 1] ??
    null;

  const picked = state.filters.seasons.length === 1 ? state.filters.seasons[0]! : null;
  const season = picked ?? preferred?.season ?? null;
  const seasonFacet = allSeasons.find((entry) => entry.season === season) ?? null;
  const seasonOverridden = picked === null && season !== null;
  // The reader asked for a season the panel has no rows for. Almost always
  // the current one, before its first gameweek lands.
  const seasonEmpty = seasonFacet !== null && seasonFacet.rows === 0;

  const schedule = useMemo(() => byTeam(fixtures, season), [fixtures, season]);
  /** Which clubs had no fixture in each gameweek on screen. */
  const blanksByGameweek = useMemo(() => {
    if (schedule.size === 0 || !data) return new Map<number, Set<string>>();
    return new Map(
      data.gameweeks.map((gw) => [gw, gameweekShape(schedule, gw).blank]),
    );
  }, [schedule, data]);

  useEffect(() => {
    if (!session || columnIndex.size === 0) return;
    let cancelled = false;
    setWorking(true);
    setFailure(null);

    (async () => {
      try {
        const selection = await select(
          session,
          {
            columns: ["element_id", "name", "team", "position", "gw", "n_fixtures", "minutes", metric],
            // The season is pinned rather than passed through -- see the
            // note above `season`.
            filters: { ...state.filters, seasons: season ? [season] : [] },
            normalized,
          },
          columnIndex,
        );

        const get = (key: string) => selection.values.get(key)!;
        const ids = get("element_id");
        const names = get("name");
        const teams = get("team");
        const positions = get("position");
        const gws = get("gw");
        const fixtures = get("n_fixtures");
        const minutes = get("minutes");
        const metrics = get(metric);

        const byPlayer = new Map<number, Row>();
        const seenGameweeks = new Set<number>();

        for (const i of selection.index) {
          const id = Number(ids[i]);
          const gw = Number(gws[i]);
          if (!Number.isFinite(id) || !Number.isFinite(gw)) continue;
          seenGameweeks.add(gw);

          let row = byPlayer.get(id);
          if (!row) {
            row = {
              id,
              name: String(names[i] ?? id),
              team: String(teams[i] ?? ""),
              position: String(positions[i] ?? ""),
              cells: new Map(),
              score: null,
              played: 0,
            };
            byPlayer.set(id, row);
          }
          const raw = metrics[i];
          row.cells.set(gw, {
            value: typeof raw === "number" ? raw : null,
            minutes: typeof minutes[i] === "number" ? (minutes[i] as number) : 0,
            fixtures: typeof fixtures[i] === "number" ? (fixtures[i] as number) : 1,
          });
        }

        const rows = [...byPlayer.values()];
        for (const row of rows) {
          const values = [...row.cells.values()].map((cell) => cell.value);
          /*
           * The row's sort value is a §5.6.2 reduction over exactly the
           * gameweeks on screen, computed by the one implementation the
           * golden test covers. `sum` for a count like points, `mean` for
           * a rate — summing a per-90 over eleven gameweeks would produce
           * a number with no unit.
           */
          const counting = columnIndex.get(metric)?.format === "d" && !readingNormalized;
          row.score = reduce(values, counting ? "sum" : "mean");
          row.played = [...row.cells.values()].filter((cell) => cell.minutes > 0).length;
        }

        rows.sort((left, right) => {
          // A player with no value sorts last, not as a zero (§5.3.3).
          if (left.score === null && right.score === null) return 0;
          if (left.score === null) return 1;
          if (right.score === null) return -1;
          return right.score - left.score;
        });

        if (!cancelled) {
          setData({
            rows: rows.slice(0, rowCount),
            gameweeks: [...seenGameweeks].sort((a, b) => a - b),
            total: rows.length,
          });
        }
      } catch (error) {
        if (!cancelled) setFailure(error as Error);
      } finally {
        if (!cancelled) setWorking(false);
      }
    })();

    return () => {
      cancelled = true;
    };
  }, [session, columnIndex, metric, state.filters, season, normalized, rowCount, readingNormalized]);

  /*
   * Above the early returns, because React counts hooks by call order
   * and the loading branch below returns before this line would run.
   */
  /*
   * The colour scale runs to the 95th percentile of what is on screen,
   * not to the maximum.
   *
   * Scaling to the maximum is what a first pass does and it makes this
   * surface useless: one 20-point haul saturates the ramp and the 1-to-6
   * range every other cell lives in collapses into a single muddy
   * purple, which is precisely the "slump or hot streak in one glance"
   * that §5.4.3 exists to show. Cells beyond the percentile clamp at the
   * pole, which is honest -- they are off the top of the scale, and the
   * number is printed in the cell regardless (§5.10 forbids colour being
   * the only encoding, and that is what rescues the clamped cells).
   *
   * `quantile` is one of §5.6.2's seven permitted reductions and this
   * goes through the same `reduce.ts` the golden test covers.
   */
  const extreme = useMemo(() => {
    if (!data) return 1;
    const magnitudes = data.rows.flatMap((row) =>
      [...row.cells.values()]
        .filter((cell) => cell.value !== null && cell.minutes > 0)
        .map((cell) => Math.abs(cell.value!)),
    );
    return Math.max(reduce(magnitudes, "quantile", 0.95) ?? 1, 1e-9);
  }, [data]);

  if (engine.status === "opening") return <Opening progress={engine.progress} />;
  if (engine.status === "absent") return <PanelAbsent />;
  if (engine.status === "error") return <Failed error={engine.error} />;

  const metricColumns = engine.columns.columns.filter(
    (column) => column.role === "quantitative",
  );

  const direction: Direction =
    spec?.higher_is_better === false ? "lower_is_better"
    : spec?.higher_is_better === true ? "higher_is_better"
    : "neutral";


  const normalizedReason = readingNormalized
    ? `Within-position z-scores, read from the export. Each player is measured against his own position group.`
    : companion
      ? "Raw units."
      : `${spec?.label ?? metric} has no exported within-position companion, so it reads raw whatever the toggle says. Nothing is standardized in the browser (§5.6).`;

  return (
    <main className={styles.form}>
      <header className={styles.head}>
        <div>
          <h1 className={styles.title}>Form Matrix</h1>
          <p className={styles.sub}>
            Every player down the rows, every gameweek across. Where a slump or a hot streak
            is one glance rather than one chart each.
          </p>
        </div>
        <Provenance
          header={engine.columns.header}
          basis={engine.columns.header.normalization_basis}
        />
      </header>

      <FilterBar
        facets={engine.facets}
        filters={state.filters}
        onChange={(filters) => dispatch({ type: "filters", filters })}
        normalized={normalized}
        onNormalized={(next) => {
          setToggleTouched(true);
          dispatch({ type: "normalized", normalized: next });
        }}
        normalizedReason={normalizedReason}
      />

      <div className={styles.controls}>
        <label className={styles.control}>
          <span>Colour by</span>
          <select value={metric} onChange={(event) => setMetric(event.target.value)}>
            {metricColumns.map((column) => (
              <option key={column.key} value={column.key}>
                {column.label}
              </option>
            ))}
          </select>
        </label>

        <label className={styles.control}>
          <span>Rows</span>
          <select value={rowCount} onChange={(event) => setRowCount(Number(event.target.value))}>
            {ROW_CHOICES.map((count) => (
              <option key={count} value={count}>
                top {count}
              </option>
            ))}
          </select>
        </label>

        <Key />

        {data && (
          <p className={styles.count}>
            {data.rows.length} of <span className="data">{data.total.toLocaleString()}</span>{" "}
            {noun(data.total, "player")}, ranked by {readingNormalized ? "mean z-score" : (spec?.label ?? metric)} over
            the {count(data.gameweeks.length, "gameweek")} shown.
          </p>
        )}
      </div>

      {seasonEmpty && (
        /*
         * §5.14.14 forbids mocked data in any state, including empty
         * ones. So this says what is absent and when it arrives, and
         * renders no grid at all rather than a grid of blanks that could
         * be mistaken for players who did not play.
         */
        <p className={styles.caution} role="note">
          <span className="data">{season}</span> has no completed gameweeks in the panel yet.
          The collector records each one after it finishes, and this grid fills in a column at
          a time as they land — nothing here is missing, it has not happened.
          {preferred && (
            <>
              {" "}
              <button
                type="button"
                className={styles.inlineAction}
                onClick={() => dispatch({ type: "filters", filters: { seasons: [preferred.season] } })}
              >
                Show {preferred.season} instead
              </button>
            </>
          )}
        </p>
      )}

      {seasonOverridden && !seasonEmpty && (
        /*
         * Never silently. The reader set a filter (or set none) and this
         * surface is showing something narrower than they asked for, so
         * it says which season and why — with the fix one click away.
         */
        <p className={styles.caution} role="note">
          Showing <span className="data">{season}</span> only. The columns here are one
          season&rsquo;s gameweeks, and FPL reissues player ids at every rollover — so two
          seasons on one row would be two different footballers sharing it. Pick a single
          season above to change which.
        </p>
      )}

      {failure && (
        <p className={styles.failure} role="alert">
          {failure.message}
        </p>
      )}

      {working && !data && !seasonEmpty && (
        <p className={styles.working}>Reading the panel…</p>
      )}

      {data && data.rows.length === 0 && !seasonEmpty && (
        <p className={styles.empty}>No player survives the current filters.</p>
      )}

      {data && data.rows.length > 0 && !seasonEmpty && (
        <Grid
          rows={data.rows}
          gameweeks={data.gameweeks}
          extreme={extreme}
          direction={direction}
          spec={spec}
          board={board}
          blanks={blanksByGameweek}
          onSelect={(id) => {
            /*
             * §5.4.3: "row click opens that player in Comparison." The
             * selection is app-level (§5.5.3) so it survives the
             * navigation, and `toggleSelect` means clicking a player
             * already on the board removes him rather than opening a
             * comparison of one — which is what a reader assembling a
             * shortlist across two surfaces expects.
             */
            dispatch({ type: "toggleSelect", id });
            if (!state.selection.includes(id)) {
              dispatch({ type: "navigate", view: "compare" });
            }
          }}
          selected={state.selection}
        />
      )}
    </main>
  );
}

/**
 * The four states a cell can be in, and none of them is "zero".
 *
 * `blank`  — no row: the player's club had no fixture that gameweek, or
 *            he was not in the squad. Nothing happened to measure.
 * `dnp`    — a fixture, and no minutes. Something happened and he was not
 *            part of it, which is a fact about him rather than about the
 *            fixture list.
 * `absent` — minutes played, and the metric is null. §5.3.3's unknown:
 *            most often a rate below the eligibility floor.
 * `value`  — a number, including a real zero.
 */
type CellState = "blank" | "noFixture" | "dnp" | "absent" | "value";

/**
 * `clubPlayed` is what the schedule says about the club, or `null` when
 * no schedule covers this season.
 *
 * Without it, a missing row is a single "blank" that conflates two
 * different facts: the club had no fixture at all, and the club played
 * while this player was not in the squad. §5.4.3 is explicit that states
 * a reader would draw different conclusions from must not share a cell
 * treatment, and those two are exactly that — one is the fixture list,
 * the other is a team sheet.
 *
 * When there is no schedule the honest answer is the old one: a row is
 * missing and we cannot say why.
 */
export function cellState(cell: Cell | undefined, clubPlayed: boolean | null = null): CellState {
  if (!cell) {
    if (clubPlayed === false) return "noFixture";
    return "blank";
  }
  if (cell.minutes === 0) return "dnp";
  if (cell.value === null) return "absent";
  return "value";
}

interface GridProps {
  rows: Row[];
  gameweeks: number[];
  extreme: number;
  direction: Direction;
  spec: ColumnSpec | undefined;
  onSelect: (id: number) => void;
  selected: number[];
  board: BoardFile | null;
  blanks: Map<number, Set<string>>;
}

function Grid({
  rows,
  gameweeks,
  extreme,
  direction,
  spec,
  onSelect,
  selected,
  board,
  blanks,
}: GridProps) {
  return (
    <div className={styles.scroll}>
      <table className={styles.grid}>
        <caption className={styles.caption}>
          Player by gameweek, coloured by {spec?.label ?? "value"}.
        </caption>
        <thead>
          <tr>
            <th scope="col" className={styles.corner}>
              Player
            </th>
            {gameweeks.map((gw) => (
              <th key={gw} scope="col" className={styles.gwHead}>
                {gw}
              </th>
            ))}
            <th scope="col" className={styles.trendHead}>
              Trend
            </th>
          </tr>
        </thead>
        <tbody>
          {rows.map((row) => (
            <tr
              key={row.id}
              className={styles.row}
              data-selected={selected.includes(row.id) || undefined}
            >
              <th scope="row" className={styles.player}>
                <button
                  type="button"
                  className={styles.playerButton}
                  onClick={() => onSelect(row.id)}
                  title={`${row.name} — ${row.team}, ${row.position}. ${row.played} of ${row.cells.size} fixtures played. Opens in Comparison.`}
                >
                  <span className={styles.playerName}>{row.name}</span>
                  <span className={styles.playerMeta}>
                    {row.team} · {row.position}
                  </span>
                </button>
                {selected.includes(row.id) && (
                  <span className={styles.rowBadge}>
                    <BucketBadge board={board} elementId={row.id} />
                  </span>
                )}
              </th>

              {gameweeks.map((gw) => {
                const cell = row.cells.get(gw);
                const blanked = blanks.get(gw);
                const clubPlayed = blanked ? !blanked.has(row.team) : null;
                const state = cellState(cell, clubPlayed);
                return (
                  <td
                    key={gw}
                    className={styles.cell}
                    data-state={state}
                    style={
                      state === "value"
                        ? {
                            background: divergingColor(cell!.value! / extreme, direction),
                            color:
                              Math.abs(cell!.value! / extreme) > 0.55
                                ? "var(--ground)"
                                : "var(--paper)",
                          }
                        : undefined
                    }
                    title={describe(row, gw, cell, state, spec)}
                  >
                    <span className={styles.cellText}>{cellLabel(cell, state, spec)}</span>
                    {cell && cell.fixtures > 1 && <span className={styles.double} aria-hidden="true" />}
                  </td>
                );
              })}

              <td className={styles.trend}>
                <Sparkline row={row} gameweeks={gameweeks} extreme={extreme} />
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

/**
 * §5.4.3: "Row hover reveals a sparkline of the same metric."
 *
 * Rendered in its own column rather than on hover, because a shape that
 * only exists while the pointer is over it cannot be compared against the
 * row above — and comparing rows is what this surface is for. Recorded as
 * part of D11.
 */
function Sparkline({
  row,
  gameweeks,
  extreme,
}: {
  row: Row;
  gameweeks: number[];
  extreme: number;
}) {
  const W = 72;
  const H = 18;
  const points = gameweeks
    .map((gw, index) => ({ gw, index, cell: row.cells.get(gw) }))
    .filter((entry) => entry.cell && entry.cell.value !== null);

  if (points.length < 2) return <span className={styles.noTrend}>—</span>;

  const x = (index: number) => (index / Math.max(gameweeks.length - 1, 1)) * W;
  const y = (value: number) => H / 2 - (value / extreme) * (H / 2 - 1);

  const d = points
    .map((entry, i) => `${i === 0 ? "M" : "L"}${x(entry.index).toFixed(1)},${y(entry.cell!.value!).toFixed(1)}`)
    .join(" ");

  return (
    <svg className={styles.spark} viewBox={`0 0 ${W} ${H}`} role="img" aria-label={`${row.name} trend`}>
      <line x1={0} y1={H / 2} x2={W} y2={H / 2} className={styles.sparkBase} />
      <path d={d} className={styles.sparkLine} />
    </svg>
  );
}

function cellLabel(cell: Cell | undefined, state: CellState, spec: ColumnSpec | undefined): string {
  switch (state) {
    case "blank":
      return "";
    case "noFixture":
      // The club had no fixture. Structurally distinct from a missing row
      // whose reason is unknown, and from a player who sat one out.
      return "";
    case "dnp":
      // Not a zero and not an em dash: this player had a fixture and did
      // not appear in it.
      return "·";
    case "absent":
      return "—";
    case "value":
      return format(cell!.value!, spec);
  }
}

function describe(
  row: Row,
  gw: number,
  cell: Cell | undefined,
  state: CellState,
  spec: ColumnSpec | undefined,
): string {
  const who = `${row.name}, GW${gw}`;
  switch (state) {
    case "blank":
      return `${who}: no row. Either ${row.team} had no fixture or he was not in the squad — no schedule is loaded for this season, so the two cannot be told apart.`;
    case "noFixture":
      return `${who}: ${row.team} had no fixture in gameweek ${gw}. A blank gameweek, not a player who was left out.`;
    case "dnp":
      return `${who}: did not play. ${row.team} had a fixture and he recorded no minutes.`;
    case "absent":
      return `${who}: ${cell!.minutes} minutes, no ${spec?.label ?? "value"} recorded. Unknown, not zero.`;
    case "value":
      return `${who}: ${format(cell!.value!, spec)} over ${cell!.minutes} minutes${
        cell!.fixtures > 1 ? ` across ${cell!.fixtures} fixtures` : ""
      }.`;
  }
}

function format(value: number, spec: ColumnSpec | undefined): string {
  switch (spec?.format) {
    case "d":
      return value.toFixed(0);
    case ".0%":
      return `${Math.round(value * 100)}`;
    case ".2f":
      return Math.abs(value) >= 10 ? value.toFixed(1) : value.toFixed(2).replace(/^0\./, ".").replace(/^-0\./, "-.");
    default:
      break;
  }
  if (Number.isInteger(value)) return value.toFixed(0);
  return Math.abs(value) >= 10 ? value.toFixed(1) : value.toFixed(2).replace(/^0\./, ".").replace(/^-0\./, "-.");
}

/** §5.10: the three states carry meaning, so they carry a key. */
function Key() {
  return (
    <ul className={styles.key} aria-label="Cell states">
      <li className={styles.keyItem}>
        <span className={styles.keySwatch} data-state="value" />
        played
      </li>
      <li className={styles.keyItem}>
        <span className={styles.keySwatch} data-state="dnp" />
        no minutes
      </li>
      <li className={styles.keyItem}>
        <span className={styles.keySwatch} data-state="noFixture" />
        no fixture
      </li>
      <li className={styles.keyItem}>
        <span className={styles.keySwatch} data-state="blank" />
        not in squad
      </li>
      <li className={styles.keyItem}>
        <span className={styles.keySwatch} data-state="double" />
        double
      </li>
    </ul>
  );
}

function Opening({ progress }: { progress: LoadProgress | null }) {
  const mb = (bytes: number) => `${(bytes / 1_048_576).toFixed(1)} MB`;
  const pct =
    progress?.total != null ? Math.round((progress.received / progress.total) * 100) : null;
  return (
    <main className={styles.form}>
      <h1 className={styles.title}>Form Matrix</h1>
      <div className={styles.loading}>
        <div
          className={styles.track}
          role="progressbar"
          aria-valuenow={pct ?? undefined}
          aria-valuemin={0}
          aria-valuemax={100}
          aria-label="Loading the panel"
        >
          <div className={styles.fill} style={{ inlineSize: pct === null ? "30%" : `${pct}%` }} />
        </div>
        <p className={`${styles.sub} data`}>
          {progress
            ? `panel.parquet — ${mb(progress.received)}${progress.total ? ` of ${mb(progress.total)}` : ""}`
            : "requesting panel.parquet"}
        </p>
      </div>
    </main>
  );
}

/** §5.14.8's explanatory empty state. */
function PanelAbsent() {
  return (
    <main className={styles.form}>
      <h1 className={styles.title}>Form Matrix</h1>
      <div className={styles.absent}>
        <p className={styles.absentHead}>The panel has not been built.</p>
        <p className={styles.sub}>
          This surface reads <span className="data">panel.parquet</span> — 85,000
          player-gameweeks, which §5.3.4 deliberately does not commit.
        </p>
        <p className={styles.sub}>
          Run <code className="data">python -m web.export panel</code> and reload.
        </p>
      </div>
    </main>
  );
}

function Failed({ error }: { error: Error }) {
  return (
    <main className={styles.form}>
      <h1 className={styles.title}>Form Matrix</h1>
      <div className={styles.absent} role="alert">
        <p className={styles.absentHead}>The panel did not load.</p>
        <p className={`${styles.sub} data`}>{error.message}</p>
      </div>
    </main>
  );
}
