/**
 * The Projections table: every player's expected points over the next
 * few gameweeks, with the fixture behind each number.
 *
 * Every other FPL tool leads with this table, and until now the app showed
 * one projected gameweek per player (`players.json`). This reads
 * `projections.json`, which is the same event model asked about each of
 * the next six.
 *
 * **It stays descriptive (§5.0.2).** It reports what the model expects and
 * lets the reader sort it. There is no "buy", no highlighted pick and no
 * suggested captain: the shading is the size of the number and nothing
 * else, and the total is a sum the reader chose the length of.
 *
 * **Later gameweeks are weaker forecasts and the page says so.** The
 * model's rates are trailing, so what changes from one column to the next
 * is mostly the fixture, and nothing in the file is refreshed by the
 * results in between. The Scorecard publishes how the one-gameweek
 * projection actually performed; it publishes nothing for a longer
 * horizon, and this note does not pretend otherwise.
 *
 * **A blank is drawn, not left empty,** and a double shows both fixtures,
 * so the 0.0 and the large number each have a visible reason.
 */

import { useEffect, useMemo, useState } from "react";
import { ErrorState } from "../components/ErrorState";
import { LoadingState } from "../components/LoadingState";
import { Provenance } from "../components/Provenance";
import { useApp } from "../app/state";
import { loadProjections, type LoadProgress } from "../data/load";
import {
  NO_FILTERS,
  clampWindow,
  fixturesFor,
  selectRows,
  windowTotal,
  type Filters,
  type Sort,
  type SortKey,
} from "../data/projections";
import type { ProjectionFixture, ProjectionsFile } from "../data/schema";
import { cellTextColor, outcomeColor } from "../design/scale";
import { count } from "../design/text";
import styles from "./Projections.module.css";

type State =
  | { status: "loading"; progress: LoadProgress | null }
  | { status: "error"; error: Error }
  | { status: "ready"; file: ProjectionsFile };

const POSITIONS = ["all", "GK", "DEF", "MID", "FWD"];

/** Rows drawn before the reader asks for the rest. */
const PAGE = 150;

/** Price ceilings offered, in tenths of a million; null is no ceiling. */
const CEILINGS: (number | null)[] = [null, 45, 55, 65, 75, 100, 130];

/**
 * An xP of 3.5 is the middle of the ramp and 7 or more the top. The ramp
 * is fixed rather than fitted to the table, so a shade means the same
 * thing whichever players the filters have left on screen.
 */
function shade(xp: number): number {
  return Math.max(-1, Math.min(1, (xp - 3.5) / 3.5));
}

function sameKey(left: SortKey, right: SortKey): boolean {
  return left.kind === right.kind && (left.kind !== "gw" || (right.kind === "gw" && left.index === right.index));
}

function money(tenths: number | null): string {
  return tenths === null ? "—" : (tenths / 10).toFixed(1);
}

export function Projections() {
  const { dispatch } = useApp();
  const [data, setData] = useState<State>({ status: "loading", progress: null });
  const [window, setWindow] = useState(3);
  const [filters, setFilters] = useState<Filters>(NO_FILTERS);
  const [sort, setSort] = useState<Sort>({ key: { kind: "total" }, desc: true });
  const [showAll, setShowAll] = useState(false);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const file = await loadProjections((progress) =>
          cancelled ? undefined : setData({ status: "loading", progress }),
        );
        if (!cancelled) setData({ status: "ready", file });
      } catch (error) {
        if (!cancelled) setData({ status: "error", error: error as Error });
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  const file = data.status === "ready" ? data.file : null;
  const span = file ? clampWindow(window, file.gameweeks.length) : 0;
  const fixturesOf = useMemo(() => (file ? fixturesFor(file) : null), [file]);
  const rows = useMemo(
    () => (file ? selectRows(file.players, filters, sort, span) : []),
    [file, filters, sort, span],
  );

  if (data.status === "loading") {
    return (
      <LoadingState
        rootClassName={styles.projections}
        title="Projections"
        file="projections.json"
        label="Loading the projections"
        progress={data.progress}
      />
    );
  }
  if (data.status === "error") {
    return (
      <ErrorState
        rootClassName={styles.projections}
        title="Projections"
        heading="projections.json did not load."
        command="projections"
        error={data.error}
      />
    );
  }

  const { gameweeks } = data.file;
  const shown = showAll ? rows : rows.slice(0, PAGE);

  const sortBy = (key: SortKey) =>
    setSort((current) =>
      sameKey(current.key, key)
        ? { key, desc: !current.desc }
        : { key, desc: key.kind !== "name" && key.kind !== "team" && key.kind !== "position" },
    );

  const head = (label: string, key: SortKey, numeric = false, title?: string) => {
    const active = sameKey(sort.key, key);
    return (
      <th
        scope="col"
        className={styles.th}
        data-numeric={numeric || undefined}
        data-active={active || undefined}
        aria-sort={active ? (sort.desc ? "descending" : "ascending") : "none"}
        title={title}
      >
        <button type="button" className={styles.sortButton} onClick={() => sortBy(key)}>
          {label}
          {active && <span className={styles.sortMark}>{sort.desc ? "▼" : "▲"}</span>}
        </button>
      </th>
    );
  };

  if (gameweeks.length === 0) {
    return (
      <main className={styles.projections}>
        <h1 className={styles.title}>Projections</h1>
        <div className={styles.absent}>
          <p className={styles.absentHead}>No gameweeks left to project.</p>
          <p className={styles.sub}>
            {data.file.season} has no gameweek after {data.file.gameweek}, or the schedule did not
            load when the file was built.
          </p>
        </div>
      </main>
    );
  }

  return (
    <main className={styles.projections}>
      <header className={styles.head}>
        <div>
          <h1 className={styles.title}>Projections</h1>
          <p className={styles.sub}>
            Expected points per player for gameweeks{" "}
            <span className="data">
              {gameweeks[0]}–{gameweeks[gameweeks.length - 1]}
            </span>
            , from the event model, with the fixture behind each number. A blank gameweek is 0.0
            and a double sums both fixtures.
          </p>
        </div>
        <Provenance header={data.file.header} basis={data.file.header.normalization_basis} />
      </header>

      <div className={styles.controls}>
        <label className={styles.searchLabel}>
          <span className={styles.searchText}>Search</span>
          <input
            type="search"
            className={styles.search}
            placeholder="Player or club"
            value={filters.search}
            onChange={(event) => setFilters({ ...filters, search: event.target.value })}
          />
        </label>

        <fieldset className={styles.group}>
          <legend className={styles.legend}>Position</legend>
          <div className={styles.chips}>
            {POSITIONS.map((position) => (
              <button
                key={position}
                type="button"
                className={styles.chip}
                data-on={filters.position === position || undefined}
                aria-pressed={filters.position === position}
                onClick={() => setFilters({ ...filters, position })}
              >
                {position === "all" ? "All" : position}
              </button>
            ))}
          </div>
        </fieldset>

        <label className={styles.searchLabel}>
          <span className={styles.searchText}>Max price</span>
          <select
            className={styles.search}
            value={filters.maxPrice === null ? "" : String(filters.maxPrice)}
            onChange={(event) =>
              setFilters({
                ...filters,
                maxPrice: event.target.value === "" ? null : Number(event.target.value),
              })
            }
          >
            {CEILINGS.map((ceiling) => (
              <option key={ceiling ?? "any"} value={ceiling === null ? "" : String(ceiling)}>
                {ceiling === null ? "Any" : `£${money(ceiling)}m`}
              </option>
            ))}
          </select>
        </label>

        <fieldset className={styles.group}>
          <legend className={styles.legend}>Total over</legend>
          <div className={styles.chips}>
            {gameweeks.map((gw, index) => (
              <button
                key={gw}
                type="button"
                className={styles.chip}
                data-on={span === index + 1 || undefined}
                aria-pressed={span === index + 1}
                onClick={() => setWindow(index + 1)}
                title={`The first ${count(index + 1, "gameweek")}, through gameweek ${gw}.`}
              >
                {index + 1} GW
              </button>
            ))}
          </div>
        </fieldset>
      </div>

      <div className={styles.scroll}>
        <table className={styles.grid}>
          <caption className={styles.caption}>
            {count(rows.length, "player")}, sorted by{" "}
            {sort.key.kind === "gw"
              ? `gameweek ${gameweeks[sort.key.index]}`
              : sort.key.kind === "total"
                ? `total over ${count(span, "gameweek")}`
                : sort.key.kind}
            . Shade is the size of the expected points, on a fixed scale.
          </caption>
          <thead>
            <tr>
              {head("Player", { kind: "name" })}
              {head("Club", { kind: "team" })}
              {head("Pos", { kind: "position" })}
              {head("£m", { kind: "price" }, true)}
              {head(
                `Total ${span} GW`,
                { kind: "total" },
                true,
                `Sum of the first ${count(span, "gameweek")}.`,
              )}
              {gameweeks.map((gw, index) => (
                <th
                  key={gw}
                  scope="col"
                  className={styles.th}
                  data-numeric
                  data-active={sameKey(sort.key, { kind: "gw", index }) || undefined}
                  data-outside={index >= span || undefined}
                  aria-sort={
                    sameKey(sort.key, { kind: "gw", index })
                      ? sort.desc
                        ? "descending"
                        : "ascending"
                      : "none"
                  }
                >
                  <button
                    type="button"
                    className={styles.sortButton}
                    onClick={() => sortBy({ kind: "gw", index })}
                  >
                    GW{gw}
                    {sameKey(sort.key, { kind: "gw", index }) && (
                      <span className={styles.sortMark}>{sort.desc ? "▼" : "▲"}</span>
                    )}
                  </button>
                </th>
              ))}
              <th
                scope="col"
                className={styles.th}
                data-numeric
                title={`Probability of 60 or more minutes in gameweek ${gameweeks[0]}, from the minutes model. Includes FPL's availability flag.`}
              >
                <span className={styles.plain}>P(60+) GW{gameweeks[0]}</span>
              </th>
            </tr>
          </thead>
          <tbody>
            {shown.map((player) => {
              const total = windowTotal(player.xp, span);
              return (
                <tr key={player.element_id} className={styles.row}>
                  <th scope="row" className={styles.name}>
                    {player.name}
                  </th>
                  <td className={styles.text}>{player.team}</td>
                  <td className={styles.text}>{player.position}</td>
                  <td className={styles.num}>{money(player.price)}</td>
                  <td className={`${styles.num} ${styles.total}`}>
                    {total === null ? "—" : total.toFixed(1)}
                  </td>
                  {gameweeks.map((gw, index) => (
                    <GameweekCell
                      key={gw}
                      xp={player.xp[index] ?? null}
                      fixtures={fixturesOf ? fixturesOf(player.team, gw) : []}
                      outside={index >= span}
                      gw={gw}
                    />
                  ))}
                  <td className={styles.num}>
                    {player.p_full === null ? "—" : `${Math.round(player.p_full * 100)}%`}
                  </td>
                </tr>
              );
            })}
            {rows.length === 0 && (
              <tr>
                <td className={styles.none} colSpan={6 + gameweeks.length}>
                  No player matches these filters.
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>

      {rows.length > shown.length && (
        <button type="button" className={styles.more} onClick={() => setShowAll(true)}>
          Show all {rows.length.toLocaleString()}
        </button>
      )}

      <p className={styles.footnote}>
        These are the model&apos;s expectations, not a recommendation. Its trailing rates do not
        update between gameweeks, so the later columns differ mostly by fixture and are weaker
        forecasts than the first. How the one-gameweek projection actually performed is on the{" "}
        <button
          type="button"
          className={styles.link}
          onClick={() => dispatch({ type: "navigate", view: "scorecard" })}
        >
          Scorecard
        </button>
        ; no accuracy has been measured for the longer horizon. P(60+) is the minutes model&apos;s
        chance of playing an hour in the first gameweek shown; the model has no separate
        probability of starting.
      </p>
    </main>
  );
}

function GameweekCell({
  xp,
  fixtures,
  outside,
  gw,
}: {
  xp: number | null;
  fixtures: ProjectionFixture[];
  outside: boolean;
  gw: number;
}) {
  const blank = fixtures.length === 0;
  const scaled = xp === null || blank ? null : shade(xp);
  const label = blank
    ? `No fixture in gameweek ${gw}`
    : fixtures
        .map(
          (f) =>
            `${f.home ? "home to" : "away at"} ${f.opponent}` +
            (f.difficulty === null ? "" : ` (FPL difficulty ${f.difficulty})`),
        )
        .join(", then ");
  return (
    <td
      className={styles.cell}
      data-blank={blank || undefined}
      data-outside={outside || undefined}
      style={
        scaled === null
          ? undefined
          : { background: outcomeColor(scaled), color: cellTextColor(scaled) }
      }
      title={`${label}${xp === null ? "" : `: ${xp.toFixed(2)} expected points`}`}
    >
      <span className={styles.xp}>{xp === null ? "—" : xp.toFixed(1)}</span>
      <span className={styles.fixtures}>
        {blank ? (
          <span className={styles.blankMark}>blank</span>
        ) : (
          fixtures.map((f, index) => (
            <span key={index} className={styles.fixture}>
              {f.opponent} {f.home ? "H" : "A"}
            </span>
          ))
        )}
      </span>
    </td>
  );
}
