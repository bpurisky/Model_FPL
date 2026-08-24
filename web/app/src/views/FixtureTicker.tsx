/**
 * The fixture ticker — clubs by gameweek, coloured by difficulty.
 *
 * **§5.16 deviation D13: this is a ninth surface, and §5.4 says "Eight."**
 * Recorded rather than slipped in, with the reason:
 *
 * Every one of the eight describes gameweeks that have *finished*. The
 * panel, `players.json` and `board.json` are all keyed to the last
 * completed gameweek by construction, the Correlation Lab pools
 * player-seasons, and the Scorecard is a backtest. The app had no surface
 * that looked forward at all — and `fixtures.json`, which is one of
 * §5.3.2's own nine exported files, was read by nothing but a contract
 * test. A committed model output with no surface is a gap whichever
 * section it falls under.
 *
 * It also matters *now* in a way the others cannot. Until the collector
 * records the first current-season gameweek, every player-keyed surface
 * necessarily describes 2025-26. The schedule does not have that problem:
 * clubs are known for 2026-27 today, so this is the one thing in the app
 * that can be about the season being played before any of it has been
 * played.
 *
 * It stays descriptive, which is what keeps it inside §5.0.2's boundary.
 * It ranks *clubs* by the run they face. It does not suggest a transfer,
 * name a captain, or propose a squad — §5.4.6's prohibitions are about
 * the model speaking, and this surface only reports a schedule and two
 * difficulty ratings that were computed elsewhere.
 *
 * **Both ratings are shown, and neither is hidden.** FPL publishes an
 * integer 1–5 per fixture; `analytics/fdr.py` derives its own from Elo.
 * The panel uses FPL's because it measured better *and* is committed data
 * (`panel.py:add_model_columns` gives the full argument). Here the reader
 * can switch, because the disagreement between them is itself
 * information — and because the Elo rating is a Phase 2 model output that
 * until now appeared on no screen.
 */

import { useMemo, useState } from "react";
import { Provenance } from "../components/Provenance";
import {
  byTeam,
  gameweekShape,
  gameweeks,
  nextGameweek,
  nextKickoff,
  runDifficulty,
  useFixtures,
  type DifficultyBasis,
  type TeamFixture,
} from "../data/fixtures";
import { divergingColor } from "../design/scale";
import { count } from "../design/text";
import styles from "./FixtureTicker.module.css";

/** How many gameweeks the grid shows at once. */
const WINDOW = 8;

/** The run length the "next N" column and the default sort use. */
const RUN = 5;

type Basis = DifficultyBasis;

export function FixtureTicker() {
  const fixtures = useFixtures();
  const [basis, setBasis] = useState<Basis>("fpl");
  const [sortByRun, setSortByRun] = useState(true);
  const [start, setStart] = useState<number | null>(null);

  const schedule = useMemo(
    () => byTeam(fixtures, fixtures?.season ?? null),
    [fixtures],
  );

  const allGameweeks = useMemo(() => gameweeks(fixtures), [fixtures]);
  const next = nextGameweek(fixtures);
  const kickoff = nextKickoff(fixtures);

  const from = start ?? next ?? allGameweeks[0] ?? 1;
  const window = allGameweeks.filter((gw) => gw >= from).slice(0, WINDOW);

  const rows = useMemo(() => {
    const teams = [...schedule.keys()];
    const scored = teams.map((team) => ({
      team,
      run: runDifficulty(schedule, team, RUN, basis),
      entries: schedule.get(team) ?? [],
    }));
    return scored.sort((left, right) => {
      if (!sortByRun) return left.team.localeCompare(right.team);
      // A club with no fixtures left sorts last rather than first — a
      // null run is "nothing to face", not "the easiest run there is".
      if (left.run === null && right.run === null) return left.team.localeCompare(right.team);
      if (left.run === null) return 1;
      if (right.run === null) return -1;
      return left.run - right.run;
    });
  }, [schedule, sortByRun, basis]);

  if (!fixtures) return <Loading />;
  if (schedule.size === 0) return <Empty />;

  const shapes = new Map(window.map((gw) => [gw, gameweekShape(schedule, gw)]));

  return (
    <main className={styles.ticker}>
      <header className={styles.head}>
        <div>
          <h1 className={styles.title}>Fixture Ticker</h1>
          <p className={styles.sub}>
            {fixtures.season}, every club by gameweek. The only surface here that looks
            forward — everything else in this app describes gameweeks that have finished.
          </p>
          {next !== null && (
            <p className={styles.now}>
              Next up: gameweek <span className="data">{next}</span>
              {kickoff && (
                <>
                  , first kickoff{" "}
                  <span className="data">
                    {new Date(kickoff).toISOString().slice(0, 16).replace("T", " ")}Z
                  </span>
                </>
              )}
              .
            </p>
          )}
        </div>
        <Provenance header={fixtures.header} basis={fixtures.header.normalization_basis} />
      </header>

      <div className={styles.controls}>
        <fieldset className={styles.group}>
          <legend className={styles.legend}>Difficulty from</legend>
          <div className={styles.chips}>
            <button
              type="button"
              className={styles.chip}
              data-on={basis === "fpl" || undefined}
              onClick={() => setBasis("fpl")}
              title="FPL's own published 1-5 rating. What the panel's clean-sheet model uses, because it measured better and is committed data."
            >
              FPL
            </button>
            <button
              type="button"
              className={styles.chip}
              data-on={basis === "custom" || undefined}
              onClick={() => setBasis("custom")}
              title="This repo's Elo-derived rating from analytics/fdr.py, on the same 1-5 scale."
            >
              Elo
            </button>
          </div>
        </fieldset>

        <fieldset className={styles.group}>
          <legend className={styles.legend}>Order</legend>
          <div className={styles.chips}>
            <button
              type="button"
              className={styles.chip}
              data-on={sortByRun || undefined}
              onClick={() => setSortByRun(true)}
            >
              kindest run
            </button>
            <button
              type="button"
              className={styles.chip}
              data-on={!sortByRun || undefined}
              onClick={() => setSortByRun(false)}
            >
              club
            </button>
          </div>
        </fieldset>

        <fieldset className={styles.group}>
          <legend className={styles.legend}>From gameweek</legend>
          <div className={styles.chips}>
            <button
              type="button"
              className={styles.chip}
              onClick={() => setStart(Math.max(from - WINDOW, allGameweeks[0] ?? 1))}
              disabled={from <= (allGameweeks[0] ?? 1)}
            >
              ←
            </button>
            <span className={styles.range}>
              {window[0]}–{window[window.length - 1]}
            </span>
            <button
              type="button"
              className={styles.chip}
              onClick={() => setStart(from + WINDOW)}
              disabled={window.length < WINDOW}
            >
              →
            </button>
            {next !== null && from !== next && (
              <button type="button" className={styles.chip} onClick={() => setStart(null)}>
                back to now
              </button>
            )}
          </div>
        </fieldset>

        <Key />
      </div>

      <div className={styles.scroll}>
        <table className={styles.grid}>
          <caption className={styles.caption}>
            {count(rows.length, "club")}, coloured by{" "}
            {basis === "fpl" ? "FPL's published difficulty" : "Elo-derived difficulty"}. Lower
            is kinder.
          </caption>
          <thead>
            <tr>
              <th scope="col" className={styles.corner}>
                Club
              </th>
              <th
                scope="col"
                className={styles.runHead}
                title={`Mean ${basis === "fpl" ? "FPL" : "Elo"} difficulty over the next ${RUN} unplayed fixtures. Switching the basis re-ranks the table.`}
              >
                Next {RUN}
                <span className={styles.basisMark}>{basis === "fpl" ? "FPL" : "Elo"}</span>
              </th>
              {window.map((gw) => (
                <th key={gw} scope="col" className={styles.gwHead}>
                  {gw}
                  {next === gw && <span className={styles.nowMark} aria-label="next gameweek" />}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.map((row) => (
              <tr key={row.team} className={styles.row}>
                <th scope="row" className={styles.club}>
                  {row.team}
                </th>
                <td className={styles.run}>
                  <span className="data">{row.run === null ? "—" : row.run.toFixed(2)}</span>
                </td>
                {window.map((gw) => {
                  const inWeek = row.entries.filter((entry) => entry.gw === gw);
                  return (
                    <td
                      key={gw}
                      className={styles.cell}
                      data-blank={inWeek.length === 0 || undefined}
                      data-double={inWeek.length > 1 || undefined}
                    >
                      {inWeek.length === 0 ? (
                        <span className={styles.blank} title={`${row.team} has no fixture in gameweek ${gw}.`}>
                          —
                        </span>
                      ) : (
                        inWeek.map((entry) => (
                          <Cell key={`${entry.opponent}-${entry.home}`} entry={entry} basis={basis} />
                        ))
                      )}
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <p className={styles.footnote}>
        Blanks and doubles are read from the schedule, so a club with two fixtures in a week
        shows both and one with none shows an em dash rather than an empty cell that could be
        mistaken for missing data. A played fixture is dimmed:{" "}
        <span className="data">difficulty_basis</span> records that its rating is what each club
        carried <em>into</em> the match, not what it holds now.
        {shapes.size > 0 && (
          <>
            {" "}
            In this window,{" "}
            <span className="data">
              {[...shapes.entries()].filter(([, shape]) => shape.blank.size > 0).length}
            </span>{" "}
            {count(
              [...shapes.entries()].filter(([, shape]) => shape.blank.size > 0).length,
              "gameweek",
            ).split(" ")[1]}{" "}
            contain a blank.
          </>
        )}
      </p>
    </main>
  );
}

function Cell({ entry, basis }: { entry: TeamFixture; basis: Basis }) {
  const value = basis === "fpl" ? entry.difficulty : entry.custom;

  /*
   * The §5.8.2 scale, oriented so a hard fixture reads as the negative
   * pole. Difficulty runs 1 to 5 with 3 as neutral, so the midpoint of
   * the ramp is 3 and the sign is inverted — low is good here, which is
   * exactly what `lower_is_better` means elsewhere in the app.
   */
  const scaled = value === null ? null : (3 - value) / 2;

  return (
    <span
      className={styles.fixture}
      data-played={entry.played || undefined}
      style={scaled === null ? undefined : { background: divergingColor(scaled) }}
      title={
        `${entry.home ? "vs" : "at"} ${entry.opponent}, gameweek ${entry.gw}` +
        (entry.difficulty === null ? "" : ` — FPL ${entry.difficulty}`) +
        (entry.custom === null ? "" : `, Elo ${entry.custom.toFixed(2)}`) +
        (entry.played ? " (played; rating is what each club carried into it)" : "")
      }
    >
      <span className={styles.opponent}>{short(entry.opponent)}</span>
      <span className={styles.venue}>{entry.home ? "H" : "A"}</span>
    </span>
  );
}

/**
 * Club names to three letters, so twenty of them fit across eight
 * gameweeks. Derived rather than mapped: a hard-coded table is wrong
 * three times a season on promotion and relegation.
 */
function short(team: string): string {
  const words = team.split(/\s+/).filter(Boolean);
  if (words.length >= 2) {
    // "Man City" -> MCI, "Nott'm Forest" -> NFO.
    return (words[0]!.slice(0, 1) + words[1]!.slice(0, 2)).toUpperCase();
  }
  return team.slice(0, 3).toUpperCase();
}

function Key() {
  return (
    <ul className={styles.key} aria-label="Cell states">
      <li className={styles.keyItem}>
        <span className={styles.keySwatch} data-kind="easy" /> kind
      </li>
      <li className={styles.keyItem}>
        <span className={styles.keySwatch} data-kind="hard" /> hard
      </li>
      <li className={styles.keyItem}>
        <span className={styles.keySwatch} data-kind="played" /> played
      </li>
      <li className={styles.keyItem}>
        <span className={styles.keySwatch} data-kind="blank" /> blank
      </li>
    </ul>
  );
}

function Loading() {
  return (
    <main className={styles.ticker}>
      <h1 className={styles.title}>Fixture Ticker</h1>
      <p className={styles.sub}>Reading the schedule…</p>
    </main>
  );
}

/** §5.14.8's register: what is absent, and what to run. */
function Empty() {
  return (
    <main className={styles.ticker}>
      <h1 className={styles.title}>Fixture Ticker</h1>
      <div className={styles.absent}>
        <p className={styles.absentHead}>No schedule to show.</p>
        <p className={styles.sub}>
          This surface reads <span className="data">fixtures.json</span>, which is committed —
          so this state means the file did not load or carries no fixtures for its own season.
          Run <code className="data">python -m web.export fixtures</code> and reload.
        </p>
      </div>
    </main>
  );
}
