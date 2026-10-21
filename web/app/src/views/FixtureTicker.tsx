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
 *
 * **The window is the reader's.** Start gameweek and length (1–8) choose
 * which gameweeks both the grid and the ranking cover, and the same window
 * feeds the rotation-pair table. Blank and double gameweeks are tagged in
 * the header. The scoring rules — a blank is worse than any fixture, a
 * double counts both — live in `data/fixtureWindow.ts`, where they are
 * tested; this file only draws them.
 */

import { useMemo, useState } from "react";
import { Provenance } from "../components/Provenance";
import {
  byTeam,
  gameweekShape,
  gameweeks,
  nextGameweek,
  nextKickoff,
  useFixtures,
  type DifficultyBasis,
  type TeamFixture,
} from "../data/fixtures";
import {
  BLANK_VALUE,
  MAX_WINDOW,
  gameweekBadges,
  rankClubs,
  rotationPairs,
  windowGameweeks,
  type Aggregate,
  type GameweekBadge,
  type RotationPair,
} from "../data/fixtureWindow";
import { cellTextColor, outcomeColor } from "../design/scale";
import { count } from "../design/text";
import styles from "./FixtureTicker.module.css";

/** The window length the ticker opens with. */
const DEFAULT_LENGTH = 5;

/** How many rotation pairs to list. */
const PAIRS = 10;

const LENGTHS = Array.from({ length: MAX_WINDOW }, (_, i) => i + 1);

type Basis = DifficultyBasis;

export function FixtureTicker() {
  const fixtures = useFixtures();
  const [basis, setBasis] = useState<Basis>("fpl");
  const [sortByRun, setSortByRun] = useState(true);
  const [start, setStart] = useState<number | null>(null);
  const [length, setLength] = useState(DEFAULT_LENGTH);
  const [aggregate, setAggregate] = useState<Aggregate>("mean");

  const schedule = useMemo(
    () => byTeam(fixtures, fixtures?.season ?? null),
    [fixtures],
  );

  const allGameweeks = useMemo(() => gameweeks(fixtures), [fixtures]);
  const next = nextGameweek(fixtures);
  const kickoff = nextKickoff(fixtures);

  const from = start ?? next ?? allGameweeks[0] ?? 1;
  const window = useMemo(
    () => windowGameweeks(allGameweeks, from, length),
    [allGameweeks, from, length],
  );

  const rows = useMemo(() => {
    // `rankClubs` puts an unscorable club last rather than first — a
    // null score is "nothing to rate", not "the easiest run there is".
    const ranked = rankClubs(schedule, window, basis, aggregate).map(({ team, score }) => ({
      team,
      run: score,
      entries: schedule.get(team) ?? [],
    }));
    if (sortByRun) return ranked;
    return [...ranked].sort((left, right) => left.team.localeCompare(right.team));
  }, [schedule, window, sortByRun, basis, aggregate]);

  const pairs = useMemo(
    () => rotationPairs(schedule, window, basis, PAIRS),
    [schedule, window, basis],
  );
  const badges = useMemo(() => gameweekBadges(schedule, window), [schedule, window]);

  if (!fixtures) return <Loading />;
  if (schedule.size === 0) return <Empty />;

  const shapes = new Map(window.map((gw) => [gw, gameweekShape(schedule, gw)]));
  const badgeFor = new Map(badges.map((badge) => [badge.gw, badge]));
  const basisName = basis === "fpl" ? "FPL" : "Elo";

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
          <legend className={styles.legend}>Window</legend>
          <div className={styles.chips}>
            <label className={styles.range}>
              from GW{" "}
              <select
                className={styles.select}
                value={from}
                onChange={(event) => setStart(Number(event.target.value))}
              >
                {allGameweeks.map((gw) => (
                  <option key={gw} value={gw}>
                    {gw}
                  </option>
                ))}
              </select>
            </label>
            {next !== null && from !== next && (
              <button type="button" className={styles.chip} onClick={() => setStart(null)}>
                back to now
              </button>
            )}
          </div>
        </fieldset>

        <fieldset className={styles.group}>
          <legend className={styles.legend}>Length (gameweeks)</legend>
          <div className={styles.chips}>
            {LENGTHS.map((n) => (
              <button
                key={n}
                type="button"
                className={styles.chip}
                data-on={length === n || undefined}
                onClick={() => setLength(n)}
              >
                {n}
              </button>
            ))}
          </div>
        </fieldset>

        <fieldset className={styles.group}>
          <legend className={styles.legend}>Score</legend>
          <div className={styles.chips}>
            <button
              type="button"
              className={styles.chip}
              data-on={aggregate === "mean" || undefined}
              onClick={() => setAggregate("mean")}
              title="Mean difficulty per fixture. A double counts both fixtures; a blank counts as worse than any fixture."
            >
              mean
            </button>
            <button
              type="button"
              className={styles.chip}
              data-on={aggregate === "sum" || undefined}
              onClick={() => setAggregate("sum")}
              title="Total difficulty. A double adds two ratings, so it can total more than a blank; mean is the fairer comparison across clubs with doubles."
            >
              sum
            </button>
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
                title={`${aggregate === "mean" ? "Mean" : "Summed"} ${basisName} difficulty over gameweeks ${window[0]}–${window[window.length - 1]}. A blank counts as worse than any fixture. Switching the basis, window or score re-ranks the table.`}
              >
                {window.length} GW {aggregate}
                <span className={styles.basisMark}>{basisName}</span>
              </th>
              {window.map((gw) => (
                <th key={gw} scope="col" className={styles.gwHead}>
                  {gw}
                  <Badges badge={badgeFor.get(gw)} />
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
                  <span className="data">
                    {row.run === null ? "—" : row.run.toFixed(aggregate === "sum" ? 1 : 2)}
                  </span>
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

      <RotationPairs pairs={pairs} window={window} basis={basis} />

      <p className={styles.footnote}>
        The score ranks clubs on one difficulty number per fixture. A blank counts as one slot
        worse than the hardest rating, never as an easy week, and a double counts both fixtures.
        There is no attack-versus-defence split: the export holds a single rating per side, and
        a separate attacking or defensive one would be new modelling. Blanks and doubles are read from the schedule, so a club with two fixtures in a week
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

/** DGW / BGW tags under a gameweek number; the club names are in the tooltip. */
function Badges({ badge }: { badge: GameweekBadge | undefined }) {
  if (!badge) return null;
  return (
    <>
      {badge.double.length > 0 && (
        <span
          className={styles.badge}
          data-kind="double"
          title={`Double gameweek: ${badge.double.join(", ")}`}
        >
          DGW {badge.double.length}
        </span>
      )}
      {badge.blank.length > 0 && (
        <span
          className={styles.badge}
          data-kind="blank"
          title={`Blank gameweek: ${badge.blank.join(", ")}`}
        >
          BGW {badge.blank.length}
        </span>
      )}
    </>
  );
}

/**
 * The best club pairings for rotation over the window: each gameweek takes
 * the easier of the two, and that pick is highlighted. It describes how two
 * schedules interlock; it does not say whom to own (§5.0.2).
 */
function RotationPairs({
  pairs,
  window,
  basis,
}: {
  pairs: RotationPair[];
  window: number[];
  basis: Basis;
}) {
  if (pairs.length === 0) return null;
  return (
    <section className={styles.pairs} aria-label="Rotation pairs">
      <h2 className={styles.pairsTitle}>Rotation pairs</h2>
      <p className={styles.sub}>
        Two clubs whose fixtures interlock: each gameweek takes the easier of the two (
        {basis === "fpl" ? "FPL" : "Elo"} difficulty, highlighted), and the pair is scored on the
        mean of those picks. A week where both clubs are blank counts as a blank.
      </p>
      <div className={styles.scroll}>
        <table className={styles.grid}>
          <thead>
            <tr>
              <th scope="col" className={styles.corner}>
                Pair
              </th>
              <th scope="col" className={styles.runHead}>
                Mean pick
              </th>
              {window.map((gw) => (
                <th key={gw} scope="col" className={styles.gwHead}>
                  {gw}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {pairs.map((pair) => (
              <tr key={`${pair.a}|${pair.b}`} className={styles.row}>
                <th scope="row" className={styles.club}>
                  {pair.a} + {pair.b}
                </th>
                <td className={styles.run}>
                  <span className="data">{pair.score.toFixed(2)}</span>
                </td>
                {pair.picks.map((pick) => (
                  <td key={pick.gw} className={styles.cell}>
                    {pick.team === null ? (
                      <span className={styles.blank}>—</span>
                    ) : (
                      <span className={styles.pick} title={`Gameweek ${pick.gw}: ${pick.team}`}>
                        {short(pick.team)}
                        <span className={styles.venue}>
                          {pick.value === null
                            ? ""
                            : pick.value >= BLANK_VALUE
                              ? "BGW"
                              : pick.value.toFixed(1)}
                        </span>
                      </span>
                    )}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}

function Cell({ entry, basis }: { entry: TeamFixture; basis: Basis }) {
  const value = basis === "fpl" ? entry.difficulty : entry.custom;

  /*
   * FPL's own convention: an easy fixture green, a hard one red.
   * Difficulty runs 1 to 5 with 3 as neutral, so the midpoint of the ramp
   * is 3 and the sign is inverted, because low difficulty is good.
   */
  const scaled = value === null ? null : (3 - value) / 2;

  return (
    <span
      className={styles.fixture}
      data-played={entry.played || undefined}
      style={
        scaled === null
          ? undefined
          : { background: outcomeColor(scaled), color: cellTextColor(scaled) }
      }
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
