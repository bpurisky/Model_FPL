/**
 * Captaincy: the model's captain candidates for the next gameweek, and the
 * public record of every pick it has made.
 *
 * **A recorded deviation from §5.0.2 (surfaces make no calls), D18.** Like
 * D15 and D17 this surface states a model call, here the player the
 * event model ranks first by expected points. The condition on which it
 * ships is the one the banner states: every pick is frozen before the
 * deadline in the paper trade's own freeze and scored after the gameweek,
 * in the same file, against two baselines (the most-owned player, and the
 * best of the model's own top five in hindsight). The call and the
 * evidence it has earned are not read apart, and the record is the second
 * thing on the page.
 *
 * **The pick is just the top of the xP ranking.** There is no
 * differential, fixture-swing or ownership adjustment; whether plain xP is
 * a good captain rule is what the record measures. P(haul) is the chance of
 * `haul_threshold` or more points, from the same model's components (see
 * `analytics/haul.py`), and is itself scored: the record reports its Brier
 * score next to the base-rate Brier it has to beat.
 *
 * **An empty record is stated, not hidden.** Until a frozen pick's
 * gameweek has finished there is nothing to score and the page says so.
 */

import { useEffect, useState } from "react";
import { ErrorState } from "../components/ErrorState";
import { LoadingState } from "../components/LoadingState";
import { Provenance } from "../components/Provenance";
import { loadCaptaincy, type LoadProgress } from "../data/load";
import type { CaptaincyCandidate, CaptaincyFile, CaptaincyGw, CaptaincyTrackRecord } from "../data/schema";
import { count } from "../design/text";
import styles from "./Captaincy.module.css";

type State =
  | { status: "loading"; progress: LoadProgress | null }
  | { status: "error"; error: Error }
  | { status: "ready"; file: CaptaincyFile };

/** The id of the track record section, which the banner scrolls to. */
export const RECORD_ID = "captaincy-record";

// --- pure logic (tested directly) -----------------------------------------

/** "ARS (H)", "ARS (H) + MCI (A)" for a double; "no fixture" for a blank. */
export function formatOpponents(opponents: CaptaincyCandidate["opponents"]): string {
  if (opponents.length === 0) return "no fixture";
  return opponents.map((o) => `${o.opponent} (${o.home ? "H" : "A"})`).join(" + ");
}

export const pct = (value: number, digits = 0): string => `${(value * 100).toFixed(digits)}%`;

/**
 * Bar fill 0..1 for a P(haul). The scale tops out at `ceiling` rather than
 * at 1 so the differences between candidates (a few percentage points to
 * about a fifth) are visible; the number beside the bar carries the value.
 */
export function haulFill(pHaul: number, ceiling = 0.4): number {
  if (!Number.isFinite(pHaul)) return 0;
  return Math.min(1, Math.max(0, pHaul / ceiling));
}

/** Points are ints; the means are not. */
export const points = (value: number | null): string => (value === null ? "n/a" : value.toFixed(1));

/**
 * One honest sentence about the record: a count when there is nothing to
 * compare yet, otherwise the pick's mean (doubled) next to the baselines'.
 */
export function recordHeadline(record: CaptaincyTrackRecord): string {
  if (record.n_gameweeks === 0) {
    return "No pick has been scored yet. The first scored gameweek appears here once its results are in.";
  }
  return (
    `Over ${count(record.n_gameweeks, "scored gameweek")}, the model's pick averaged ${points(record.mean_pick_points_x2)} captain points, ` +
    `the most-owned player ${points(record.mean_most_owned_points_x2)}, and the best of the model's top five in hindsight ` +
    `${points(record.mean_best_of_shortlist_points_x2)}.`
  );
}

/**
 * How P(haul) has scored: the Brier score beside the base-rate Brier it
 * has to beat. Null until something is scored.
 */
export function calibrationLine(record: CaptaincyTrackRecord): string | null {
  const c = record.calibration;
  if (c.n === 0 || c.brier === null || c.brier_climatology === null) return null;
  const verdict = c.brier < c.brier_climatology ? "better than" : "no better than";
  return (
    `P(haul), scored over ${count(c.n, "frozen candidate")}: Brier ${c.brier.toFixed(3)}, ${verdict} ` +
    `always predicting the base rate (${c.brier_climatology.toFixed(3)}). Mean P(haul) ${pct(c.mean_p_haul ?? 0, 1)} against a ` +
    `realized haul rate of ${pct(c.haul_rate ?? 0, 1)}.`
  );
}

/** The pick's result beside the most-owned player's: wins, ties, losses. */
export function pickVersusMostOwned(rows: CaptaincyGw[]): { won: number; tied: number; lost: number } {
  let won = 0;
  let tied = 0;
  let lost = 0;
  for (const row of rows) {
    if (row.most_owned_points_x2 === null) continue;
    if (row.pick_points_x2 > row.most_owned_points_x2) won += 1;
    else if (row.pick_points_x2 === row.most_owned_points_x2) tied += 1;
    else lost += 1;
  }
  return { won, tied, lost };
}

// --- view ------------------------------------------------------------------

export function Captaincy() {
  const [data, setData] = useState<State>({ status: "loading", progress: null });

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const file = await loadCaptaincy((progress) =>
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

  if (data.status === "loading")
    return (
      <LoadingState
        rootClassName={styles.captaincy}
        title="Captaincy"
        file="captaincy.json"
        label="Loading the captaincy table"
        progress={data.progress}
      />
    );
  if (data.status === "error")
    return (
      <ErrorState
        rootClassName={styles.captaincy}
        title="Captaincy"
        heading="captaincy.json did not load."
        command="captaincy"
        error={data.error}
      />
    );

  const { file } = data;
  return (
    <main className={styles.captaincy}>
      <header className={styles.head}>
        <div>
          <h1 className={styles.title}>Captaincy</h1>
          <p className={styles.sub}>
            The model&rsquo;s captain candidates for gameweek {file.gameweek}, ranked by expected points,
            with the chance each has of a haul ({file.haul_threshold} or more points).
          </p>
        </div>
        <Provenance header={file.header} basis={file.header.normalization_basis} />
      </header>

      <Banner file={file} />
      <Candidates file={file} />
      <Record file={file} />
    </main>
  );
}

function Banner({ file }: { file: CaptaincyFile }) {
  const scrollToRecord = () =>
    document.getElementById(RECORD_ID)?.scrollIntoView({ behavior: "smooth", block: "start" });
  return (
    <section className={styles.banner} aria-labelledby="banner-heading" data-scored={file.track_record.n_gameweeks > 0 || undefined}>
      <h2 id="banner-heading" className={styles.bannerTitle}>
        A model pick, scored in public
      </h2>
      <p className={styles.bannerLead}>
        This is the one place the app names a player for you. It ships on a condition: the top pick
        every gameweek is frozen before the deadline and scored after it, against the most-owned
        player and the best of the model&rsquo;s own top five, the same way the paper trade scores
        everything else. The pick is simply the top of the expected-points ranking; nothing is
        adjusted for ownership or differentials.
      </p>
      <p className={styles.headline}>{recordHeadline(file.track_record)}</p>
      <button type="button" className={styles.jump} onClick={scrollToRecord}>
        See the full track record
      </button>
    </section>
  );
}

function Candidates({ file }: { file: CaptaincyFile }) {
  return (
    <section className={styles.panel} aria-labelledby="candidates-heading">
      <h2 id="candidates-heading" className={styles.panelTitle}>
        Gameweek {file.gameweek} candidates
      </h2>
      <p className={styles.panelSub}>
        The top {file.candidates.length} by expected points. The first row is the model&rsquo;s pick.
        A double gameweek sums both fixtures. P(haul) is the model&rsquo;s own tail, not a second
        model; it ignores bonus-point and goals-conceded randomness beyond their averages and is
        scored below.
      </p>
      {file.candidates.length === 0 ? (
        <p className={styles.empty}>No candidates: the season has no further gameweek.</p>
      ) : (
        <div className={styles.tableScroll}>
          <table className={styles.table}>
            <thead>
              <tr>
                <th scope="col">#</th>
                <th scope="col" className={styles.left}>Player</th>
                <th scope="col" className={styles.left}>Club</th>
                <th scope="col" className={styles.left}>Opponent</th>
                <th scope="col">xP</th>
                <th scope="col" className={styles.left}>P(haul)</th>
                <th scope="col">Owned</th>
              </tr>
            </thead>
            <tbody>
              {file.candidates.map((c, i) => (
                <tr key={c.element_id} data-pick={i === 0 || undefined}>
                  <td>{i + 1}</td>
                  <th scope="row" className={styles.left}>
                    {c.name} <span className={styles.pos}>{c.position}</span>
                  </th>
                  <td className={styles.left}>{c.team}</td>
                  <td className={styles.left}>{formatOpponents(c.opponents)}</td>
                  <td>{c.xp.toFixed(1)}</td>
                  <td className={styles.left}>
                    <span className={styles.progress}>
                      <span className={styles.bar} role="img" aria-label={`${pct(c.p_haul, 1)} chance of a haul`}>
                        <span className={styles.fill} style={{ inlineSize: `${haulFill(c.p_haul) * 100}%` }} />
                      </span>
                      <span className="data">{pct(c.p_haul)}</span>
                    </span>
                  </td>
                  <td>{c.ownership === null ? "n/a" : `${c.ownership.toFixed(1)}%`}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </section>
  );
}

function Record({ file }: { file: CaptaincyFile }) {
  const record = file.track_record;
  const calibration = calibrationLine(record);
  const versus = pickVersusMostOwned(record.gameweeks);
  return (
    <section id={RECORD_ID} className={styles.panel} aria-labelledby="record-heading">
      <h2 id="record-heading" className={styles.panelTitle}>
        Track record
      </h2>
      <p className={styles.panelSub}>
        Captain points are doubled. &ldquo;Best of top 5&rdquo; is hindsight, an upper bound for the
        model&rsquo;s own shortlist, not a pick anyone could have made in advance. A captain who does not
        play is scored as the 0 he got; the vice-captain&rsquo;s cover is not modelled.
      </p>
      <p className={styles.headline}>{recordHeadline(record)}</p>
      {record.gameweeks.length > 0 && (
        <p className={styles.note}>
          Pick against most-owned: {versus.won} won, {versus.tied} tied, {versus.lost} lost. The pick hauled in{" "}
          {record.pick_haul_rate === null ? "n/a" : pct(record.pick_haul_rate)} of scored gameweeks.
        </p>
      )}
      {calibration && <p className={styles.note}>{calibration}</p>}

      {record.gameweeks.length > 0 && (
        <div className={styles.tableScroll}>
          <table className={styles.table}>
            <thead>
              <tr>
                <th scope="col">GW</th>
                <th scope="col" className={styles.left}>Model pick</th>
                <th scope="col">P(haul)</th>
                <th scope="col">Points (x2)</th>
                <th scope="col" className={styles.left}>Most owned</th>
                <th scope="col">Points (x2)</th>
                <th scope="col">Best of top 5 (x2)</th>
              </tr>
            </thead>
            <tbody>
              {[...record.gameweeks].reverse().map((g) => (
                <tr key={g.gw}>
                  <th scope="row">{g.gw}</th>
                  <td className={styles.left}>{g.pick.name ?? g.pick.element_id}</td>
                  <td>{pct(g.pick.p_haul)}</td>
                  <td>
                    {g.pick_points} ({g.pick_points_x2})
                  </td>
                  <td className={styles.left}>{g.most_owned?.name ?? "n/a"}</td>
                  <td>
                    {g.most_owned_points === null ? "n/a" : `${g.most_owned_points} (${g.most_owned_points_x2})`}
                  </td>
                  <td>
                    {g.best_of_shortlist_points} ({g.best_of_shortlist_points_x2})
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {record.skipped.length > 0 && (
        <p className={styles.note}>
          Not scored: {record.skipped.map((s) => `gw${s.gw} (${s.reason})`).join(", ")}. Freezes are
          immutable, so those written before the captaincy record existed stay without one.
        </p>
      )}
    </section>
  );
}
