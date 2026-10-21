/**
 * Price Watch: the rise-only price model's call for tonight's price
 * changes, and the evidence for it.
 *
 * **A recorded deviation from §5.0.2 (surfaces make no calls).** Like D15
 * for the Squad Optimizer, this surface states a model call. The
 * justification is the same shape: it names no squad, captain or transfer
 * and decides nothing for the reader; it reports one mechanical rule (a
 * player whose 24h net transfers are at or above a fitted cut, and whose
 * price has not moved in 72h, is called "rise"), and the rule's own
 * out-of-sample record is the first thing on the page, ahead of every
 * call. The claim and the evidence for it are not read apart.
 *
 * **Rise-only, and says so.** The model never calls falls: the fall signal
 * was too weak to be worth a claim. The header says that plainly rather
 * than leaving a reader to wonder why nobody is ever flagged to drop.
 *
 * **The headline is out-of-sample, or it is "not enough yet".** The fit
 * statistics from `config/price_model.yaml` are in-sample (the threshold
 * was chosen to maximise them), so they are shown, labelled, below the
 * headline and never as the model's hit rate. Until `min_calls` scored
 * rise calls exist the headline is a count, not a rate.
 *
 * Flat on `--ground`, not walled (§5.8.6): it reports what a rule says
 * about the numbers on disk, like Scorecard.
 */

import { useEffect, useMemo, useState } from "react";
import { ErrorState } from "../components/ErrorState";
import { LoadingState } from "../components/LoadingState";
import { Provenance } from "../components/Provenance";
import { loadPrices, type LoadProgress } from "../data/load";
import type { PriceTrackRecord, PriceWatchPlayer, PricesFile } from "../data/schema";
import { count } from "../design/text";
import styles from "./PriceWatch.module.css";

type State =
  | { status: "loading"; progress: LoadProgress | null }
  | { status: "error"; error: Error }
  | { status: "ready"; file: PricesFile };

const POSITIONS = ["GK", "DEF", "MID", "FWD"] as const;
/** Rows shown before "show all": the top of the ranking is the point. */
const INITIAL_ROWS = 60;

// --- pure logic (tested directly) -----------------------------------------

export interface PriceFilters {
  position: string;
  team: string;
  query: string;
}

export const NO_PRICE_FILTERS: PriceFilters = { position: "all", team: "all", query: "" };

/** Position, team and a case-insensitive name search; order is preserved
 * (the file arrives sorted by progress). */
export function filterPlayers(players: PriceWatchPlayer[], filters: PriceFilters): PriceWatchPlayer[] {
  const query = filters.query.trim().toLowerCase();
  return players.filter(
    (player) =>
      (filters.position === "all" || player.position === filters.position) &&
      (filters.team === "all" || player.team === filters.team) &&
      (query === "" || player.name.toLowerCase().includes(query)),
  );
}

/** Bar fill 0..1. A negative progress (net outflow) is an empty bar, and
 * past the cut the bar is full: the number beside it carries the rest. */
export function barFill(progress: number | null): number {
  if (progress === null || !Number.isFinite(progress)) return 0;
  return Math.min(1, Math.max(0, progress));
}

export function progressPercent(progress: number | null): string {
  return progress === null ? "n/a" : `${Math.round(progress * 100)}%`;
}

export function reasonLabel(reason: PriceWatchPlayer["ineligible_reason"], hours: number): string {
  if (reason === "moved_within_72h") return `price moved in the last ${hours}h`;
  if (reason === "insufficient_history") return "not enough snapshot history";
  return "";
}

const pct = (value: number) => `${Math.round(value * 100)}%`;

/**
 * The honesty headline. Out-of-sample hit rate with its interval and n,
 * or an explicit "not enough scored calls yet (n/min)" -- never a bare
 * rate over too few calls, which is what the gate itself refuses to do.
 */
export function trackRecordHeadline(record: PriceTrackRecord): { text: string; scored: boolean } {
  if (record.status === "insufficient data" || record.hit_rate === null) {
    return {
      text: `Not enough scored calls yet (${record.n_calls}/${record.min_calls}). No out-of-sample hit rate is reported until there are ${record.min_calls}.`,
      scored: false,
    };
  }
  const interval =
    record.ci_low !== null && record.ci_high !== null
      ? ` (95% CI ${pct(record.ci_low)} to ${pct(record.ci_high)})`
      : "";
  return {
    text: `Out-of-sample hit rate ${pct(record.hit_rate)}${interval} over ${count(record.n_calls, "rise call")}.`,
    scored: true,
  };
}

export function formatUtc(iso: string | null): string {
  if (iso === null) return "n/a";
  return `${new Date(iso).toISOString().slice(0, 16).replace("T", " ")} UTC`;
}

// --- view ------------------------------------------------------------------

export function PriceWatch() {
  const [data, setData] = useState<State>({ status: "loading", progress: null });

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const file = await loadPrices((progress) =>
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

  if (data.status === "loading") return (
    <LoadingState
      rootClassName={styles.priceWatch}
      title="Price Watch"
      file="prices.json"
      label="Loading price watch"
      progress={data.progress}
    />
  );
  if (data.status === "error") return (
    <ErrorState
      rootClassName={styles.priceWatch}
      title="Price Watch"
      heading="prices.json did not load."
      command="prices"
      error={data.error}
    />
  );

  const { file } = data;

  return (
    <main className={styles.priceWatch}>
      <header className={styles.head}>
        <div>
          <h1 className={styles.title}>Price Watch</h1>
          <p className={styles.sub}>
            Which players the model calls to rise in price tonight, from their raw net
            transfers over the last 24 hours. This reports the model&rsquo;s call; it does not
            suggest a transfer.
          </p>
        </div>
        <Provenance header={file.header} basis={file.header.normalization_basis} />
      </header>

      <Honesty file={file} />
      {file.status === "ok" ? <Cycle file={file} /> : <Unavailable file={file} />}
      <History file={file} />
    </main>
  );
}

function Honesty({ file }: { file: PricesFile }) {
  const record = file.track_record;
  const headline = trackRecordHeadline(record);
  const fit = record.in_sample;

  return (
    <section className={styles.honesty} aria-labelledby="honesty-heading" data-scored={headline.scored || undefined}>
      <h2 id="honesty-heading" className={styles.honestyTitle}>
        What this model does and does not claim
      </h2>
      <p className={styles.honestyLead}>
        <strong>Rise-only.</strong> It never predicts falls: the fall signal in this data was
        too weak to call. A call is &ldquo;rise&rdquo; only for a player at or above{" "}
        <span className="data">{file.threshold.toLocaleString()}</span> net transfers in 24h
        whose price has not moved in the last {file.recent_move_hours}h.
      </p>
      <p className={styles.headline}>{headline.text}</p>
      <p className={styles.note}>
        Scored only on cycles after the threshold was fitted
        {record.span ? ` (${record.span}, ${count(record.n_cycles, "overnight cycle")})` : ""}; a call is a hit if
        the price rose overnight. Launch-gate status: <span className="data">{record.status}</span>.
      </p>
      {fit && (
        <p className={styles.note}>
          In-sample, for reference only: the threshold was chosen to maximise these over{" "}
          {fit.fit_span ?? "its fit span"}
          {fit.fit_cycles !== null ? ` (${count(fit.fit_cycles, "cycle")})` : ""}, so they say how well a rule can be
          made to fit the past, not how it will do.
          {fit.hit_rate !== null && <> Hit rate {pct(fit.hit_rate)}</>}
          {fit.rise_n_called !== null && <> over {count(fit.rise_n_called, "call")}</>}
          {fit.rise_f1 !== null && <>, F1 {fit.rise_f1.toFixed(2)}</>}.
        </p>
      )}
    </section>
  );
}

function Unavailable({ file }: { file: PricesFile }) {
  return (
    <section className={styles.panel} aria-labelledby="unavailable-heading">
      <h2 id="unavailable-heading" className={styles.panelTitle}>
        No call right now
      </h2>
      <p className={styles.empty}>
        {file.status === "stale"
          ? "The collector's latest snapshot is too old to be a current reading, so the model is not predicting from it. "
          : "The model cannot make a call from the snapshots on disk. "}
        {file.status_detail ?? ""}
        {file.status_detail && !file.status_detail.endsWith(".") ? "." : ""}
      </p>
    </section>
  );
}

function Cycle({ file }: { file: PricesFile }) {
  const [filters, setFilters] = useState<PriceFilters>(NO_PRICE_FILTERS);
  const [showAll, setShowAll] = useState(false);

  const teams = useMemo(() => [...new Set(file.players.map((p) => p.team))].sort(), [file.players]);
  const visible = useMemo(() => filterPlayers(file.players, filters), [file.players, filters]);
  const rows = showAll ? visible : visible.slice(0, INITIAL_ROWS);
  const called = file.players.filter((p) => p.call === "rise").length;
  const patch = (next: Partial<PriceFilters>) => {
    setFilters({ ...filters, ...next });
    setShowAll(false);
  };

  return (
    <section className={styles.panel} aria-labelledby="cycle-heading">
      <h2 id="cycle-heading" className={styles.panelTitle}>
        {file.locked ? "Tonight’s calls (locked)" : "Tonight’s calls (running total)"}
      </h2>
      <p className={styles.panelSub}>
        Read from the snapshot at <span className="data">{formatUtc(file.snapshot_ts)}</span>.{" "}
        {file.locked
          ? `The ${formatUtc(file.boundary_ts)} cut has passed, so these calls are fixed and the overnight change is pending or done. `
          : `Prices move overnight after the ${formatUtc(file.boundary_ts)} cut; scores can still move before then. `}
        The next cycle opens at <span className="data">{formatUtc(file.next_boundary_ts)}</span>.{" "}
        <span className="data">{count(called, "player")}</span> called to rise.
      </p>

      <div className={styles.filters}>
        <div className={styles.group} role="group" aria-label="Position filter">
          {["all", ...POSITIONS].map((position) => (
            <button
              key={position}
              type="button"
              className={styles.chip}
              aria-pressed={filters.position === position}
              onClick={() => patch({ position })}
            >
              {position === "all" ? "All" : position}
            </button>
          ))}
        </div>
        <label className={styles.field}>
          <span className={styles.fieldLabel}>Team</span>
          <select value={filters.team} onChange={(e) => patch({ team: e.target.value })}>
            <option value="all">All teams</option>
            {teams.map((team) => (
              <option key={team} value={team}>
                {team}
              </option>
            ))}
          </select>
        </label>
        <label className={styles.field}>
          <span className={styles.fieldLabel}>Search</span>
          <input
            type="search"
            value={filters.query}
            placeholder="Player name"
            onChange={(e) => patch({ query: e.target.value })}
          />
        </label>
      </div>

      <div className={styles.tableScroll}>
        <table className={styles.table}>
          <thead>
            <tr>
              <th scope="col" className={styles.left}>Player</th>
              <th scope="col" className={styles.left}>Team</th>
              <th scope="col">Pos</th>
              <th scope="col">Price</th>
              <th scope="col">Net transfers 24h</th>
              <th scope="col" className={styles.left}>Progress to the cut</th>
              <th scope="col" className={styles.left}>Call</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((player) => (
              <tr key={player.element_id} data-ineligible={!player.eligible || undefined}>
                <th scope="row" className={styles.left}>{player.name}</th>
                <td className={styles.left}>{player.team}</td>
                <td>{player.position}</td>
                <td>{(player.now_cost / 10).toFixed(1)}</td>
                <td>{player.net_transfers_24h === null ? "n/a" : player.net_transfers_24h.toLocaleString()}</td>
                <td className={styles.left}>
                  <span className={styles.progress}>
                    <span
                      className={styles.bar}
                      role="img"
                      aria-label={`${progressPercent(player.progress)} of the way to the cut`}
                    >
                      <span className={styles.fill} style={{ inlineSize: `${barFill(player.progress) * 100}%` }} />
                    </span>
                    <span className="data">{progressPercent(player.progress)}</span>
                  </span>
                </td>
                <td className={styles.left}>
                  {player.eligible ? (
                    <span className={styles.call} data-call={player.call}>{player.call}</span>
                  ) : (
                    <span className={styles.reason}>
                      not eligible: {reasonLabel(player.ineligible_reason, file.recent_move_hours)}
                    </span>
                  )}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {visible.length === 0 && <p className={styles.empty}>No players match these filters.</p>}
      {visible.length > rows.length && (
        <button type="button" className={styles.more} onClick={() => setShowAll(true)}>
          Show all {visible.length.toLocaleString()} players
        </button>
      )}
    </section>
  );
}

function History({ file }: { file: PricesFile }) {
  const cycles = file.track_record.recent_cycles;
  if (cycles.length === 0) return null;

  return (
    <section className={styles.panel} aria-labelledby="history-heading">
      <h2 id="history-heading" className={styles.panelTitle}>
        Recent cycles: calls against outcomes
      </h2>
      <p className={styles.panelSub}>
        The last {count(cycles.length, "scored cycle")}. A hit is a called player whose price
        rose overnight. Cycles marked in-sample were seen by the threshold fit, so they
        flatter the model and are not part of the hit rate above.
      </p>
      <div className={styles.tableScroll}>
        <table className={styles.table}>
          <thead>
            <tr>
              <th scope="col" className={styles.left}>Cycle</th>
              <th scope="col">Calls</th>
              <th scope="col">Hits</th>
              <th scope="col" className={styles.left}>Basis</th>
            </tr>
          </thead>
          <tbody>
            {[...cycles].reverse().map((cycle) => (
              <tr key={cycle.date} data-ineligible={!cycle.out_of_sample || undefined}>
                <th scope="row" className={styles.left}>{cycle.date}</th>
                <td>{cycle.calls}</td>
                <td>{cycle.hits}</td>
                <td className={styles.left}>{cycle.out_of_sample ? "out-of-sample" : "in-sample"}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}
