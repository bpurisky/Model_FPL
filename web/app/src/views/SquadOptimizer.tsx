/**
 * Squad Optimizer (§5.4, Phase 3's `squad/optimize.py`) — the one surface
 * in this app that calls a live backend instead of reading a committed
 * export from `data/v1/`.
 *
 * This was `out_of_phase` through 2026-08-25: the 2026-08-25
 * fpl-trends-superprompt.md entry records in detail why an ILP solve
 * cannot run in the browser under this app's own architecture rules
 * (§5.1.1 forbids an API server and runtime Python in the browser; §5.6
 * forbids client-side inference outright, and a live solve *is*
 * inference) and says the surface stays stubbed "until an operator
 * explicitly chooses to add a backend or a browser-side solver". That
 * choice was made 2026-09-02: `service/app.py` is a thin HTTP wrapper
 * around the exact CLI path `squad/__main__.py recommend` already
 * exercised live (see its module docstring), and this view calls it.
 *
 * §7.3 requires the static analytics to render with the Cloudflare Worker
 * offline; the same argument applies to this new dependency. A build with
 * no backend configured (`VITE_OPTIMIZER_API_URL` unset) shows an honest
 * explanation rather than a broken form or a raw network error.
 *
 * Team rating and chip comparison (squad/rating.py) are projections of the
 * same xP the solver uses, not a chip planner: the chip table says what each
 * chip would add in each horizon gameweek, and whether the entry's history
 * shows it already used this half-season; it does not schedule chips.
 */

import { useEffect, useRef, useState, type FormEvent } from "react";
import { useApp } from "../app/state";
import {
  fetchRecommendation,
  OPTIMIZER_API_URL,
  OptimizerError,
  parseEntryId,
  TIMEOUT_STATUS,
  warmUp,
  type OptimizerChipGain,
  type OptimizerRecommendation,
  type OptimizerXiPlayer,
} from "../data/optimizer";
import styles from "./SquadOptimizer.module.css";

type ResultState =
  | { status: "idle" }
  | { status: "loading" }
  | { status: "error"; error: Error }
  | { status: "ready"; data: OptimizerRecommendation };

interface RunArgs {
  entryId: number;
  horizon?: number[] | undefined;
  maxTransfers?: number | undefined;
}

/** Where the backend is, as far as the warm-up ping knows. */
type Backend = "unknown" | "awake" | "unreachable";

const POSITIONS = ["GK", "DEF", "MID", "FWD"] as const;

function price(tenths: number): string {
  return `£${(tenths / 10).toFixed(1)}m`;
}

function parseHorizon(raw: string): number[] | undefined {
  const trimmed = raw.trim();
  if (!trimmed) return undefined;
  const values = trimmed
    .split(",")
    .map((part) => Number(part.trim()))
    .filter((n) => Number.isFinite(n) && n > 0);
  return values.length > 0 ? values : undefined;
}

export function SquadOptimizer() {
  const { state, dispatch } = useApp();
  const [entryInput, setEntryInput] = useState(state.entry ? String(state.entry) : "");
  const [horizonInput, setHorizonInput] = useState("");
  const [maxTransfersInput, setMaxTransfersInput] = useState("");
  const [result, setResult] = useState<ResultState>({ status: "idle" });
  const [backend, setBackend] = useState<Backend>("unknown");
  const lastRun = useRef<RunArgs | null>(null);

  const configured = Boolean(OPTIMIZER_API_URL);

  async function run(args: RunArgs) {
    lastRun.current = args;
    setResult({ status: "loading" });
    try {
      const data = await fetchRecommendation(args.entryId, { horizon: args.horizon, maxTransfers: args.maxTransfers });
      setBackend("awake");
      setResult({ status: "ready", data });
    } catch (error) {
      setResult({ status: "error", error: error as Error });
    }
  }

  // The free-tier backend sleeps after ~15 idle minutes and takes about a
  // minute to wake. Pinging it the moment this view opens overlaps that
  // wake-up with the reader typing their team ID, and tells the loading
  // note whether a run is waiting on the server or on the solve.
  //
  // A linked run (`?view=optimizer&entry=123`, §5.5) fires once on mount
  // rather than on every render — horizon/max-transfers aren't in the URL
  // (see data/optimizer.ts's contract note), so a link reproduces the
  // team ID and lets the reader re-run with whatever controls they want.
  useEffect(() => {
    if (!configured) return;
    let cancelled = false;
    void warmUp().then((awake) => {
      if (!cancelled) setBackend(awake ? "awake" : "unreachable");
    });
    if (state.entry) void run({ entryId: state.entry });
    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  function onSubmit(event: FormEvent) {
    event.preventDefault();
    const entryId = parseEntryId(entryInput);
    if (entryId === undefined) {
      setResult({
        status: "error",
        error: new OptimizerError("Enter your FPL team ID: the number, or a link to your Points page."),
      });
      return;
    }
    setEntryInput(String(entryId));
    dispatch({ type: "entry", entryId });
    const horizon = parseHorizon(horizonInput);
    const maxTransfers = maxTransfersInput.trim() === "" ? undefined : Number(maxTransfersInput);
    void run({ entryId, horizon, maxTransfers });
  }

  function retry() {
    if (lastRun.current) void run(lastRun.current);
  }

  return (
    <main className={styles.optimizer}>
      <header className={styles.head}>
        <div>
          <h1 className={styles.title}>Squad Optimizer</h1>
          <p className={styles.sub}>
            Given a team ID, solves for the highest-value legal transfer under budget,
            club-limit and formation constraints — an integer linear program
            (<code className="data">squad/optimize.py</code>), not a sort. The solve runs
            server-side, against your entry&rsquo;s real squad, right now.
          </p>
        </div>
      </header>

      {!configured ? (
        <NotConfigured />
      ) : (
        <>
          <form className={styles.form} onSubmit={onSubmit}>
            <label className={styles.field}>
              <span className={styles.fieldLabel}>FPL team ID</span>
              <input
                className={styles.input}
                type="text"
                inputMode="numeric"
                autoComplete="off"
                value={entryInput}
                onChange={(event) => setEntryInput(event.target.value)}
                placeholder="e.g. 2986528, or paste your Points page link"
                required
              />
            </label>
            <label className={styles.field}>
              <span className={styles.fieldLabel}>Horizon, gameweeks (optional)</span>
              <input
                className={styles.input}
                type="text"
                value={horizonInput}
                onChange={(event) => setHorizonInput(event.target.value)}
                placeholder="defaults to the next 3"
              />
            </label>
            <label className={styles.field}>
              <span className={styles.fieldLabel}>Max transfers (optional)</span>
              <input
                className={styles.input}
                type="number"
                min={0}
                value={maxTransfersInput}
                onChange={(event) => setMaxTransfersInput(event.target.value)}
                placeholder="unlimited"
              />
            </label>
            <button type="submit" className={styles.submit} disabled={result.status === "loading"}>
              {result.status === "loading" ? "Solving…" : "Get recommendation"}
            </button>
          </form>

          <TeamIdHelp />

          {result.status === "loading" && <LoadingNote backend={backend} />}
          {result.status === "error" && (
            <ErrorNote error={result.error} onRetry={lastRun.current ? retry : undefined} />
          )}
          {result.status === "ready" && <Recommendation data={result.data} />}
        </>
      )}
    </main>
  );
}

function NotConfigured() {
  return (
    <p className={styles.empty}>
      No optimizer backend is configured for this build (
      <code className="data">VITE_OPTIMIZER_API_URL</code> is unset). §7.3&rsquo;s rule for the
      Cloudflare Worker applies here too: the rest of the app renders with a live dependency
      offline, and this surface degrades to an explanation rather than a broken form. See{" "}
      <code className="data">service/app.py</code> and the repo README for how to run or deploy
      the backend.
    </p>
  );
}

const TEAM_ID_EXAMPLE = "fantasy.premierleague.com/entry/2986528/event/7";

function TeamIdHelp() {
  return (
    <details className={styles.help}>
      <summary className={styles.helpSummary}>How do I find my team ID?</summary>
      <ol className={styles.helpSteps}>
        <li>
          Sign in at <code className="data">fantasy.premierleague.com</code> in a web browser.
        </li>
        <li>
          Open the <strong>Points</strong> tab.
        </li>
        <li>
          Look at the address bar: your team ID is the number after <code className="data">/entry/</code>.
          In <code className="data">{TEAM_ID_EXAMPLE}</code> it&rsquo;s <code className="data">2986528</code>.
        </li>
      </ol>
      <p className={styles.sub}>
        You can also paste the whole link into the box above and the ID is read out of it. The FPL
        phone app doesn&rsquo;t show the ID, so use a browser (a phone browser works). Your ID stays
        the same all season.
      </p>
    </details>
  );
}

function LoadingNote({ backend }: { backend: Backend }) {
  if (backend !== "awake") {
    return (
      <p className={styles.loadingNote} role="status">
        Waking the optimizer server. It runs on free hosting that sleeps when nobody has used it
        for a while, so the first request can take up to about a minute. Requests after that are
        much faster.
      </p>
    );
  }
  return (
    <p className={styles.loadingNote} role="status">
      Fetching your squad and transfer history from the FPL API and running the solver. This
      usually takes a few seconds, longer when the FPL API is busy (around a deadline, for example).
    </p>
  );
}

function ErrorNote({ error, onRetry }: { error: Error; onRetry?: (() => void) | undefined }) {
  const status = error instanceof OptimizerError ? error.status : undefined;
  const hint =
    status === 404
      ? `Check the team ID: it's the number after /entry/ in your Points page link, e.g. ${TEAM_ID_EXAMPLE}.`
      : status === TIMEOUT_STATUS
      ? "Trying again usually works: the server is normally awake by now."
      : status === 422
      ? "This usually means the entry has no gameweek history yet (before gw1's deadline)."
      : status === 429
      ? "The backend rate-limits requests per IP address. Wait a minute and try again."
      : status === 502
      ? "The FPL API itself was unreachable or erroring. Usually transient — try again shortly."
      : undefined;
  return (
    <div className={styles.failure} role="alert">
      <p className={styles.failureHead}>{error.message}</p>
      {hint && <p className={styles.sub}>{hint}</p>}
      {onRetry && status !== 404 && status !== 422 && (
        <button type="button" className={styles.retry} onClick={onRetry}>
          Try again
        </button>
      )}
    </div>
  );
}

function Recommendation({ data }: { data: OptimizerRecommendation }) {
  const firstGw = data.horizon[0];
  const xi = firstGw !== undefined ? (data.starting_xi[String(firstGw)] ?? []) : [];

  return (
    <div className={styles.results}>
      <p className={styles.caveat}>{data.caveat}</p>

      <TeamRating data={data} />
      <ChipTable data={data} />

      <section className={styles.panel}>
        <h2 className={styles.panelTitle}>Summary</h2>
        <dl className={styles.summary}>
          <div>
            <dt>Free transfers</dt>
            <dd className="data">{data.free_transfers}</dd>
          </div>
          <div>
            <dt>Bank</dt>
            <dd className="data">{price(data.bank)}</dd>
          </div>
          <div>
            <dt>Horizon</dt>
            <dd className="data">gw {data.horizon.join(", ")}</dd>
          </div>
          <div>
            <dt>Hits taken</dt>
            <dd className="data">
              {data.hits_taken} (&minus;{data.hits_taken * 4} pts)
            </dd>
          </div>
          <div>
            <dt>Bank after</dt>
            <dd className="data">{price(data.bank_after)}</dd>
          </div>
        </dl>
      </section>

      <section className={styles.panel}>
        <h2 className={styles.panelTitle}>Recommended transfers</h2>
        {data.transfers.length === 0 ? (
          <p className={styles.empty}>No transfer recommended — hold the squad.</p>
        ) : (
          <ul className={styles.transfers}>
            {data.transfers.map((transfer) => (
              <li key={`${transfer.out.element_id}-${transfer.in.element_id}`} className={styles.transferRow}>
                <span className={styles.transferOut}>
                  OUT: {transfer.out.name} ({transfer.out.position}, {transfer.out.club})
                </span>
                <span className={styles.transferIn}>
                  IN: {transfer.in.name} ({transfer.in.position}, {transfer.in.club},{" "}
                  {price(transfer.in.now_cost)})
                </span>
              </li>
            ))}
          </ul>
        )}
      </section>

      {data.template_risk.length > 0 && (
        <section className={styles.panel}>
          <h2 className={styles.panelTitle}>Template risk</h2>
          <ul className={styles.riskList}>
            {data.template_risk.map((risk) => (
              <li key={risk.element_id} className={styles.riskItem}>
                <strong>{risk.name}</strong>: {risk.message}
              </li>
            ))}
          </ul>
        </section>
      )}

      <section className={styles.panel}>
        <h2 className={styles.panelTitle}>
          Starting XI{firstGw !== undefined ? `, gameweek ${firstGw}` : ""}
        </h2>
        <XiGrid xi={xi} />
      </section>

      <section className={styles.panel}>
        <h2 className={styles.panelTitle}>Bench order (substitute keeper, then outfield)</h2>
        {data.bench_order.length === 0 ? (
          <p className={styles.empty}>No bench players in this recommendation.</p>
        ) : (
          <ol className={styles.bench}>
            {data.bench_order.map((player) => (
              <li key={player.element_id}>
                {player.name} ({player.position})
              </li>
            ))}
          </ol>
        )}
      </section>

      {data.plan && data.plan.length > 0 && (
        <section className={styles.panel}>
          <h2 className={styles.panelTitle}>The plan behind it</h2>
          <p className={styles.sub}>
            The later weeks this recommendation was planned around. Only this week&rsquo;s
            transfers are advice; the rest is re-solved next week with new information.
          </p>
          <ul className={styles.transfers}>
            {data.plan.map((week) => (
              <li key={week.gw} className={styles.transferRow}>
                <strong>
                  Gameweek {week.gw} ({week.free_transfers} free
                  {week.hits > 0 ? `, ${week.hits} hit${week.hits > 1 ? "s" : ""}` : ""})
                </strong>
                {week.transfers.length === 0 ? (
                  <span>Roll the transfer.</span>
                ) : (
                  week.transfers.map((transfer) => (
                    <span key={`${transfer.out.element_id}-${transfer.in.element_id}`}>
                      {transfer.out.name} &rarr; {transfer.in.name} ({price(transfer.in.now_cost)})
                    </span>
                  ))
                )}
              </li>
            ))}
          </ul>
        </section>
      )}

      <p className={styles.note}>
        Squad size {data.squad_size}; {data.unchanged_from_current} of the currently-owned
        players kept unsold.
      </p>
    </div>
  );
}

const CHIP_LABELS: Record<OptimizerChipGain["chip"], string> = {
  bboost: "Bench Boost",
  "3xc": "Triple Captain",
  freehit: "Free Hit",
  wildcard: "Wildcard",
};
const CHIP_ORDER = ["bboost", "3xc", "freehit", "wildcard"] as const;

function TeamRating({ data }: { data: OptimizerRecommendation }) {
  if (data.team_rating === undefined || data.current_xi_xp === undefined) return null;
  return (
    <section className={styles.panel}>
      <h2 className={styles.panelTitle}>Team rating</h2>
      <div className={styles.rating}>
        <span className={`${styles.ratingValue} data`}>{data.team_rating.toFixed(0)}</span>
        <span className={styles.ratingScale}>/ 100</span>
      </div>
      <dl className={styles.summary}>
        <div>
          <dt>Your squad, no transfers</dt>
          <dd className="data">{data.current_xi_xp.toFixed(1)} xP</dd>
        </div>
        {data.recommended_xp !== undefined && (
          <div>
            <dt>Recommended plan, net of hits</dt>
            <dd className="data">{data.recommended_xp.toFixed(1)} xP</dd>
          </div>
        )}
        {data.optimum_xp !== undefined && (
          <div>
            <dt>Best squad in budget</dt>
            <dd className="data">{data.optimum_xp.toFixed(1)} xP</dd>
          </div>
        )}
      </dl>
      <p className={styles.sub}>
        Projected points over gameweek {data.horizon.join(", ")}, best XI and captain each week
        (captain counted double). The rating is your squad&rsquo;s projection as a share of the
        best squad your budget could hold, ignoring transfer limits and hits. These are model
        projections, not guarantees.
      </p>
    </section>
  );
}

function ChipTable({ data }: { data: OptimizerRecommendation }) {
  const chips = data.chips;
  if (!chips || chips.length === 0) return null;
  return (
    <section className={styles.panel}>
      <h2 className={styles.panelTitle}>Chip comparison</h2>
      <p className={styles.sub}>
        Projected extra points if the chip is played in that gameweek, against the recommended
        plan. The best gameweek for each chip is highlighted. Wildcard is judged at the first
        gameweek only. Estimates, not guarantees.
      </p>
      <div className={styles.tableWrap}>
        <table className={styles.chipTable}>
          <thead>
            <tr>
              <th scope="col">Chip</th>
              {data.horizon.map((gw) => (
                <th key={gw} scope="col" className="data">
                  GW{gw}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {CHIP_ORDER.map((chip) => {
              const rows = chips.filter((row) => row.chip === chip);
              if (rows.length === 0) return null;
              const best = data.chip_best_gw?.[chip];
              const used = rows.every((row) => row.available === false);
              return (
                <tr key={chip} className={used ? styles.chipUsed : undefined}>
                  <th scope="row">
                    {CHIP_LABELS[chip]}
                    {used && <span className={styles.chipNote}> used</span>}
                  </th>
                  {data.horizon.map((gw) => {
                    const cell = rows.find((row) => row.gw === gw);
                    return (
                      <td key={gw} className={`data ${cell && gw === best ? styles.chipBest : ""}`}>
                        {cell ? `${cell.gain >= 0 ? "+" : "−"}${Math.abs(cell.gain).toFixed(1)}` : "–"}
                      </td>
                    );
                  })}
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </section>
  );
}

function XiGrid({ xi }: { xi: OptimizerXiPlayer[] }) {
  return (
    <div className={styles.xi}>
      {POSITIONS.map((position) => {
        const players = xi.filter((player) => player.position === position);
        if (players.length === 0) return null;
        return (
          <div key={position} className={styles.xiRow}>
            <span className={styles.xiPos}>{position}</span>
            <span className={styles.xiPlayers}>
              {players.map((player) => (
                <span key={player.element_id} className={styles.xiPlayer}>
                  {player.name}
                  {player.captain && <span className={styles.captainBadge}>C</span>}
                  {player.vice_captain && <span className={styles.captainBadge}>V</span>}
                </span>
              ))}
            </span>
          </div>
        );
      })}
    </div>
  );
}
