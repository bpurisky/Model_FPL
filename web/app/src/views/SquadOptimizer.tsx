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
 * What this does *not* do: name a chip. §7.2's "My team" view also wants
 * a chip planner respecting the gameweek 19 wildcard/free-hit expiry, but
 * `squad/optimize.py` itself has no chip-strategy logic to call — nothing
 * here would be reading a real computation, and §5.14.14 forbids shipping
 * placeholder data in any state. Left as a documented gap, not invented.
 */

import { useEffect, useState, type FormEvent } from "react";
import { useApp } from "../app/state";
import {
  fetchRecommendation,
  OPTIMIZER_API_URL,
  OptimizerError,
  type OptimizerRecommendation,
  type OptimizerXiPlayer,
} from "../data/optimizer";
import styles from "./SquadOptimizer.module.css";

type ResultState =
  | { status: "idle" }
  | { status: "loading" }
  | { status: "error"; error: Error }
  | { status: "ready"; data: OptimizerRecommendation };

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

  const configured = Boolean(OPTIMIZER_API_URL);

  async function run(entryId: number, horizon?: number[], maxTransfers?: number) {
    setResult({ status: "loading" });
    try {
      const data = await fetchRecommendation(entryId, { horizon, maxTransfers });
      setResult({ status: "ready", data });
    } catch (error) {
      setResult({ status: "error", error: error as Error });
    }
  }

  // A linked run (`?view=optimizer&entry=123`, §5.5) fires once on mount
  // rather than on every render — horizon/max-transfers aren't in the URL
  // (see data/optimizer.ts's contract note), so a link reproduces the
  // team ID and lets the reader re-run with whatever controls they want.
  useEffect(() => {
    if (configured && state.entry) void run(state.entry);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  function onSubmit(event: FormEvent) {
    event.preventDefault();
    const entryId = Number(entryInput);
    if (!Number.isFinite(entryId) || entryId <= 0) {
      setResult({ status: "error", error: new OptimizerError("Enter a valid FPL team ID (a positive number).") });
      return;
    }
    dispatch({ type: "entry", entryId });
    const horizon = parseHorizon(horizonInput);
    const maxTransfers = maxTransfersInput.trim() === "" ? undefined : Number(maxTransfersInput);
    void run(entryId, horizon, maxTransfers);
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
                type="number"
                inputMode="numeric"
                min={1}
                value={entryInput}
                onChange={(event) => setEntryInput(event.target.value)}
                placeholder="e.g. 2986528"
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

          {result.status === "loading" && <LoadingNote />}
          {result.status === "error" && <ErrorNote error={result.error} />}
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

function LoadingNote() {
  return (
    <p className={styles.loadingNote}>
      Requesting a live solve. This fetches your squad and transfer history from the FPL API,
      builds this season&rsquo;s trailing rates, and runs the ILP solver — typically 5&ndash;30
      seconds depending on how many retries the FPL API needs. No progress bar here: unlike a
      committed export, there is no byte count to report until the solve finishes.
    </p>
  );
}

function ErrorNote({ error }: { error: Error }) {
  const status = error instanceof OptimizerError ? error.status : undefined;
  const hint =
    status === 404
      ? "Check the team ID — it's the number in your FPL ‘Points’ page URL, e.g. .../entry/2986528/event/1."
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
    </div>
  );
}

function Recommendation({ data }: { data: OptimizerRecommendation }) {
  const firstGw = data.horizon[0];
  const xi = firstGw !== undefined ? (data.starting_xi[String(firstGw)] ?? []) : [];

  return (
    <div className={styles.results}>
      <p className={styles.caveat}>{data.caveat}</p>

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
        <h2 className={styles.panelTitle}>Bench order (best-first)</h2>
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

      <p className={styles.note}>
        Squad size {data.squad_size}; {data.unchanged_from_current} of the currently-owned
        players kept unsold.
      </p>
    </div>
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
                </span>
              ))}
            </span>
          </div>
        );
      })}
    </div>
  );
}
