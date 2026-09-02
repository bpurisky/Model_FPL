/**
 * Paper Trade Results (§6.3-6.5, §5.16 deviation D14).
 *
 * `fpl-trends-frontend-superprompt-v2.md` §5.1.3/§5.13 keep Phase 3 and 4
 * surfaces stubbed "throughout" Phase 5. This one ships real anyway,
 * because of what it actually is: a report of what an already-frozen
 * shadow team already scored. It decides nothing — no transfer, no
 * captain, no squad — so it stays inside §5.0.2's Job 1/Job 2 boundary
 * the same way Model Board's own D1 deviation did for a different
 * Phase 3/4 boundary claim.
 *
 * That is also why this surface is **not walled off (§5.8.6)**. Model
 * Board sits on `--panel` with a left rule because it is the model
 * actively classifying players right now. This page is closer to
 * Scorecard — an existing report made legible, sitting flat on
 * `--ground` — so it copies Scorecard's layout, not Model Board's.
 *
 * **Two shas, never one number.** `file.header.model_git_sha` describes
 * the code that ran *this export* — the live model today. Each row in
 * the freeze-provenance table carries *that gameweek's own* frozen sha,
 * from when its freeze was written. They usually disagree, and this page
 * labels them separately rather than implying they are the same fact.
 *
 * **The empty state is the real state right now.** `papertrade/freezes/`
 * is empty and `data/current_season/` does not exist yet, so every panel
 * below renders its real, current answer — zero gameweeks evaluated, the
 * gate not ready — rather than a placeholder. §5.14.14 forbids shipping
 * mocked data in any state, including empty ones.
 */

import { useEffect, useState } from "react";
import { Provenance } from "../components/Provenance";
import { loadPaperTrade, type LoadProgress } from "../data/load";
import type { PaperTradeFile, PaperTradeFreezeProvenance } from "../data/schema";
import { count } from "../design/text";
import styles from "./PaperTradeResults.module.css";

type State =
  | { status: "loading"; progress: LoadProgress | null }
  | { status: "error"; error: Error }
  | { status: "ready"; file: PaperTradeFile };

/** Ordered as §6.5 lists them, not alphabetically. */
const CRITERION_ORDER = [
  "beats_fixture_adjusted_trailing_mean_mae",
  "beats_baselines_on_rank_correlation",
  "no_leakage_assertion_fired",
  "squad_reconstruction_ran_13_consecutive_gws_without_manual_correction",
  "price_change_model_reports_hit_rate_with_ci",
] as const;

const CRITERION_LABELS: Record<string, string> = {
  beats_fixture_adjusted_trailing_mean_mae: "Beats the fixture-adjusted trailing mean (MAE)",
  beats_baselines_on_rank_correlation: "Beats all three baselines on rank correlation",
  no_leakage_assertion_fired: "No leakage assertion fired",
  squad_reconstruction_ran_13_consecutive_gws_without_manual_correction:
    "13 consecutive gameweeks, no manual correction",
  price_change_model_reports_hit_rate_with_ci: "Price-change model reports its hit rate with a CI",
};

/**
 * §6.5's five criteria report five different strings, not a boolean —
 * `"PASS"`, `"FAIL"`, `"insufficient data"`, `"not tracked"`, and
 * `"not wired to live baselines yet"`. A reader needs three buckets, not
 * five, to scan the gate at a glance; the full string still renders
 * alongside every badge this collapses.
 */
export function criterionTone(status: string): "pass" | "fail" | "pending" {
  if (status === "PASS") return "pass";
  if (status === "FAIL") return "fail";
  return "pending";
}

export function PaperTradeResults() {
  const [data, setData] = useState<State>({ status: "loading", progress: null });

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const file = await loadPaperTrade((progress) =>
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

  if (data.status === "loading") return <Loading progress={data.progress} />;
  if (data.status === "error") return <Failed error={data.error} />;

  const { file } = data;

  return (
    <main className={styles.paperTrade}>
      <header className={styles.head}>
        <div>
          <h1 className={styles.title}>Paper Trade Results</h1>
          <p className={styles.sub}>
            What the frozen shadow team has actually scored against the real entry it
            shadows, gameweek by gameweek. This reports what already happened; it does not
            pick a squad, name a captain, or suggest a transfer.
          </p>
        </div>
        <Provenance header={file.header} basis={file.header.normalization_basis} />
      </header>

      <LaunchGate file={file} />
      <SquadLevel file={file} />
      <PlayerLevel file={file} />
      <FreezeProvenance file={file} />
    </main>
  );
}

// --- the launch gate, always rendered -----------------------------------

/** §6.5, reported honestly rather than forced — every string here is the
 * export's own `detail`, verbatim, because it already says exactly what
 * is or isn't measurable. Paraphrasing it would make it less accurate. */
function LaunchGate({ file }: { file: PaperTradeFile }) {
  const gate = file.launch_gate;

  return (
    <section className={styles.panel} aria-labelledby="gate-heading">
      <h2 id="gate-heading" className={styles.panelTitle}>
        §6.5 launch gate
      </h2>
      <p className={styles.panelSub}>
        Five criteria, all required, reported as they actually stand rather than forced
        toward a verdict. <span className="data">{count(gate.gameweeks_evaluated, "gameweek")}</span>{" "}
        evaluated toward the 13 the gate needs.
      </p>

      <p className={styles.readiness} data-ready={gate.ready_to_launch || undefined}>
        {gate.ready_to_launch ? "READY" : "NOT READY"}
      </p>

      <ul className={styles.criteria}>
        {CRITERION_ORDER.filter((key) => key in gate.criteria).map((key) => {
          const criterion = gate.criteria[key]!;
          const tone = criterionTone(criterion.status);
          return (
            <li key={key} className={styles.criterion} data-tone={tone}>
              <div className={styles.criterionHead}>
                <span className={styles.criterionBadge} data-tone={tone}>
                  {criterion.status}
                </span>
                <span className={styles.criterionName}>{CRITERION_LABELS[key] ?? key}</span>
              </div>
              <p className={styles.criterionDetail}>{criterion.detail}</p>
            </li>
          );
        })}
      </ul>

      {gate.gameweeks_excluded.length > 0 && (
        <p className={styles.note}>
          Excluded from the count as null observations:{" "}
          {gate.gameweeks_excluded.map((entry, index) => (
            <span key={entry.gw}>
              {index > 0 && ", "}
              gw{entry.gw} ({entry.reason})
            </span>
          ))}
          .
        </p>
      )}
    </section>
  );
}

// --- squad-level: the real entry vs the shadow team ----------------------

function SquadLevel({ file }: { file: PaperTradeFile }) {
  const squad = file.squad_level;

  return (
    <section className={styles.panel} aria-labelledby="squad-heading">
      <h2 id="squad-heading" className={styles.panelTitle}>
        Real entry vs shadow team
      </h2>
      <p className={styles.panelSub}>{squad.warning}</p>

      {squad.n_gameweeks === 0 ? (
        <p className={styles.empty}>
          The shadow team has not yet been compared against the real entry for any
          gameweek. A comparison exists once{" "}
          <code className="data">python -m papertrade freeze</code> has run at least once
          before a deadline, and a gameweek only counts once both the real entry and the
          shadow team&rsquo;s results are recorded.
        </p>
      ) : (
        <>
          <div className={styles.tableScroll}>
            <table className={styles.table}>
              <thead>
                <tr>
                  <th scope="col" className={styles.left}>
                    Gameweek
                  </th>
                  <th scope="col">Real</th>
                  <th scope="col">Shadow</th>
                  <th scope="col">Diff</th>
                </tr>
              </thead>
              <tbody>
                {squad.per_gw.map((row) => {
                  const diff = row.shadow_points - row.real_points;
                  return (
                    <tr key={row.gw}>
                      <th scope="row" className={styles.left}>
                        gw{row.gw}
                      </th>
                      <td className="data">{row.real_points}</td>
                      <td className="data">{row.shadow_points}</td>
                      <td className="data">
                        <span className={diff >= 0 ? styles.better : styles.worse}>
                          {diff >= 0 ? "+" : "−"}
                          {Math.abs(diff)}
                        </span>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
              <tfoot>
                <tr>
                  <th scope="row" className={styles.left}>
                    Cumulative
                  </th>
                  <td className="data">{squad.cumulative_real_points}</td>
                  <td className="data">{squad.cumulative_shadow_points}</td>
                  <td className="data">
                    <span className={squad.shadow_minus_real >= 0 ? styles.better : styles.worse}>
                      {squad.shadow_minus_real >= 0 ? "+" : "−"}
                      {Math.abs(squad.shadow_minus_real)}
                    </span>
                  </td>
                </tr>
              </tfoot>
            </table>
          </div>

          {squad.excluded_gameweeks.length > 0 && (
            <p className={styles.note}>
              Excluded: {squad.excluded_gameweeks.map((entry, index) => (
                <span key={entry.gw}>
                  {index > 0 && ", "}
                  gw{entry.gw} ({entry.reason})
                </span>
              ))}
              .
            </p>
          )}
        </>
      )}
    </section>
  );
}

// --- player-level accuracy ------------------------------------------------

function PlayerLevel({ file }: { file: PaperTradeFile }) {
  return (
    <section className={styles.panel} aria-labelledby="player-heading">
      <h2 id="player-heading" className={styles.panelTitle}>
        Player-level accuracy, live
      </h2>
      <p className={styles.panelSub}>
        The frozen projection against the real result, per evaluated gameweek. Unlike the
        squad-level comparison above, this pools every player each week rather than one
        team&rsquo;s fifteen, so it carries real statistical weight far sooner.
      </p>

      {file.player_level.length === 0 ? (
        <p className={styles.empty}>
          No gameweek has both a recorded freeze and recorded actuals yet.
          {file.player_level_skipped.length > 0 && (
            <>
              {" "}
              {count(file.player_level_skipped.length, "gameweek")} skipped so far:{" "}
              {file.player_level_skipped.map((entry, index) => (
                <span key={entry.gw}>
                  {index > 0 && ", "}
                  gw{entry.gw} ({entry.reason})
                </span>
              ))}
              .
            </>
          )}
        </p>
      ) : (
        <div className={styles.tableScroll}>
          <table className={styles.table}>
            <thead>
              <tr>
                <th scope="col" className={styles.left}>
                  Gameweek
                </th>
                <th scope="col">n</th>
                <th scope="col">MAE</th>
                <th scope="col">Spearman</th>
              </tr>
            </thead>
            <tbody>
              {file.player_level.map((row) => (
                <tr key={row.gw}>
                  <th scope="row" className={styles.left}>
                    gw{row.gw}
                  </th>
                  <td className={`data ${styles.muted}`}>{row.n.toLocaleString()}</td>
                  <td className="data">{row.mae === null ? "—" : row.mae.toFixed(4)}</td>
                  <td className="data">
                    {row.spearman_mean === null ? "—" : row.spearman_mean.toFixed(4)}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {file.player_level_excluded.length > 0 && (
        <p className={styles.note}>
          Excluded as null observations:{" "}
          {file.player_level_excluded.map((entry, index) => (
            <span key={entry.gw}>
              {index > 0 && ", "}
              gw{entry.gw} ({entry.reason})
            </span>
          ))}
          .
        </p>
      )}
    </section>
  );
}

// --- freeze provenance -----------------------------------------------------

/** That gameweek's own sha, deliberately labelled apart from the
 * page-header Provenance bar's sha (§6.5 criteria 3-4). */
function FreezeProvenance({ file }: { file: PaperTradeFile }) {
  const rows = file.launch_gate.freeze_provenance;

  return (
    <section className={styles.panel} aria-labelledby="provenance-heading">
      <h2 id="provenance-heading" className={styles.panelTitle}>
        What each freeze recorded about itself
      </h2>
      <p className={styles.panelSub}>
        Not the export&rsquo;s own sha above — each row is that gameweek&rsquo;s own frozen
        sha, from when its freeze was written before its deadline.
      </p>

      {rows.length === 0 ? (
        <p className={styles.empty}>No freezes recorded yet.</p>
      ) : (
        <div className={styles.tableScroll}>
          <table className={styles.table}>
            <thead>
              <tr>
                <th scope="col" className={styles.left}>
                  Gameweek
                </th>
                <th scope="col" className={styles.left}>
                  Sha at freeze time
                </th>
                <th scope="col">Leakage check</th>
                <th scope="col">Manual correction</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((row) => (
                <ProvenanceRow key={row.gw} row={row} />
              ))}
            </tbody>
          </table>
        </div>
      )}
    </section>
  );
}

function ProvenanceRow({ row }: { row: PaperTradeFreezeProvenance }) {
  return (
    <tr>
      <th scope="row" className={styles.left}>
        gw{row.gw}
      </th>
      <td className={`data ${styles.left}`}>
        {row.model_git_sha ? row.model_git_sha.slice(0, 10) : "unknown"}
      </td>
      <td className="data">
        {row.leakage_check === null ? (
          <span className={styles.muted}>not tracked</span>
        ) : row.leakage_verified ? (
          "verified"
        ) : (
          "failed"
        )}
      </td>
      <td className="data">
        {row.records_manual_correction_field ? (
          (row.manual_correction ?? "none")
        ) : (
          <span className={styles.muted}>not tracked</span>
        )}
      </td>
    </tr>
  );
}

// --- states ----------------------------------------------------------------

function Loading({ progress }: { progress: LoadProgress | null }) {
  const kb = (bytes: number) => `${Math.round(bytes / 1024)} KB`;
  const pct =
    progress?.total != null ? Math.round((progress.received / progress.total) * 100) : null;
  return (
    <main className={styles.paperTrade}>
      <h1 className={styles.title}>Paper Trade Results</h1>
      <div className={styles.loading}>
        <div
          className={styles.track}
          role="progressbar"
          aria-valuenow={pct ?? undefined}
          aria-valuemin={0}
          aria-valuemax={100}
          aria-label="Loading paper trade results"
        >
          <div className={styles.fill} style={{ inlineSize: pct === null ? "30%" : `${pct}%` }} />
        </div>
        <p className={`${styles.sub} data`}>
          {progress
            ? `papertrade.json — ${kb(progress.received)}${progress.total ? ` of ${kb(progress.total)}` : ""}`
            : "requesting papertrade.json"}
        </p>
      </div>
    </main>
  );
}

function Failed({ error }: { error: Error }) {
  return (
    <main className={styles.paperTrade}>
      <h1 className={styles.title}>Paper Trade Results</h1>
      <div className={styles.failure} role="alert">
        <p className={styles.failureHead}>papertrade.json did not load.</p>
        <p className={`${styles.sub} data`}>{error.message}</p>
        <p className={styles.sub}>
          Run <code className="data">python -m web.export papertrade</code> and reload.
        </p>
      </div>
    </main>
  );
}
