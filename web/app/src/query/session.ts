/**
 * The DuckDB-WASM session (§5.2 D4, §5.1.2).
 *
 * > "It is a query engine over static files shipped to the browser, not a
 * > service. No SQL written by the app may compute an inferential
 * > statistic (§5.6)."
 *
 * What the engine is here for is the work an engine is actually good at:
 * decoding parquet, pushing filter predicates down into it, and grouping
 * 85,000 player-gameweeks. What it is deliberately *not* here for is the
 * reduction itself — that happens in `reduce.ts`, in plain TypeScript,
 * so a single implementation can be held against Python by a golden test
 * that needs no engine to run (§5.11.2). Writing `avg(x)` in SQL would
 * have been shorter and would have left the numbers on screen covered by
 * no test that runs in CI.
 *
 * **Lazy by construction.** §5.9 budgets the engine at 1.2 MB and says
 * it must appear in no initial-load chunk. Everything here is behind a
 * dynamic `import()`, so the bundler splits it out and the Correlation
 * Lab — the landing surface — never pays for it. Nothing in this module
 * may be imported statically by a view; call `openSession()`.
 *
 * **Self-hosted, not CDN.** The bundles are resolved through Vite's
 * `?url` imports and served from our own origin. §5.1.1's "zero
 * operational surface" argument is weakened by a runtime dependency on
 * jsdelivr being up, and a static site that stops working when someone
 * else's CDN has a bad day is not static.
 */

import type * as duckdb from "@duckdb/duckdb-wasm";
import type { LoadProgress } from "../data/load";

/** Thrown when `panel.parquet` is not on disk — §5.14.8's empty state. */
export class PanelMissingError extends Error {
  constructor() {
    super(
      "panel.parquet is not published. It is a build artifact, not committed data " +
        "(§5.3.4) — run `python -m web.export panel` to create it.",
    );
    this.name = "PanelMissingError";
  }
}

export interface Session {
  /**
   * Run a named query. Views never call this directly; they call the
   * helpers in `panel.ts`, which is the only module that writes SQL.
   */
  run(sql: string): Promise<Record<string, unknown>[]>;
  close(): Promise<void>;
}

const PANEL_URL = "/data/v1/panel.parquet";
const PANEL_TABLE = "panel";

let pending: Promise<Session> | null = null;

/**
 * The engine, the panel, and a registered view over it — created once
 * and shared. A second Graph Builder mount must not pay for a second
 * 3 MB download or a second 1.2 MB engine.
 */
export function openSession(onProgress?: (progress: LoadProgress) => void): Promise<Session> {
  pending ??= create(onProgress).catch((error) => {
    // A failed open must not poison every later attempt: the usual cause
    // is a missing build artifact, and the usual fix is to build it and
    // reload the route rather than the page.
    pending = null;
    throw error;
  });
  return pending;
}

async function create(onProgress?: (progress: LoadProgress) => void): Promise<Session> {
  // The parquet is fetched before the engine so a missing artifact costs
  // nothing: §5.14.8's empty state is the common case on a fresh clone,
  // and downloading 1.2 MB of engine to discover there is no data would
  // be the wrong order to learn it in.
  const panel = await fetchPanel(onProgress);

  const [duck, bundle] = await Promise.all([
    import("@duckdb/duckdb-wasm"),
    resolveBundle(),
  ]);

  const worker = new Worker(bundle.mainWorker!, { type: "module" });
  const logger = new duck.VoidLogger();
  const db = new duck.AsyncDuckDB(logger, worker);
  await db.instantiate(bundle.mainModule, bundle.pthreadWorker ?? null);

  await db.registerFileBuffer("panel.parquet", panel);
  const connection = await db.connect();
  // A view rather than a table: the parquet stays the source of record
  // and DuckDB reads only the columns a query projects.
  await connection.query(
    `CREATE OR REPLACE VIEW ${PANEL_TABLE} AS SELECT * FROM read_parquet('panel.parquet')`,
  );

  return {
    async run(sql: string) {
      const table = await connection.query(sql);
      return table.toArray().map((row) => row.toJSON() as Record<string, unknown>);
    },
    async close() {
      await connection.close();
      await db.terminate();
      worker.terminate();
      pending = null;
    },
  };
}

/**
 * Vite resolves these to URLs on our own origin at build time. The `eh`
 * bundle (exception handling) is preferred and `mvp` is the fallback for
 * browsers without it — `selectBundle` does the feature detection.
 */
async function resolveBundle(): Promise<duckdb.DuckDBBundle> {
  const duck = await import("@duckdb/duckdb-wasm");
  const [mvpModule, mvpWorker, ehModule, ehWorker] = await Promise.all([
    import("@duckdb/duckdb-wasm/dist/duckdb-mvp.wasm?url"),
    import("@duckdb/duckdb-wasm/dist/duckdb-browser-mvp.worker.js?url"),
    import("@duckdb/duckdb-wasm/dist/duckdb-eh.wasm?url"),
    import("@duckdb/duckdb-wasm/dist/duckdb-browser-eh.worker.js?url"),
  ]);

  return duck.selectBundle({
    mvp: { mainModule: mvpModule.default, mainWorker: mvpWorker.default },
    eh: { mainModule: ehModule.default, mainWorker: ehWorker.default },
  });
}

/**
 * The panel, with real byte counts.
 *
 * Fetched whole and handed to `registerFileBuffer` rather than pointed
 * at over HTTP. DuckDB can range-request a remote parquet and read only
 * the pages it needs, which sounds better — but Vite's dev middleware
 * answers no range requests, so the engine would silently fall back to
 * refetching the whole file per query. Downloading it once, visibly, is
 * both faster and honest about what the user is waiting for, which is
 * what §5.9 asks for: "the user should know whether they are waiting on
 * 200 KB or 8 MB".
 */
async function fetchPanel(onProgress?: (progress: LoadProgress) => void): Promise<Uint8Array> {
  const response = await fetch(PANEL_URL);
  if (response.status === 404) throw new PanelMissingError();
  if (!response.ok) {
    throw new Error(`${PANEL_URL}: ${response.status} ${response.statusText}`);
  }

  const header = response.headers.get("content-length");
  const total = header ? Number(header) : null;

  if (!response.body || !onProgress) {
    return new Uint8Array(await response.arrayBuffer());
  }

  const reader = response.body.getReader();
  const chunks: Uint8Array[] = [];
  let received = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    received += value.length;
    onProgress({ received, total });
  }

  const merged = new Uint8Array(received);
  let offset = 0;
  for (const chunk of chunks) {
    merged.set(chunk, offset);
    offset += chunk.length;
  }

  // The dev middleware serves any file in the export directory, so a
  // missing parquet arrives as a 404 above — but a *stale* one that is
  // not parquet at all would reach DuckDB as a confusing internal error.
  // PAR1 is the format's magic number at both ends of the file.
  if (merged.length < 8 || String.fromCharCode(...merged.slice(0, 4)) !== "PAR1") {
    throw new Error(`${PANEL_URL} is not a parquet file`);
  }

  return merged;
}
