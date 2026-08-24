/**
 * The panel session — a column store over `panel.parquet` (§5.1.2).
 *
 * **This replaces DuckDB-WASM, and the reason is §5.9.** D4 locked the
 * engine on the argument that "group-by across the panel with multiple
 * filter predicates is SQL-shaped, and DuckDB's parquet reader removes a
 * separate decode step". Both halves of that are true. The budget was
 * still unmeetable: `duckdb-eh.wasm` is 34.8 MB raw, 6.9 MB gzipped and
 * 4.6 MB brotli, against §5.9's 1.2 MB for the lazy engine chunk, and no
 * compression closes a 4x gap.
 *
 * What made the swap cheap is that the engine had already stopped doing
 * the interesting part. §5.6.2 only permits the browser to reduce because
 * its seven operations have one implementation held against Python by a
 * golden test — so the reductions live in `reduce.ts`, and SQL was left
 * doing `WHERE`, `GROUP BY` and `list(...)`. That is the ~80 lines in
 * `panel.ts`, not a database.
 *
 * Recorded as §5.16 deviation D10 against §5.2's locked stack.
 *
 * What this module owns is decode and cache. `hyparquet` reads column
 * chunks out of the file; a column is decoded on first use and kept, so
 * changing one drop zone re-reads one column rather than the file. The
 * panel is 83 columns wide and a chart never touches more than about six.
 */

import { DATA_BASE, type LoadProgress } from "../data/load";

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

/** One decoded column. Nulls are preserved and never coerced (§5.3.3). */
export type PanelColumn = readonly (number | string | boolean | null)[];

export interface Session {
  /** Rows in the panel. */
  readonly rows: number;
  /** Column names the file actually carries. */
  readonly names: readonly string[];
  /** A decoded column, from cache after the first call. */
  column(key: string): Promise<PanelColumn>;
  close(): Promise<void>;
}

const PANEL_URL = `${DATA_BASE}/panel.parquet`;

let pending: Promise<Session> | null = null;

/**
 * The panel, opened once and shared. A second Graph Builder mount must
 * not pay for a second download or a second decode.
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
  const bytes = await fetchPanel(onProgress);

  // Dynamic, so the reader lands in the route chunk rather than the
  // landing bundle (§5.9). Reader plus codecs is ~72 KB gzipped against
  // DuckDB's 4.6 MB brotli, and the codecs earn their share back several
  // times over: they are what lets `panel.py` keep zstd, which is 1.3 MB
  // less parquet on every cold visit to this route.
  const [{ parquetMetadataAsync, parquetRead, parquetSchema }, { compressors }] =
    await Promise.all([import("hyparquet"), import("hyparquet-compressors")]);

  const file = {
    byteLength: bytes.byteLength,
    slice: (start: number, end?: number) =>
      bytes.buffer.slice(
        bytes.byteOffset + start,
        bytes.byteOffset + (end ?? bytes.byteLength),
      ) as ArrayBuffer,
  };

  const metadata = await parquetMetadataAsync(file);
  const schema = parquetSchema(metadata);
  const names = schema.children.map((child) => child.element.name);
  const rows = Number(metadata.num_rows);

  const cache = new Map<string, PanelColumn>();
  const inflight = new Map<string, Promise<PanelColumn>>();

  async function decode(key: string): Promise<PanelColumn> {
    if (!names.includes(key)) {
      throw new Error(`${key} is not a column in panel.parquet`);
    }

    let values: PanelColumn = [];
    await parquetRead({
      file,
      metadata,
      compressors,
      columns: [key],
      // Rows arrive as arrays in the order `columns` names them, so one
      // column is one value per row and the transpose is a map.
      onComplete: (data: unknown[][]) => {
        values = data.map((row) => normalize(row[0]));
      },
    });
    return values;
  }

  return {
    rows,
    names,
    column(key: string) {
      const cached = cache.get(key);
      if (cached) return Promise.resolve(cached);

      // Two channels can ask for the same column in the same render; a
      // second decode of 85,000 rows is pure waste.
      const running = inflight.get(key);
      if (running) return running;

      const task = decode(key).then((values) => {
        cache.set(key, values);
        inflight.delete(key);
        return values;
      });
      inflight.set(key, task);
      return task;
    },
    async close() {
      cache.clear();
      inflight.clear();
      pending = null;
    },
  };
}

/**
 * Whatever the decoder produced, as something the rest of the app can
 * compare and reduce.
 *
 * BigInt is the case that matters: parquet INT64 arrives as one, and a
 * BigInt silently fails every arithmetic comparison against a Number.
 * `null` and `undefined` both mean absent and both stay absent — never
 * zero (§5.3.3).
 */
function normalize(value: unknown): number | string | boolean | null {
  if (value === null || value === undefined) return null;
  if (typeof value === "bigint") return Number(value);
  if (typeof value === "number") return Number.isFinite(value) ? value : null;
  if (typeof value === "string" || typeof value === "boolean") return value;
  if (value instanceof Date) return value.getTime();
  return String(value);
}

/**
 * The panel, with real byte counts.
 *
 * §5.9: "the user should know whether they are waiting on 200 KB or
 * 8 MB", and §5.8.8 forbids skeletons — a shimmer implies content shape
 * before it is known, which on a data tool is a small lie. So this
 * reports bytes and the UI renders a determinate bar.
 */
async function fetchPanel(onProgress?: (progress: LoadProgress) => void): Promise<Uint8Array> {
  const response = await fetch(PANEL_URL);
  if (response.status === 404) throw new PanelMissingError();
  if (!response.ok) {
    throw new Error(`${PANEL_URL}: ${response.status} ${response.statusText}`);
  }

  const header = response.headers.get("content-length");
  const total = header ? Number(header) : null;

  let merged: Uint8Array;
  if (!response.body || !onProgress) {
    merged = new Uint8Array(await response.arrayBuffer());
  } else {
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
    merged = new Uint8Array(received);
    let offset = 0;
    for (const chunk of chunks) {
      merged.set(chunk, offset);
      offset += chunk.length;
    }
  }

  // The dev middleware serves any file in the export directory, so a
  // missing parquet arrives as a 404 above — but a *stale* one that is
  // not parquet at all would reach the decoder as a confusing internal
  // error. PAR1 is the format's magic number at both ends of the file.
  if (merged.length < 8 || String.fromCharCode(...merged.slice(0, 4)) !== "PAR1") {
    throw new Error(`${PANEL_URL} is not a parquet file`);
  }

  return merged;
}
