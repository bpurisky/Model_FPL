/**
 * `timeseries.parquet`, for §5.4.8.
 *
 * A separate module from `session.ts` and deliberately so: that one owns
 * the 3.1 MB player-gameweek panel and caches columns out of it, and this
 * is 150 KB of snapshot history with a different grain and a different
 * lifetime. Sharing the session would mean the Trend Explorer paid for
 * the panel it never reads.
 *
 * Rows come back whole rather than by column. At 30,000 rows and six
 * fields the difference is not worth a second cache, and the overlay
 * needs every field of the rows it draws anyway.
 *
 * §5.6 is not engaged here at all: this module reads and filters, and
 * every number it hands over is one the collector recorded.
 */

import { DATA_BASE, type LoadProgress } from "../data/load";

/** Thrown when the artifact is absent — §5.14.8's empty state. */
export class SeriesMissingError extends Error {
  constructor() {
    super(
      "timeseries.parquet is not published. It is a build artifact, not committed data " +
        "(§5.3.4) — run `python -m web.export timeseries` to create it.",
    );
    this.name = "SeriesMissingError";
  }
}

export interface SeriesPoint {
  /** Epoch milliseconds, so the axis is arithmetic rather than string. */
  snapshot_ts: number;
  gw: number | null;
  element_id: number;
  name: string;
  team: string;
  position: string;
  now_cost: number | null;
  selected_by_percent: number | null;
  model_projection: number | null;
}

const URL_PATH = `${DATA_BASE}/timeseries.parquet`;

const FIELDS = [
  "snapshot_ts",
  "gw",
  "element_id",
  "name",
  "team",
  "position",
  "now_cost",
  "selected_by_percent",
  "model_projection",
] as const;

let pending: Promise<SeriesPoint[]> | null = null;

export function openSeries(
  onProgress?: (progress: LoadProgress) => void,
): Promise<SeriesPoint[]> {
  pending ??= read(onProgress).catch((error) => {
    pending = null;
    throw error;
  });
  return pending;
}

async function read(onProgress?: (progress: LoadProgress) => void): Promise<SeriesPoint[]> {
  const response = await fetch(URL_PATH);
  if (response.status === 404) throw new SeriesMissingError();
  if (!response.ok) {
    throw new Error(`${URL_PATH}: ${response.status} ${response.statusText}`);
  }

  const header = response.headers.get("content-length");
  const total = header ? Number(header) : null;

  let bytes: Uint8Array;
  if (!response.body || !onProgress) {
    bytes = new Uint8Array(await response.arrayBuffer());
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
    bytes = new Uint8Array(received);
    let offset = 0;
    for (const chunk of chunks) {
      bytes.set(chunk, offset);
      offset += chunk.length;
    }
  }

  if (bytes.length < 8 || String.fromCharCode(...bytes.slice(0, 4)) !== "PAR1") {
    throw new Error(`${URL_PATH} is not a parquet file`);
  }

  const [{ parquetRead }, { compressors }] = await Promise.all([
    import("hyparquet"),
    import("hyparquet-compressors"),
  ]);

  const file = {
    byteLength: bytes.byteLength,
    slice: (start: number, end?: number) =>
      bytes.buffer.slice(
        bytes.byteOffset + start,
        bytes.byteOffset + (end ?? bytes.byteLength),
      ) as ArrayBuffer,
  };

  let rows: SeriesPoint[] = [];
  await parquetRead({
    file,
    compressors,
    columns: [...FIELDS],
    // Rows arrive as arrays in the order `columns` names them.
    onComplete: (data: unknown[][]) => {
      rows = data.map((row) => ({
        snapshot_ts: asTime(row[0]),
        gw: asNumber(row[1]),
        element_id: asNumber(row[2]) ?? -1,
        name: String(row[3] ?? ""),
        team: String(row[4] ?? ""),
        position: String(row[5] ?? ""),
        now_cost: asNumber(row[6]),
        selected_by_percent: asNumber(row[7]),
        model_projection: asNumber(row[8]),
      }));
    },
  });

  return rows;
}

/**
 * Parquet timestamps arrive as a Date, a BigInt of microseconds, or a
 * number, depending on the writer. All three become epoch milliseconds,
 * because an axis needs to subtract.
 */
function asTime(value: unknown): number {
  if (value instanceof Date) return value.getTime();
  if (typeof value === "bigint") return Number(value / 1000n);
  if (typeof value === "number") return value;
  const parsed = Date.parse(String(value));
  return Number.isFinite(parsed) ? parsed : 0;
}

/** §5.3.3: absent stays absent. Never zero. */
function asNumber(value: unknown): number | null {
  if (value === null || value === undefined) return null;
  if (typeof value === "bigint") return Number(value);
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : null;
}
