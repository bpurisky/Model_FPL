/**
 * Fetching and validating the committed exports (§5.3.4, §5.12).
 *
 * Every payload is parsed through the zod schema before anything renders.
 * That is the whole point of the contract being hand-written on both
 * sides: a field that changed shape upstream fails here, at the boundary,
 * with a message naming it — rather than three layers in as `undefined`
 * rendering an empty cell that looks like a real absence.
 *
 * §5.12 requires the build to fail loudly on a contract version mismatch
 * rather than degrade silently. A file written under a different contract
 * may carry the same field names and different meanings, which is the
 * failure a version number exists to catch.
 */

import { z } from "zod";
import {
  ColumnsFile,
  CorrelationsFile,
  EXPECTED_CONTRACT_VERSION,
  ObservationsFile,
  PlayersFile,
} from "./schema";

const BASE = "/data/v1";

export class ContractError extends Error {
  constructor(
    readonly file: string,
    message: string,
  ) {
    super(message);
    this.name = "ContractError";
  }
}

export interface LoadProgress {
  /** Bytes received so far, when the server reported a length. */
  received: number;
  /** Total bytes, or null when the response was not length-delimited. */
  total: number | null;
}

/**
 * §5.9 requires the user to know whether they are waiting on 200 KB or
 * 8 MB, and §5.8.8 forbids skeletons: a shimmer implies content shape
 * before it is known, which on a data tool is a small lie. So the fetch
 * reports real bytes and the UI renders a determinate bar.
 */
async function fetchWithProgress(
  url: string,
  onProgress?: (progress: LoadProgress) => void,
): Promise<unknown> {
  const response = await fetch(url);
  if (!response.ok) {
    throw new ContractError(url, `${response.status} ${response.statusText}`);
  }

  const header = response.headers.get("content-length");
  const total = header ? Number(header) : null;

  if (!response.body || !onProgress) {
    return response.json();
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
  return JSON.parse(new TextDecoder().decode(merged));
}

function assertContractVersion(file: string, version: number): void {
  if (version !== EXPECTED_CONTRACT_VERSION) {
    throw new ContractError(
      file,
      `contract version ${version}, expected ${EXPECTED_CONTRACT_VERSION}. ` +
        "The export and the app were built against different contracts; " +
        "rebuild the export or update the app rather than reading it anyway.",
    );
  }
}

async function loadFile<T extends z.ZodTypeAny>(
  name: string,
  schema: T,
  onProgress?: (progress: LoadProgress) => void,
): Promise<z.infer<T>> {
  const payload = await fetchWithProgress(`${BASE}/${name}`, onProgress);
  const parsed = schema.safeParse(payload);
  if (!parsed.success) {
    // The first issue is the useful one; a full zod dump buries it.
    const issue = parsed.error.issues[0];
    const path = issue?.path.join(".") ?? "(root)";
    throw new ContractError(name, `${path}: ${issue?.message ?? "failed validation"}`);
  }
  assertContractVersion(name, parsed.data.header.contract_version ?? EXPECTED_CONTRACT_VERSION);
  return parsed.data;
}

export const loadCorrelations = (onProgress?: (p: LoadProgress) => void) =>
  loadFile("correlations.json", CorrelationsFile, onProgress);

export const loadColumns = (onProgress?: (p: LoadProgress) => void) =>
  loadFile("columns.json", ColumnsFile, onProgress);

export const loadPlayers = (onProgress?: (p: LoadProgress) => void) =>
  loadFile("players.json", PlayersFile, onProgress);

/**
 * The values behind the matrix (§5.6.1). Fetched only when the reader
 * changes the season selection — it is several times the size of the
 * correlations themselves, and most loads never need it.
 */
export const loadObservations = (onProgress?: (p: LoadProgress) => void) =>
  loadFile("observations.json", ObservationsFile, onProgress);

/**
 * Whether the export is old enough to say so on screen (§5.6.3).
 * The threshold is `config/frontend.yaml:staleness.warn_after_hours`; it
 * is duplicated here as a constant rather than fetched because the app
 * has no reader for that file and a wrong-by-a-few-hours warning is a
 * smaller error than an extra request on first paint.
 */
export const STALE_AFTER_HOURS = 6;

export function hoursSince(generatedAt: string, now: Date = new Date()): number {
  return (now.getTime() - new Date(generatedAt).getTime()) / 3_600_000;
}

export function isStale(generatedAt: string, now: Date = new Date()): boolean {
  return hoursSince(generatedAt, now) > STALE_AFTER_HOURS;
}
