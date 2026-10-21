/**
 * The Squad Optimizer's client: the one surface in this app that calls a
 * live backend rather than reading a committed export from `data/v1/`.
 *
 * Why this file exists at all, rather than the surface staying the
 * `out_of_phase` stub it was through 2026-08-25: `squad/optimize.py`'s
 * ILP solve cannot run in the browser (§5.1.1/§5.6 forbid a server
 * connection *and* client-side inference respectively, and a live solve
 * is inference, not a reduction of one) and cannot be precomputed (it
 * takes an arbitrary team ID as input, decided by the reader at request
 * time, not known at export time). `service/app.py` is the backend this
 * calls; see its module docstring for the full architecture decision.
 *
 * This schema is deliberately separate from `data/schema.ts`. That file
 * is one half of a two-sided contract with `web/export/contract.py`,
 * guarded by `tests/test_schema_ts.py` comparing shapes on the Python
 * side. There is no Python contract test for a live HTTP response the
 * way there is for a committed export, so folding this in would imply a
 * guarantee that does not exist. `tests/test_service.py::
 * test_shape_response_pairs_transfers_and_marks_the_captain` is the one
 * test standing behind this shape, from the Python side.
 */

import { z } from "zod";

const Player = z.object({
  element_id: z.number(),
  name: z.string(),
  position: z.enum(["GK", "DEF", "MID", "FWD"]),
  club: z.string(),
  now_cost: z.number(),
});

const Transfer = z.object({ out: Player, in: Player });

const TemplateRisk = z.object({
  element_id: z.number(),
  name: z.string(),
  message: z.string(),
});

const XiPlayer = Player.extend({ captain: z.boolean(), vice_captain: z.boolean().optional() });

/** A later week the optimizer planned around this one's decision. */
const PlannedWeek = z.object({
  gw: z.number(),
  free_transfers: z.number(),
  hits: z.number(),
  transfers: z.array(Transfer),
});

/** One chip played at one gameweek: projected extra points, not a promise. */
const ChipGain = z.object({
  chip: z.enum(["bboost", "3xc", "freehit", "wildcard"]),
  gw: z.number(),
  gain: z.number(),
  /** False when the entry's history shows the chip already used this half-season. */
  available: z.boolean().optional(),
});

const Recommendation = z.object({
  // Team rating (squad/rating.py): projected xP over the horizon. Optional
  // so a backend that predates them still parses.
  current_xi_xp: z.number().optional(),
  recommended_xp: z.number().optional(),
  optimum_xp: z.number().optional(),
  team_rating: z.number().optional(),
  chips: z.array(ChipGain).optional(),
  chip_best_gw: z.record(z.string(), z.number()).optional(),
  entry_id: z.number(),
  data_gw: z.number(),
  history_gws: z.number(),
  teams_with_played_data: z.number(),
  teams_total: z.number(),
  caveat: z.string(),
  free_transfers: z.number(),
  bank: z.number(),
  horizon: z.array(z.number()),
  transfers: z.array(Transfer),
  hits_taken: z.number(),
  bank_after: z.number(),
  template_risk: z.array(TemplateRisk),
  starting_xi: z.record(z.string(), z.array(XiPlayer)),
  bench_order: z.array(Player),
  plan: z.array(PlannedWeek).optional(),
  squad_size: z.number(),
  unchanged_from_current: z.number(),
});

export type OptimizerPlayer = z.infer<typeof Player>;
export type OptimizerTransfer = z.infer<typeof Transfer>;
export type OptimizerTemplateRisk = z.infer<typeof TemplateRisk>;
export type OptimizerXiPlayer = z.infer<typeof XiPlayer>;
export type OptimizerChipGain = z.infer<typeof ChipGain>;
export type OptimizerRecommendation= z.infer<typeof Recommendation>;
export type OptimizerPlannedWeek = z.infer<typeof PlannedWeek>;

/**
 * Set at build time via `VITE_OPTIMIZER_API_URL` (see `.env.example`).
 * Left unset in every environment that hasn't deployed `service/app.py`
 * yet -- §7.3 requires the analytics half to render with the Worker
 * offline, and the same argument applies here: a reader on a fresh
 * clone or a preview deploy with no backend configured should see an
 * honest explanation, not a network error with no context.
 */
export const OPTIMIZER_API_URL = (import.meta.env.VITE_OPTIMIZER_API_URL as string | undefined)?.replace(
  /\/+$/,
  "",
);

/** `OptimizerError.status` for a client-side timeout — not an HTTP status. */
export const TIMEOUT_STATUS = 0;

export class OptimizerError extends Error {
  constructor(
    message: string,
    readonly status?: number,
  ) {
    super(message);
    this.name = "OptimizerError";
  }
}

export interface RecommendOptions {
  horizon?: number[] | undefined;
  maxTransfers?: number | undefined;
  hitCost?: number | undefined;
  /** Overrides `RECOMMEND_TIMEOUT_MS`; tests use a short one. */
  timeoutMs?: number | undefined;
}

/**
 * Long enough to cover a free-tier cold start (~1 minute) plus a slow
 * solve; short enough that a request the host or the FPL API has quietly
 * dropped ends in an explanation and a retry button, not an endless
 * "Solving…". Before this existed, a stalled request never ended at all.
 */
export const RECOMMEND_TIMEOUT_MS = 120_000;

/**
 * A team ID from whatever the reader pasted: the bare number, or any FPL
 * URL that contains `/entry/<id>/` — which is where the ID is easiest to
 * find (the Points page's address bar). Returns undefined for anything
 * that doesn't yield a positive integer.
 */
export function parseEntryId(raw: string): number | undefined {
  const trimmed = raw.trim();
  const fromUrl = /\/entry\/(\d+)/.exec(trimmed);
  const digits = fromUrl ? fromUrl[1] : /^\d+$/.test(trimmed) ? trimmed : undefined;
  if (digits === undefined) return undefined;
  const id = Number(digits);
  return Number.isSafeInteger(id) && id > 0 ? id : undefined;
}

/**
 * Pings `/api/health` so a sleeping free-tier backend starts waking while
 * the reader is still typing their team ID. Resolves true once the
 * backend answers, false if it can't be reached; never throws — this is
 * a head start, not a request anyone waits on to succeed.
 */
export async function warmUp(): Promise<boolean> {
  if (!OPTIMIZER_API_URL) return false;
  try {
    const response = await fetch(`${OPTIMIZER_API_URL}/api/health`);
    return response.ok;
  } catch {
    return false;
  }
}

function errorDetail(body: unknown, fallback: string): string {
  if (body && typeof body === "object" && "detail" in body) {
    const detail = (body as { detail: unknown }).detail;
    if (typeof detail === "string") return detail;
  }
  return fallback;
}

export async function fetchRecommendation(
  entryId: number,
  options: RecommendOptions = {},
): Promise<OptimizerRecommendation> {
  if (!OPTIMIZER_API_URL) {
    throw new OptimizerError(
      "No optimizer backend is configured (VITE_OPTIMIZER_API_URL is unset at build time).",
    );
  }

  const timeoutMs = options.timeoutMs ?? RECOMMEND_TIMEOUT_MS;
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);

  let response: Response;
  let body: unknown;
  try {
    response = await fetch(`${OPTIMIZER_API_URL}/api/recommend`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({
        entry_id: entryId,
        ...(options.horizon ? { horizon: options.horizon } : {}),
        ...(options.maxTransfers != null ? { max_transfers: options.maxTransfers } : {}),
        ...(options.hitCost != null ? { hit_cost: options.hitCost } : {}),
      }),
      signal: controller.signal,
    });
    body = await response.json().catch(() => null);
  } catch (error) {
    if (controller.signal.aborted) {
      throw new OptimizerError(
        `The optimizer didn't answer within ${Math.round(timeoutMs / 1000)} seconds. The server may still have ` +
          "been waking up, or the FPL API was slow to respond.",
        TIMEOUT_STATUS,
      );
    }
    throw new OptimizerError(
      `Could not reach the optimizer backend at ${OPTIMIZER_API_URL}: ${(error as Error).message}`,
    );
  } finally {
    clearTimeout(timer);
  }

  if (!response.ok) {
    throw new OptimizerError(errorDetail(body, response.statusText), response.status);
  }

  const parsed = Recommendation.safeParse(body);
  if (!parsed.success) {
    const issue = parsed.error.issues[0];
    const path = issue?.path.join(".") ?? "(root)";
    throw new OptimizerError(`Unexpected response shape: ${path}: ${issue?.message ?? "failed validation"}`);
  }
  return parsed.data;
}
