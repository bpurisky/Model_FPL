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

const XiPlayer = Player.extend({ captain: z.boolean() });

const Recommendation = z.object({
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
  squad_size: z.number(),
  unchanged_from_current: z.number(),
});

export type OptimizerPlayer = z.infer<typeof Player>;
export type OptimizerTransfer = z.infer<typeof Transfer>;
export type OptimizerTemplateRisk = z.infer<typeof TemplateRisk>;
export type OptimizerXiPlayer = z.infer<typeof XiPlayer>;
export type OptimizerRecommendation = z.infer<typeof Recommendation>;

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

  let response: Response;
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
    });
  } catch (error) {
    throw new OptimizerError(
      `Could not reach the optimizer backend at ${OPTIMIZER_API_URL}: ${(error as Error).message}`,
    );
  }

  const body: unknown = await response.json().catch(() => null);

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
