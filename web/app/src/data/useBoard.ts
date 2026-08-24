/**
 * `board.json`, loaded once and shared across the surfaces that need it
 * for §5.5.4's reverse path.
 *
 * The Graph Builder, the Form Matrix and Comparison all want the same
 * question answered — "what did the model make of this player?" — and
 * none of them wants to fail because the answer was unavailable. So this
 * resolves to `null` on error rather than throwing: a missing board costs
 * a badge, not a surface.
 *
 * The promise is memoised at module scope, so navigating between the
 * three fetches once.
 */

import { useEffect, useState } from "react";
import { loadBoard } from "./load";
import type { BoardFile } from "./schema";

let pending: Promise<BoardFile | null> | null = null;

function board(): Promise<BoardFile | null> {
  pending ??= loadBoard().catch(() => null);
  return pending;
}

export function useBoard(): BoardFile | null {
  const [file, setFile] = useState<BoardFile | null>(null);

  useEffect(() => {
    let cancelled = false;
    board().then((loaded) => {
      if (!cancelled) setFile(loaded);
    });
    return () => {
      cancelled = true;
    };
  }, []);

  return file;
}
