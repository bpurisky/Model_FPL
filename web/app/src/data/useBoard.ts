/**
 * `board.json`, as the surfaces that need it for §5.5.4's reverse path
 * see it.
 *
 * The Graph Builder, the Form Matrix and Comparison all want the same
 * question answered — "what did the model make of this player?" — and
 * none of them wants to fail because the answer was unavailable. So this
 * resolves to `null` on error rather than throwing: a missing board costs
 * a badge, not a surface.
 *
 * The download itself is shared by `loadBoard`'s own cache, so navigating
 * between the three fetches once. That cache drops failures, so a board
 * that was unavailable is asked for again on the next mount.
 */

import { useEffect, useState } from "react";
import { loadBoard } from "./load";
import type { BoardFile } from "./schema";

const board = (): Promise<BoardFile | null> => loadBoard().catch(() => null);

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
