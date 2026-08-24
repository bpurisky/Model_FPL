/**
 * §5.5.4's reverse path.
 *
 * > "The reverse path exists too: any player selected in Graph Builder or
 * > Form Matrix shows their current bucket as a small inline badge,
 * > linking to their Model Board card."
 *
 * The badge is deliberately quiet, and that is a judgement the accuracy
 * panel forces. Two of the four buckets measured **worse** than the
 * players they were picked out from — rising at −0.077 forward points,
 * declining at −0.144 — so a badge that announced "Rising" in a colour
 * would be the app leaning on a classification its own export says is
 * worth nothing. It reads as a label with a link, not as a verdict.
 *
 * §5.8.6's rule points the same way: model-authored claims are marked
 * structurally, not chromatically, and there are no new colours to spend.
 */

import { useApp } from "../app/state";
import type { BoardFile } from "../data/schema";
import styles from "./BucketBadge.module.css";

export interface BucketBadgeProps {
  board: BoardFile | null;
  elementId: number;
}

export function BucketBadge({ board, elementId }: BucketBadgeProps) {
  const { dispatch } = useApp();
  if (!board) return null;

  const player = board.players.find((entry) => entry.element_id === elementId);
  // Most players are not classified at all — the board keeps only those
  // with enough gameweeks behind them — and saying nothing is right.
  if (!player) return null;

  return (
    <button
      type="button"
      className={styles.badge}
      data-bucket={player.bucket}
      title={`The model puts ${player.name} in "${player.bucket}", rank ${player.rank} among ${player.position}. Open his card.`}
      onClick={() => {
        dispatch({ type: "filters", filters: { positions: [player.position] } });
        dispatch({ type: "navigate", view: "board" });
      }}
    >
      <span className={styles.label}>{player.bucket}</span>
      <span className={styles.rank}>#{player.rank}</span>
    </button>
  );
}
