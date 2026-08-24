/**
 * The four drop zones (§5.4.2).
 *
 * > "Drop zones — exactly four. X, Y, Color, Wrap. No Overlay, no Group
 * > X/Y, no Size, no Shape, no Page."
 *
 * The list is closed and comes from `CHANNELS`, so a fifth cannot be
 * added here by accident — it would have to be added to the encoding
 * type, the mark-inference table, and its exhaustive test first.
 *
 * §5.10: "Graph Builder drop zones are keyboard-operable: a column can be
 * assigned to a channel without a pointer, and current assignments are
 * announced." Each zone is a real `<button>`, so it is in the tab order
 * and activates on Enter and Space for free. The announcement is the
 * `aria-live` region below, which reads the whole encoding after every
 * change rather than a fragment of it — a reader who just filled Y needs
 * to know what X holds to know what will be drawn.
 */

import type { ColumnSpec } from "../data/schema";
import { CHANNELS, type Channel, type Encoding } from "../encoding/spec";
import styles from "./DropZones.module.css";

const LABELS: Record<Channel, string> = {
  x: "X",
  y: "Y",
  color: "Colour",
  wrap: "Wrap",
};

const HINTS: Record<Channel, string> = {
  x: "a number, a category, or the gameweek",
  y: "usually a number",
  color: "a category to split by, or a number to shade by",
  wrap: "a category — one panel per value",
};

export interface DropZonesProps {
  encoding: Encoding;
  columns: ReadonlyMap<string, ColumnSpec>;
  /** The column picked up in the list, if any. */
  held: string | null;
  onAssign: (channel: Channel, key: string | null) => void;
  onHold: (key: string | null) => void;
}

export function DropZones({ encoding, columns, held, onAssign, onHold }: DropZonesProps) {
  const describe = (channel: Channel) => {
    const key = encoding[channel];
    if (!key) return "empty";
    return columns.get(key)?.label ?? key;
  };

  return (
    <div className={styles.zones}>
      {CHANNELS.map((channel) => {
        const key = encoding[channel];
        const spec = key ? columns.get(key) : undefined;
        return (
          <div key={channel} className={styles.zoneWrap}>
            <button
              type="button"
              className={styles.zone}
              data-filled={key ? true : undefined}
              data-armed={held ? true : undefined}
              onDragOver={(event) => {
                event.preventDefault();
                event.dataTransfer.dropEffect = "copy";
              }}
              onDrop={(event) => {
                event.preventDefault();
                const dropped = event.dataTransfer.getData("text/plain");
                if (dropped) onAssign(channel, dropped);
                onHold(null);
              }}
              onClick={() => {
                if (held) {
                  onAssign(channel, held);
                  onHold(null);
                } else if (key) {
                  // A filled zone with nothing held picks its column back
                  // up, so a reader can move X to Y without a round trip
                  // through the list.
                  onHold(key);
                  onAssign(channel, null);
                }
              }}
              aria-label={`${LABELS[channel]}: ${describe(channel)}. ${
                held ? "Press to assign the held column." : "Press to pick this column up."
              }`}
            >
              <span className={styles.channel}>{LABELS[channel]}</span>
              {spec ? (
                <span className={styles.assigned}>{spec.label}</span>
              ) : (
                <span className={styles.hint}>{HINTS[channel]}</span>
              )}
            </button>

            {key && (
              <button
                type="button"
                className={styles.clear}
                onClick={() => onAssign(channel, null)}
                aria-label={`Clear ${LABELS[channel]}`}
                title={`Clear ${LABELS[channel]}`}
              >
                ×
              </button>
            )}
          </div>
        );
      })}

      {/*
       * The announcement §5.10 asks for. `polite` rather than `assertive`:
       * assigning a column is the user's own action and does not need to
       * interrupt them mid-word.
       */}
      <p className={styles.announce} aria-live="polite">
        {CHANNELS.map((channel) => `${LABELS[channel]} ${describe(channel)}`).join(", ")}
      </p>
    </div>
  );
}
