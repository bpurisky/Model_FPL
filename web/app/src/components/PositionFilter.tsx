import styles from "./PositionFilter.module.css";

export interface PositionFilterProps {
  groups: { key: string; n_player_seasons: number; mixed_position: boolean }[];
  value: string;
  onChange: (value: string) => void;
}

/**
 * §5.4.1's primary interaction: "Position filter (GK/DEF/MID/FWD/all)
 * swaps to the precomputed per-position matrix. Must be instant — hence
 * precomputation, not client-side recompute."
 *
 * Rendered as buttons rather than a select so the whole set is visible
 * and one click away; §5.9 budgets this interaction at 100ms with no
 * spinner, and every matrix it can reach is already in memory.
 */
export function PositionFilter({ groups, value, onChange }: PositionFilterProps) {
  return (
    <div className={styles.bar} role="group" aria-label="Position filter">
      <span className={styles.legend}>Position</span>
      {groups.map((group) => (
        <button
          key={group.key}
          type="button"
          className={`${styles.option} ${group.key === value ? styles.active : ""}`}
          aria-pressed={group.key === value}
          onClick={() => onChange(group.key)}
        >
          {group.key === "all" ? "All" : group.key}
          <span className="data"> {group.n_player_seasons}</span>
        </button>
      ))}
    </div>
  );
}
