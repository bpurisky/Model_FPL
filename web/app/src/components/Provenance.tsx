import type { Header } from "../data/schema";
import { freshness, STALE_AFTER_HOURS } from "../data/load";
import styles from "./Provenance.module.css";

export interface ProvenanceProps {
  header: Header;
  basis: string;
}

/**
 * §5.14.1: "Every number on screen is traceable to a `model_git_sha`
 * visible in the UI." §5.6.3 adds the normalization basis and the
 * staleness statement — "stale data is self-announcing".
 *
 * Always visible rather than behind a tooltip, because a provenance you
 * have to go looking for is one nobody checks.
 */
export function Provenance({ header, basis }: ProvenanceProps) {
  const { stale, unchanged, age: hours } = freshness(header.generated_at);
  const age = Math.floor(hours);

  return (
    <div className={styles.bar}>
      <span className={styles.item}>
        <span className={styles.key}>sha</span>
        <span className={`${styles.value} data`}>
          {header.model_git_sha ? header.model_git_sha.slice(0, 10) : "unknown"}
        </span>
      </span>
      <span className={styles.item}>
        <span className={styles.key}>generated</span>
        <span className={`${stale ? styles.stale : styles.value} data`}>
          {new Date(header.generated_at).toISOString().replace("T", " ").slice(0, 16)}Z
          {stale && ` · ${age}h old, past the ${STALE_AFTER_HOURS}h threshold`}
          {unchanged && ` · unchanged since, rechecked ${age}h ago`}
        </span>
      </span>
      <span className={styles.item}>
        <span className={styles.key}>basis</span>
        <span className={`${styles.value} data`}>{basis}</span>
      </span>
      <span className={styles.item}>
        <span className={styles.key}>contract</span>
        <span className={`${styles.value} data`}>v{header.contract_version ?? 1}</span>
      </span>
    </div>
  );
}
