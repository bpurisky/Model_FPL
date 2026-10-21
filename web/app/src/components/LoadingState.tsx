import type { LoadProgress } from "../data/load";
import styles from "./LoadingState.module.css";

export interface LoadingStateProps {
  /** The view's own root class, so the loading screen keeps its layout. */
  rootClassName: string | undefined;
  /** The view's h1. */
  title: string;
  /** The file being fetched, as named in the byte-count line. */
  file: string;
  /** aria-label for the progress bar. */
  label: string;
  progress: LoadProgress | null;
  /** KB for the JSON exports, MB for panel.parquet. Default KB. */
  unit?: "KB" | "MB";
  /** Extra top margin, for views whose root column gap is tighter. */
  spaced?: boolean;
}

const FORMAT = {
  KB: (bytes: number) => `${Math.round(bytes / 1024)} KB`,
  MB: (bytes: number) => `${(bytes / 1_048_576).toFixed(1)} MB`,
};

/**
 * §5.9: the user should know whether they are waiting on 200 KB or 8 MB.
 * §5.8.8 forbids skeletons — they imply content shape before it is known —
 * so this is a determinate bar with a byte count, or an honest
 * indeterminate bar when the server sent no length (`total` is null for
 * gzip-served JSON, where Content-Length is the compressed size).
 */
export function LoadingState({
  rootClassName,
  title,
  file,
  label,
  progress,
  unit = "KB",
  spaced = false,
}: LoadingStateProps) {
  const format = FORMAT[unit];
  const pct =
    progress?.total != null ? Math.round((progress.received / progress.total) * 100) : null;
  return (
    <main className={rootClassName}>
      <h1 className={styles.title}>{title}</h1>
      <div className={spaced ? `${styles.loading} ${styles.spaced}` : styles.loading}>
        <div
          className={styles.track}
          role="progressbar"
          aria-valuenow={pct ?? undefined}
          aria-valuemin={0}
          aria-valuemax={100}
          aria-label={label}
        >
          <div className={styles.fill} style={{ inlineSize: pct === null ? "30%" : `${pct}%` }} />
        </div>
        <p className={`${styles.sub} data`}>
          {progress
            ? `${file} — ${format(progress.received)}${progress.total ? ` of ${format(progress.total)}` : ""}`
            : `requesting ${file}`}
        </p>
      </div>
    </main>
  );
}
