import styles from "./ErrorState.module.css";

export interface ErrorStateProps {
  /** The view's own root class, so the error screen keeps its layout. */
  rootClassName: string | undefined;
  /** The view's h1. */
  title: string;
  /** What failed, e.g. "players.json did not load." */
  heading: string;
  error: Error;
  /** A `python -m web.export` target; omitted where no export can help. */
  command?: string;
  /** "failure" for the committed-JSON views, "quiet" for panel-backed ones. */
  variant?: "failure" | "quiet";
}

/**
 * §5.8.7: errors state what failed and what to do, and never apologise.
 * Always `role="alert"`; the message itself is the raw `Error.message`.
 */
export function ErrorState({
  rootClassName,
  title,
  heading,
  error,
  command,
  variant = "failure",
}: ErrorStateProps) {
  const quiet = variant === "quiet";
  return (
    <main className={rootClassName}>
      <h1 className={styles.title}>{title}</h1>
      <div className={quiet ? styles.quiet : styles.failure} role="alert">
        <p className={quiet ? styles.quietHead : styles.failureHead}>{heading}</p>
        <p className={`${styles.sub} data`}>{error.message}</p>
        {command && (
          <p className={styles.sub}>
            Run <code className="data">{`python -m web.export ${command}`}</code> and reload.
          </p>
        )}
      </div>
    </main>
  );
}
