/**
 * The Data & API page: the committed exports, documented as the public
 * JSON API they already are.
 *
 * Every file the app fetches is a static file at a stable URL on the
 * deployed site, so there is no second API to describe — this lists the
 * files `data/catalog.ts` knows about, with the URL built from the app's
 * own `DATA_BASE` at runtime (so it is right on GitHub Pages under
 * `/<repo>/` and in dev alike), what each contains, and how often it
 * refreshes.
 *
 * Nothing is fetched on load. A file's `generated_at` is shown only when
 * the reader asks, through the same loader the other views use: the
 * response is validated and cached there, and the 2 MB `players.json` is
 * not downloaded by someone who only wanted its address.
 *
 * No licence or usage terms are stated: the repository carries none, and
 * inventing one here would be a claim nobody made.
 */

import { useState } from "react";
import { CADENCE_TEXT, DATA_FILES, type DataFile } from "../data/catalog";
import { DATA_BASE } from "../data/load";
import { EXPECTED_CONTRACT_VERSION } from "../data/schema";
import styles from "./DataApi.module.css";

/** Absolute URL for a file, from wherever the app is being served. */
export function publicUrl(name: string, origin: string = window.location.origin): string {
  return new URL(`${DATA_BASE}/${name}`, origin).href;
}

type Probe =
  | { status: "idle" }
  | { status: "loading" }
  | { status: "ready"; generatedAt: string }
  | { status: "error"; message: string };

export function DataApi() {
  return (
    <main className={styles.data}>
      <header>
        <h1 className={styles.title}>Data &amp; API</h1>
        <p className={styles.sub}>
          The files this site reads are plain static files at the addresses below: fetch them
          directly, no key and no separate endpoint. Each carries a{" "}
          <code className="data">header</code> with <code className="data">generated_at</code>,{" "}
          <code className="data">model_git_sha</code> and{" "}
          <code className="data">contract_version</code>. The contract is at version{" "}
          <span className="data">{EXPECTED_CONTRACT_VERSION}</span>; fields added since are
          optional, and a change to the meaning of an existing field would bump it.
        </p>
      </header>

      <ul className={styles.list}>
        {DATA_FILES.map((file) => (
          <FileCard key={file.name} file={file} />
        ))}
      </ul>

      <p className={styles.note}>
        A file is rewritten only when its numbers change, so its{" "}
        <code className="data">generated_at</code> is when it last changed, not when it last ran.
      </p>
    </main>
  );
}

function FileCard({ file }: { file: DataFile }) {
  const [probe, setProbe] = useState<Probe>({ status: "idle" });
  const url = publicUrl(file.name);

  const check = async () => {
    if (!file.loadHeader) return;
    setProbe({ status: "loading" });
    try {
      const loaded = await file.loadHeader();
      setProbe({ status: "ready", generatedAt: loaded.header.generated_at });
    } catch (error) {
      setProbe({ status: "error", message: (error as Error).message });
    }
  };

  return (
    <li className={styles.file}>
      <div className={styles.fileHead}>
        <h2 className={styles.name}>{file.name}</h2>
        <a className={`${styles.url} data`} href={url}>
          {url}
        </a>
      </div>
      <p className={styles.desc}>{file.description}</p>
      <dl className={styles.meta}>
        <dt>Refresh</dt>
        <dd>{CADENCE_TEXT[file.cadence]}</dd>
        <dt>Contract</dt>
        <dd className="data">
          {file.format === "json"
            ? `v${EXPECTED_CONTRACT_VERSION}`
            : `v${EXPECTED_CONTRACT_VERSION} (header in the parquet footer, key fpl_trends_header)`}
        </dd>
        {file.format === "parquet" && (
          <>
            <dt>Note</dt>
            <dd>A build artifact, published with the site but not committed to the repository.</dd>
          </>
        )}
        {file.loadHeader && (
          <>
            <dt>Generated</dt>
            <dd>
              {probe.status === "ready" ? (
                <span className="data">
                  {new Date(probe.generatedAt).toISOString().replace("T", " ").slice(0, 16)}Z
                </span>
              ) : probe.status === "error" ? (
                <span className={styles.error} role="alert">
                  {probe.message}
                </span>
              ) : (
                <button
                  type="button"
                  className={styles.action}
                  onClick={check}
                  disabled={probe.status === "loading"}
                >
                  {probe.status === "loading" ? "Loading…" : "Show generated_at"}
                </button>
              )}
            </dd>
          </>
        )}
      </dl>
    </li>
  );
}
