import { copyFileSync, mkdirSync, readFileSync, readdirSync, statSync } from "node:fs";
import { join, resolve } from "node:path";
import react from "@vitejs/plugin-react";
import { defineConfig, type Plugin } from "vite";

/**
 * The committed exports live in `data/web/v1/` at the repo root, not in
 * `public/`. Copying them into the app would put two copies of every
 * number in the repository, and §5.3.4 is explicit that the committed
 * JSON *is* the artifact — a second copy is a second thing to go stale.
 *
 * So: served from disk in dev, copied once at build. The parquet files
 * are deliberately excluded; §5.3.4 does not commit them and §5.14.8
 * requires the parquet-backed routes to show their empty state rather
 * than an error when they are absent.
 */
const EXPORT_DIR = resolve(__dirname, "../../data/web/v1");
const SERVE_AT = "/data/v1/";

function committedExports(): Plugin {
  return {
    name: "fpl-committed-exports",
    configureServer(server) {
      server.middlewares.use((req, res, next) => {
        if (!req.url?.startsWith(SERVE_AT)) return next();
        const name = req.url.slice(SERVE_AT.length).split("?")[0] ?? "";
        // No path traversal: only a bare filename from the export dir.
        if (!name || name.includes("/") || name.includes("..")) return next();
        const path = join(EXPORT_DIR, name);
        try {
          if (!statSync(path).isFile()) return next();
        } catch {
          return next();
        }
        res.setHeader("Content-Type", name.endsWith(".json") ? "application/json" : "application/octet-stream");
        res.end(readFileSync(path));
      });
    },
    closeBundle() {
      const out = resolve(__dirname, "dist", "data", "v1");
      mkdirSync(out, { recursive: true });
      for (const name of readdirSync(EXPORT_DIR)) {
        if (!name.endsWith(".json")) continue;
        copyFileSync(join(EXPORT_DIR, name), join(out, name));
      }
    },
  };
}

export default defineConfig({
  plugins: [react(), committedExports()],
  build: {
    // §5.9: the initial bundle is budgeted at 250 KB gzipped, and the way
    // that budget is lost is by not noticing.
    chunkSizeWarningLimit: 250,
    target: "es2022",
  },
  test: {
    globals: true,
    environment: "jsdom",
    setupFiles: ["./src/test/setup.ts"],
  },
});
