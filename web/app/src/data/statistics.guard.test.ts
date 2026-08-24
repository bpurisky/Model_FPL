import { readFileSync, readdirSync, statSync } from "node:fs";
import { join, relative, resolve, sep } from "node:path";
import { describe, expect, it } from "vitest";

/**
 * §5.11.2's last item, and the spec is candid about what it is:
 *
 * > "A test asserting no z-score, percentile, slope, or regression
 * > appears in `src/` outside `data/spearman.ts`. A grep-level guard is
 * > crude and it is the guard most likely to catch the §5.6 erosion this
 * > spec predicts."
 *
 * Crude, so it needs to be crude about the right thing. A literal grep
 * for "z-score" fails on every comment in `query/reduce.ts` explaining
 * why it does not compute one, which would make the guard noise inside a
 * week and then make it deleted. So this strips comments and string
 * literals first and then looks at what is left — the code.
 *
 * Three signals, each chosen because it is hard to compute a forbidden
 * statistic without tripping it:
 *
 *   1. `Math.sqrt` — a standard deviation, a variance, a Pearson
 *      denominator and a confidence interval all need it. `spearman.ts`
 *      is the one module §5.6.1 licenses to have one.
 *   2. A binding *named* for a forbidden operation. Reading an exported
 *      `p_value` off a validated payload is fine and necessary; declaring
 *      `function pValue()` is the thing §5.6 forbids.
 *   3. A forbidden aggregate inside a SQL string. This one guards the
 *      other half of the §5.6.2 bargain: the reductions have one
 *      implementation, in `reduce.ts`, held against Python by a golden
 *      test. An `avg()` in a query string would be a second one that no
 *      test in CI ever runs.
 */

const SRC = resolve(__dirname, "..");

/** §5.6.1's single licensed exception, by path. */
const SPEARMAN = join("data", "spearman.ts");

function sources(dir: string, out: string[] = []): string[] {
  for (const entry of readdirSync(dir)) {
    const path = join(dir, entry);
    if (statSync(path).isDirectory()) {
      sources(path, out);
    } else if (/\.tsx?$/.test(entry)) {
      out.push(path);
    }
  }
  return out;
}

/**
 * Comments and string literals removed, so the guard reads code rather
 * than prose about code. Deliberately simple — a regex-based stripper
 * mangles a few exotic cases, and every way it is wrong leaves *more*
 * text in rather than less, which fails safe for a guard.
 */
function code(text: string): string {
  return text
    .replace(/\/\*[\s\S]*?\*\//g, " ")
    .replace(/(^|[^:])\/\/.*$/gm, "$1 ")
    .replace(/`(?:[^`\\]|\\.)*`/g, '""')
    .replace(/'(?:[^'\\\n]|\\.)*'/g, '""')
    .replace(/"(?:[^"\\\n]|\\.)*"/g, '""');
}

/** String literals only — where SQL lives. */
function strings(text: string): string[] {
  const withoutComments = text
    .replace(/\/\*[\s\S]*?\*\//g, " ")
    .replace(/(^|[^:])\/\/.*$/gm, "$1 ");
  return [
    ...(withoutComments.match(/`(?:[^`\\]|\\.)*`/g) ?? []),
    ...(withoutComments.match(/'(?:[^'\\\n]|\\.)*'/g) ?? []),
    ...(withoutComments.match(/"(?:[^"\\\n]|\\.)*"/g) ?? []),
  ];
}

const files = sources(SRC).map((path) => ({
  path,
  rel: relative(SRC, path),
  text: readFileSync(path, "utf-8"),
}));

describe("§5.6 — the browser does not infer", () => {
  it("finds source to check", () => {
    // A guard that silently scans nothing is worse than no guard.
    expect(files.length).toBeGreaterThan(15);
    expect(files.some((file) => file.rel === SPEARMAN)).toBe(true);
  });

  it("has no Math.sqrt outside the licensed Spearman port", () => {
    const offenders = files
      .filter((file) => file.rel !== SPEARMAN)
      .filter((file) => /Math\s*\.\s*sqrt/.test(code(file.text)))
      .map((file) => file.rel);

    expect(offenders).toEqual([]);
  });

  it("declares nothing named for a forbidden statistic", () => {
    /*
     * Bindings, not mentions. `\b(const|let|var|function|class)\s+NAME`
     * and `NAME(` as a definition site — reading `cell.p_value` off a
     * validated payload has to stay legal, because the export computes
     * it and §5.6.3 requires it on screen.
     */
    const FORBIDDEN = [
      "zscore",
      "z_score",
      "standarddeviation",
      "stddev",
      "variance",
      "regression",
      "regress",
      "loess",
      "lowess",
      "trendline",
      "slope",
      "percentilerank",
      "percentile_rank",
      "pvalue",
      "p_value",
      "confidenceinterval",
      "covariance",
      "shrinkage",
      "impute",
      "imputation",
      "smoothing",
    ];

    const offenders: string[] = [];

    for (const file of files) {
      if (file.rel === SPEARMAN) continue;
      const body = code(file.text);
      for (const name of FORBIDDEN) {
        const declaration = new RegExp(
          `\\b(?:const|let|var|function|class|interface|type)\\s+\\w*${name}\\w*\\b`,
          "i",
        );
        if (declaration.test(body)) {
          offenders.push(`${file.rel}: declares something named "${name}"`);
        }
      }
    }

    expect(offenders).toEqual([]);
  });

  it("expresses no reduction in SQL", () => {
    /*
     * The other half of §5.6.2's bargain. `query/panel.ts` groups and
     * collects; `query/reduce.ts` reduces. Moving an aggregate into a
     * query string would produce numbers that render on screen and that
     * `reduce.golden.test.ts` never sees.
     *
     * There is no SQL in `src/` at all since §5.16 deviation D10 replaced
     * DuckDB-WASM with a parquet reader, so this currently scans nothing
     * — which is exactly why it is kept. The pressure that would bring a
     * query engine back is the pressure that would bring `avg()` with
     * it, and a guard that only starts working once the mistake is
     * available is still a guard.
     */
    const FORBIDDEN_SQL = [
      /\bavg\s*\(/i,
      /\bmean\s*\(/i,
      /\bsum\s*\(/i,
      /\bmedian\s*\(/i,
      /\bquantile\w*\s*\(/i,
      /\bstddev\w*\s*\(/i,
      /\bvar_(?:pop|samp)\s*\(/i,
      /\bcorr\s*\(/i,
      /\bregr_\w+\s*\(/i,
      /\bpercentile_(?:cont|disc)\s*\(/i,
      /\bapprox_quantile\s*\(/i,
    ];

    const offenders: string[] = [];

    for (const file of files) {
      for (const literal of strings(file.text)) {
        // Only literals that are actually SQL.
        if (!/\bSELECT\b/i.test(literal)) continue;
        for (const pattern of FORBIDDEN_SQL) {
          if (pattern.test(literal)) {
            offenders.push(`${file.rel}: ${pattern} in a SELECT`);
          }
        }
      }
    }

    expect(offenders).toEqual([]);
  });

  it("keeps the Spearman port in one module", () => {
    // §5.6.1 condition 1: "a single module, `src/data/spearman.ts`".
    const offenders = files
      .filter((file) => file.rel !== SPEARMAN)
      .filter((file) => !file.rel.endsWith(".test.ts") && !file.rel.endsWith(".test.tsx"))
      .filter((file) => /function\s+\w*spearman\w*/i.test(code(file.text)))
      .map((file) => file.rel);

    expect(offenders).toEqual([]);
  });
});

describe("§5.6.2 — the closed set stays closed", () => {
  it("names exactly the seven permitted reductions", () => {
    const source = readFileSync(join(SRC, "query", "reduce.ts"), "utf-8");
    const declared = /export type Reduction =([^;]+);/.exec(source)?.[1] ?? "";
    const names = [...declared.matchAll(/"([a-z]+)"/g)].map((match) => match[1]).sort();

    expect(names).toEqual([
      "count",
      "max",
      "mean",
      "median",
      "min",
      "quantile",
      "sum",
    ]);
  });

  it("offers no aggregate in the UI that is not in that set", () => {
    const spec = readFileSync(join(SRC, "encoding", "spec.ts"), "utf-8");
    const declared = /export const AGGREGATES = \[([^\]]+)\]/.exec(spec)?.[1] ?? "";
    const offered = [...declared.matchAll(/"([a-z]+)"/g)].map((match) => match[1]);

    const permitted = ["count", "sum", "mean", "median", "min", "max", "quantile"];
    for (const name of offered) {
      expect(permitted, `${name} is offered in a drop zone`).toContain(name);
    }
  });
});

describe("the guard itself", () => {
  it("would catch a violation", () => {
    // A guard nobody has seen fail is a guard nobody knows works.
    const planted = `const zScore = (x: number) => Math.sqrt(x);`;
    expect(/Math\s*\.\s*sqrt/.test(code(planted))).toBe(true);
    expect(
      /\b(?:const|let|var|function|class|interface|type)\s+\w*zscore\w*\b/i.test(code(planted)),
    ).toBe(true);
  });

  it("does not fire on prose explaining the rule", () => {
    const prose = `/** No z-score here. See regression, slope, p_value. */\nconst x = 1;`;
    expect(
      /\b(?:const|let|var|function)\s+\w*(?:zscore|regression|slope)\w*\b/i.test(code(prose)),
    ).toBe(false);
  });

  it("scans paths relative to src", () => {
    expect(files.every((file) => !file.rel.startsWith(`..${sep}`))).toBe(true);
  });
});
