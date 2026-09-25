/**
 * The app shell: navigation, route switch, and the surfaces that exist.
 *
 * Navigation is two levels. The purple header carries the five sections
 * (Model, Players, Fixtures, My Team, Lab); the strip beneath it carries
 * the pages inside the current section. On a phone the sections move to a
 * bottom tab bar, where a thumb can reach them, and the page strip scrolls
 * sideways instead of wrapping.
 *
 * The Graph Builder is behind `lazy()` and that is a budget decision, not
 * a style one. §5.9 allows the query chunk 1.2 MB and requires it in "no
 * initial-load chunk"; the only thing that actually enforces that is the
 * split point being here, above anything that imports `query/`. A static
 * import of a panel-backed view anywhere in this file would pull the
 * parquet reader into the landing bundle and the budget would erode with
 * no error to notice it by.
 */

import { lazy, Suspense, useEffect, useRef, useState, type ReactNode } from "react";
import { Comparison } from "../views/Comparison";
import { CorrelationLab } from "../views/CorrelationLab";
import { Explorer } from "../views/Explorer";
import { FixtureTicker } from "../views/FixtureTicker";
import { ModelBoard } from "../views/ModelBoard";
import { PaperTradeResults } from "../views/PaperTradeResults";
import { Scorecard } from "../views/Scorecard";
import { SquadOptimizer } from "../views/SquadOptimizer";
import { Planned } from "../views/Planned";
import { AppState, useApp } from "./state";
import { SECTIONS, SURFACES, surfacesIn, type SectionId } from "./surfaces";
import type { View } from "./url";
import styles from "./Shell.module.css";

const GraphBuilder = lazy(() =>
  import("../views/GraphBuilder").then((module) => ({ default: module.GraphBuilder })),
);

const TrendExplorer = lazy(() =>
  import("../views/TrendExplorer").then((module) => ({ default: module.TrendExplorer })),
);

const FormMatrix = lazy(() =>
  import("../views/FormMatrix").then((module) => ({ default: module.FormMatrix })),
);

export function App() {
  return (
    <AppState>
      <Shell />
    </AppState>
  );
}

function Shell() {
  const { state, dispatch } = useApp();
  const current = SURFACES.find((surface) => surface.view === state.view) ?? SURFACES[0]!;
  const pages = surfacesIn(current.section);

  /*
   * Coming back to a section returns you to the page you left it on,
   * rather than always its first page. Remembered for the session only;
   * the URL is still the source of truth for what is on screen.
   */
  const lastInSection = useRef(new Map<SectionId, View>());
  useEffect(() => {
    lastInSection.current.set(current.section, current.view);
  }, [current.section, current.view]);

  const openSection = (section: SectionId) => {
    const view = lastInSection.current.get(section) ?? surfacesIn(section)[0]!.view;
    if (view !== state.view) dispatch({ type: "navigate", view });
  };

  const openPage = (view: View) => {
    if (view !== state.view) dispatch({ type: "navigate", view });
  };

  /*
   * §5.5 makes the URL linkable, and a link people keep is a link they
   * bookmark. A tab title that names the wrong page makes every bookmark
   * wrong about what it points at.
   */
  useEffect(() => {
    document.title = `${current.label} · fpl-trends`;
  }, [current.label]);

  // A new page starts at the top, as a new page on any site does.
  useEffect(() => {
    window.scrollTo({ top: 0 });
  }, [state.view]);

  /*
   * On a phone the page strip is wider than the screen. Scroll the
   * current page's pill into view, or a reader landing on the last page
   * of a section sees it cut off at the edge, or not at all.
   */
  const pagesRef = useRef<HTMLDivElement>(null);
  useEffect(() => {
    const strip = pagesRef.current;
    const active = strip?.querySelector<HTMLElement>('[aria-current="page"]');
    if (!strip || !active) return;
    const target = active.offsetLeft - (strip.clientWidth - active.offsetWidth) / 2;
    strip.scrollTo({ left: Math.max(0, target) });
  }, [state.view]);

  return (
    <div className={styles.shell}>
      <header className={styles.band}>
        <div className={styles.bandInner}>
          <button
            type="button"
            className={styles.wordmark}
            onClick={() => openPage("board")}
            aria-label="fpl-trends, go to the Model Board"
          >
            <span className={styles.crest} aria-hidden="true" />
            fpl<span className={styles.wordmarkDim}>trends</span>
          </button>

          <nav className={styles.sections} aria-label="Sections">
            {SECTIONS.map((section) => {
              const active = section.id === current.section;
              return (
                <button
                  key={section.id}
                  type="button"
                  className={styles.section}
                  data-active={active || undefined}
                  aria-current={active ? "true" : undefined}
                  onClick={() => openSection(section.id)}
                >
                  {section.label}
                </button>
              );
            })}
          </nav>

          <ThemeToggle />
        </div>
      </header>

      {pages.length > 1 && (
        <nav className={styles.pages} aria-label={`${sectionLabel(current.section)} pages`}>
          <div className={styles.pagesInner} ref={pagesRef}>
            {pages.map((surface) => {
              const active = surface.view === state.view;
              const reachable = surface.status !== "out_of_phase";
              return (
                <button
                  key={surface.view}
                  type="button"
                  className={styles.page}
                  data-active={active || undefined}
                  aria-current={active ? "page" : undefined}
                  disabled={!reachable}
                  /*
                   * §5.1.3: a Phase 3/4 entry is disabled and says what
                   * will live there. A disabled control with no
                   * explanation is just a dead end, so the blurb is the
                   * title rather than a tooltip nobody finds.
                   */
                  title={reachable ? surface.blurb : `${surface.milestone}: ${surface.blurb}`}
                  onClick={() => openPage(surface.view)}
                >
                  {surface.label}
                  {surface.status !== "live" && (
                    <span className={styles.badge}>{surface.milestone}</span>
                  )}
                </button>
              );
            })}
          </div>
        </nav>
      )}

      <div className={styles.body}>
        {state.view === "correlations" && <CorrelationLab />}
        {state.view === "graph" && (
          <Suspense fallback={<EngineLoading />}>
            <GraphBuilder />
          </Suspense>
        )}
        {state.view === "form" && (
          <Suspense fallback={<EngineLoading />}>
            <FormMatrix />
          </Suspense>
        )}
        {state.view === "compare" && <Comparison />}
        {state.view === "fixtures" && <FixtureTicker />}
        {state.view === "board" && <ModelBoard />}
        {state.view === "scorecard" && <Scorecard />}
        {state.view === "explorer" && <Explorer />}
        {state.view === "trend" && (
          <Suspense fallback={<EngineLoading />}>
            <TrendExplorer />
          </Suspense>
        )}
        {state.view === "papertrade" && <PaperTradeResults />}
        {state.view === "optimizer" && <SquadOptimizer />}
        {current.status !== "live" && <Planned surface={current} />}
      </div>

      <nav className={styles.tabbar} aria-label="Sections">
        {SECTIONS.map((section) => {
          const active = section.id === current.section;
          return (
            <button
              key={section.id}
              type="button"
              className={styles.tab}
              data-active={active || undefined}
              aria-current={active ? "true" : undefined}
              onClick={() => openSection(section.id)}
            >
              <SectionIcon section={section.id} />
              <span className={styles.tabLabel}>{section.label}</span>
            </button>
          );
        })}
      </nav>
    </div>
  );
}

function sectionLabel(id: SectionId): string {
  return SECTIONS.find((section) => section.id === id)?.label ?? "";
}

type Theme = "light" | "dark";

function systemTheme(): Theme {
  return window.matchMedia?.("(prefers-color-scheme: dark)").matches ? "dark" : "light";
}

/**
 * Light by default, dark by system setting, and a toggle that overrides
 * either. The choice is stamped on <html> as `data-theme`, which is what
 * tokens.css keys its palettes on, and remembered in localStorage.
 * index.html applies a saved choice before first paint.
 */
function ThemeToggle() {
  const [theme, setTheme] = useState<Theme>(() => {
    const stamped = document.documentElement.dataset.theme;
    return stamped === "dark" || stamped === "light" ? stamped : systemTheme();
  });

  const toggle = () => {
    const next: Theme = theme === "dark" ? "light" : "dark";
    setTheme(next);
    document.documentElement.dataset.theme = next;
    try {
      localStorage.setItem("fpl-trends-theme", next);
    } catch {
      // Private windows can refuse storage; the toggle still works for
      // this visit.
    }
  };

  const label = theme === "dark" ? "Switch to light theme" : "Switch to dark theme";
  return (
    <button type="button" className={styles.theme} onClick={toggle} aria-label={label} title={label}>
      {theme === "dark" ? <SunIcon /> : <MoonIcon />}
    </button>
  );
}

function Svg({ children }: { children: ReactNode }) {
  return (
    <svg
      viewBox="0 0 24 24"
      width="22"
      height="22"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.8"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
    >
      {children}
    </svg>
  );
}

function SectionIcon({ section }: { section: SectionId }) {
  switch (section) {
    case "model":
      // A ranking: three bars, tallest first.
      return (
        <Svg>
          <path d="M5 20V9M12 20V4M19 20v-7" />
        </Svg>
      );
    case "players":
      return (
        <Svg>
          <circle cx="12" cy="8" r="3.5" />
          <path d="M5 20c.8-3.6 3.6-5.5 7-5.5s6.2 1.9 7 5.5" />
        </Svg>
      );
    case "fixtures":
      return (
        <Svg>
          <rect x="3.5" y="5" width="17" height="15" rx="2.5" />
          <path d="M3.5 10h17M8 3v4M16 3v4" />
        </Svg>
      );
    case "team":
      // A shirt.
      return (
        <Svg>
          <path d="M8 3.5 4 6l1.5 4H8v10h8V10h2.5L20 6l-4-2.5c-.6 1.6-2.1 2.5-4 2.5s-3.4-.9-4-2.5Z" />
        </Svg>
      );
    case "lab":
      return (
        <Svg>
          <path d="M9 3.5h6M10 3.5v6L4.8 18.2A1.5 1.5 0 0 0 6.1 20.5h11.8a1.5 1.5 0 0 0 1.3-2.3L14 9.5v-6" />
          <path d="M7.5 14.5h9" />
        </Svg>
      );
  }
}

function SunIcon() {
  return (
    <Svg>
      <circle cx="12" cy="12" r="4" />
      <path d="M12 2.5v2M12 19.5v2M4.6 4.6l1.4 1.4M18 18l1.4 1.4M2.5 12h2M19.5 12h2M4.6 19.4 6 18M18 6l1.4-1.4" />
    </Svg>
  );
}

function MoonIcon() {
  return (
    <Svg>
      <path d="M20 14.5A8 8 0 0 1 9.5 4a8 8 0 1 0 10.5 10.5Z" />
    </Svg>
  );
}

/**
 * §5.8.8 forbids skeletons and §5.9 wants the user to know what they are
 * waiting on. This is the chunk, not the data — the data reports its own
 * bytes once the builder mounts — so it says which of the two it is.
 */
function EngineLoading() {
  return (
    <div className={styles.pending}>
      <p className={styles.pendingText}>
        Loading the player-gameweek reader. It is fetched only on the pages that need it.
      </p>
    </div>
  );
}
