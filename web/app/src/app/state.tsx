/**
 * App state (§5.2's "React context + `useReducer`", §5.5.3).
 *
 * > "App state is *one dataset + current selection + current encoding*.
 * > Redux/Zustand would be ceremony."
 *
 * And §5.5.3 draws the line this module enforces: **player selection, the
 * position filter and the raw/normalized toggle are app-level and carry
 * across every surface; encoding state is view-level.** Encoding still
 * lives in this reducer rather than in the Graph Builder's own `useState`
 * for one reason — §5.5 requires it in the URL, and there is exactly one
 * URL. Keeping it here means one writer, so a channel change and a
 * position change cannot race each other into history.
 *
 * The URL is the single source of truth for everything linkable. State
 * changes `replaceState` by default rather than pushing: dragging a
 * column through four zones should not cost four presses of the back
 * button to undo. Navigation between surfaces pushes, because that is
 * the step a reader means to walk back.
 */

import {
  createContext,
  useContext,
  useEffect,
  useMemo,
  useReducer,
  type Dispatch,
  type ReactNode,
} from "react";
import type { Encoding, Channel } from "../encoding/spec";
import type { PanelFilters } from "../query/panel";
import { DEFAULT_STATE, parseUrl, toSearch, type AppUrlState, type View } from "./url";

export type Action =
  | { type: "navigate"; view: View }
  | { type: "position"; position: string }
  | { type: "normalized"; normalized: boolean }
  | { type: "filters"; filters: Partial<PanelFilters> }
  | { type: "encode"; channel: Channel; key: string | null }
  | { type: "encoding"; encoding: Partial<Encoding> }
  | { type: "select"; ids: number[] }
  | { type: "toggleSelect"; id: number }
  | { type: "entry"; entryId: number | null }
  | { type: "restore"; state: AppUrlState };

/**
 * Whether an action is a step a reader would expect the back button to
 * undo. Only navigation qualifies — see the module docstring.
 */
function pushes(action: Action): boolean {
  return action.type === "navigate";
}

export function reducer(state: AppUrlState, action: Action): AppUrlState {
  switch (action.type) {
    case "navigate":
      return { ...state, view: action.view };

    case "position":
      return { ...state, position: action.position };

    case "normalized":
      return { ...state, normalized: action.normalized };

    case "filters":
      return { ...state, filters: { ...state.filters, ...action.filters } };

    case "encode": {
      /*
       * Assigning a column to a zone removes it from whichever zone it
       * was in. Without this a user who drags X onto Y gets the same
       * metric on both axes and a perfect diagonal, which reads as a
       * finding for about two seconds.
       */
      const encoding: Encoding = { ...state.encoding };
      if (action.key !== null) {
        for (const channel of ["x", "y", "color", "wrap"] as const) {
          if (encoding[channel] === action.key) encoding[channel] = null;
        }
      }
      encoding[action.channel] = action.key;
      return { ...state, encoding };
    }

    case "encoding":
      return { ...state, encoding: { ...state.encoding, ...action.encoding } };

    case "select":
      return { ...state, selection: action.ids };

    case "toggleSelect": {
      const has = state.selection.includes(action.id);
      return {
        ...state,
        selection: has
          ? state.selection.filter((id) => id !== action.id)
          : [...state.selection, action.id],
      };
    }

    case "entry":
      return { ...state, entry: action.entryId };

    case "restore":
      return action.state;
  }
}

interface Store {
  state: AppUrlState;
  dispatch: Dispatch<Action>;
}

const StoreContext = createContext<Store | null>(null);

export function AppState({ children }: { children: ReactNode }) {
  const [state, dispatch] = useReducer(
    historyReducer,
    undefined,
    () => parseUrl(window.location.search),
  );

  /*
   * The back button. `popstate` fires with the URL already changed, so
   * the store follows it rather than the other way round — restoring
   * from the URL is the only way a linked state and a walked-back state
   * end up identical.
   */
  useEffect(() => {
    const onPop = () => dispatch({ type: "restore", state: parseUrl(window.location.search) });
    window.addEventListener("popstate", onPop);
    return () => window.removeEventListener("popstate", onPop);
  }, []);

  const store = useMemo(() => ({ state, dispatch }), [state]);
  return <StoreContext.Provider value={store}>{children}</StoreContext.Provider>;
}

/**
 * The reducer, plus the history write.
 *
 * Done here rather than in an effect so the URL moves in the same tick as
 * the state. An effect would let a render happen with the two disagreeing,
 * which is the window in which a copied link points at the previous view.
 */
function historyReducer(state: AppUrlState, action: Action): AppUrlState {
  const next = reducer(state, action);
  if (action.type !== "restore" && typeof window !== "undefined") {
    const search = toSearch(next);
    const url = `${window.location.pathname}${search}`;
    if (url !== `${window.location.pathname}${window.location.search}`) {
      if (pushes(action)) window.history.pushState(null, "", url);
      else window.history.replaceState(null, "", url);
    }
  }
  return next;
}

export function useApp(): Store {
  const store = useContext(StoreContext);
  if (!store) throw new Error("useApp outside AppState");
  return store;
}

export { DEFAULT_STATE };
export type { AppUrlState, View };
