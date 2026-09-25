import { useEffect, useState } from "react";

/**
 * The phone breakpoint, in one place. Must match the `max-width: 760px`
 * media queries in tokens.css and Shell.module.css, which are where the
 * bottom tab bar appears.
 */
export const NARROW_QUERY = "(max-width: 760px)";

/**
 * True on phone-width screens, and tracks rotation and resizing. For the
 * few places where layout alone is not enough and a component has to
 * behave differently (a filter bar that starts collapsed, say). Styling
 * should stay in CSS media queries.
 */
export function useNarrow(): boolean {
  const [narrow, setNarrow] = useState(() => window.matchMedia?.(NARROW_QUERY).matches ?? false);

  useEffect(() => {
    const media = window.matchMedia?.(NARROW_QUERY);
    if (!media) return;
    const update = () => setNarrow(media.matches);
    update();
    media.addEventListener("change", update);
    return () => media.removeEventListener("change", update);
  }, []);

  return narrow;
}
