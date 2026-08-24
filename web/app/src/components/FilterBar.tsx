/**
 * The global filter bar (§5.4.2).
 *
 * > "Position, team, price range, minutes floor, gameweek range. Filters
 * > are app-level state and carry across surfaces (§5.5.3)."
 *
 * Every option comes from `facets()` over the panel rather than from a
 * constant in this file. A hard-coded team list is wrong three times a
 * season on promotion and relegation, and a hard-coded gameweek range is
 * wrong every week.
 *
 * Price is stored in the panel's own units — tenths of a million, so
 * Haaland at £14.0m is 140 — and rendered in pounds. Converting on the
 * way in and out rather than storing pounds keeps the filter and the
 * column comparable without a second unit anywhere in `query/`.
 */

import type { PanelFacets, PanelFilters } from "../query/panel";
import styles from "./FilterBar.module.css";

export interface FilterBarProps {
  facets: PanelFacets;
  filters: PanelFilters;
  onChange: (next: Partial<PanelFilters>) => void;
  normalized: boolean;
  onNormalized: (next: boolean) => void;
  /** Why the toggle currently sits where it does (§5.7.3). */
  normalizedReason: string;
}

export function FilterBar({
  facets,
  filters,
  onChange,
  normalized,
  onNormalized,
  normalizedReason,
}: FilterBarProps) {
  const toggle = (list: string[], value: string): string[] =>
    list.includes(value) ? list.filter((entry) => entry !== value) : [...list, value];

  const pounds = (tenths: number | null): string =>
    tenths === null ? "" : (tenths / 10).toFixed(1);

  const tenths = (value: string): number | null => {
    if (value.trim() === "") return null;
    const parsed = Number(value);
    return Number.isFinite(parsed) ? Math.round(parsed * 10) : null;
  };

  return (
    <div className={styles.bar}>
      <fieldset className={styles.group}>
        <legend className={styles.legend}>Position</legend>
        <div className={styles.chips}>
          {facets.positions.map((position) => (
            <button
              key={position}
              type="button"
              className={styles.chip}
              data-on={filters.positions.includes(position) || undefined}
              aria-pressed={filters.positions.includes(position)}
              onClick={() => onChange({ positions: toggle(filters.positions, position) })}
            >
              {position}
            </button>
          ))}
        </div>
      </fieldset>

      <fieldset className={styles.group}>
        <legend className={styles.legend}>Season</legend>
        <div className={styles.chips}>
          {facets.seasons.map((season) => (
            <button
              key={season}
              type="button"
              className={styles.chip}
              data-on={filters.seasons.includes(season) || undefined}
              aria-pressed={filters.seasons.includes(season)}
              onClick={() => onChange({ seasons: toggle(filters.seasons, season) })}
            >
              {season}
            </button>
          ))}
        </div>
      </fieldset>

      <fieldset className={styles.group}>
        <legend className={styles.legend}>Team</legend>
        <select
          className={styles.select}
          multiple
          size={4}
          value={filters.teams}
          onChange={(event) =>
            onChange({
              teams: [...event.target.selectedOptions].map((option) => option.value),
            })
          }
        >
          {facets.teams.map((team) => (
            <option key={team} value={team}>
              {team}
            </option>
          ))}
        </select>
      </fieldset>

      <fieldset className={styles.group}>
        <legend className={styles.legend}>Price £m</legend>
        <div className={styles.range}>
          <input
            type="number"
            className={styles.number}
            step="0.1"
            min={facets.priceMin / 10}
            max={facets.priceMax / 10}
            placeholder={pounds(facets.priceMin)}
            value={pounds(filters.priceMin)}
            aria-label="Minimum price"
            onChange={(event) => onChange({ priceMin: tenths(event.target.value) })}
          />
          <span className={styles.dash}>–</span>
          <input
            type="number"
            className={styles.number}
            step="0.1"
            min={facets.priceMin / 10}
            max={facets.priceMax / 10}
            placeholder={pounds(facets.priceMax)}
            value={pounds(filters.priceMax)}
            aria-label="Maximum price"
            onChange={(event) => onChange({ priceMax: tenths(event.target.value) })}
          />
        </div>
      </fieldset>

      <fieldset className={styles.group}>
        <legend className={styles.legend}>Gameweek</legend>
        <div className={styles.range}>
          <input
            type="number"
            className={styles.number}
            min={facets.gwMin}
            max={facets.gwMax}
            placeholder={String(facets.gwMin)}
            value={filters.gwMin ?? ""}
            aria-label="First gameweek"
            onChange={(event) =>
              onChange({ gwMin: event.target.value === "" ? null : Number(event.target.value) })
            }
          />
          <span className={styles.dash}>–</span>
          <input
            type="number"
            className={styles.number}
            min={facets.gwMin}
            max={facets.gwMax}
            placeholder={String(facets.gwMax)}
            value={filters.gwMax ?? ""}
            aria-label="Last gameweek"
            onChange={(event) =>
              onChange({ gwMax: event.target.value === "" ? null : Number(event.target.value) })
            }
          />
        </div>
      </fieldset>

      <fieldset className={styles.group}>
        <legend className={styles.legend}>Minutes played</legend>
        <input
          type="number"
          className={styles.number}
          min={0}
          step={90}
          placeholder="any"
          value={filters.minutesFloor ?? ""}
          aria-label="Minimum season-to-date minutes"
          title="Season-to-date minutes, so a rotation week is not dropped from a regular starter's series."
          onChange={(event) =>
            onChange({
              minutesFloor: event.target.value === "" ? null : Number(event.target.value),
            })
          }
        />
      </fieldset>

      <fieldset className={styles.group}>
        <legend className={styles.legend}>Units</legend>
        <label className={styles.toggle} title={normalizedReason}>
          <input
            type="checkbox"
            checked={normalized}
            onChange={(event) => onNormalized(event.target.checked)}
          />
          <span>Within position</span>
        </label>
        {/*
         * §5.7.4: a normalized number renders its basis. The toggle's own
         * label is where the basis belongs when the toggle is what put it
         * there.
         */}
        <p className={styles.basis}>{normalizedReason}</p>
      </fieldset>
    </div>
  );
}
