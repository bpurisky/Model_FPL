/**
 * Counted nouns.
 *
 * Trivial, and it exists because the place these get noticed is the worst
 * possible place: gameweek one. Every surface that says "over the 1
 * gameweeks shown" or "1 marks" says it for the whole first week of the
 * season and then never again — so the bug is invisible in development
 * against three complete seasons of archive, and visible to the only
 * reader on the only week they are watching a season begin.
 *
 * §5.8.7 asks for copy in the model's own vocabulary. That vocabulary is
 * still English.
 */

/** `1 gameweek`, `3 gameweeks`, `0 gameweeks`. */
export function count(n: number, singular: string, plural = `${singular}s`): string {
  return `${n.toLocaleString()} ${n === 1 ? singular : plural}`;
}

/** The noun alone, for callers that format the number themselves. */
export function noun(n: number, singular: string, plural = `${singular}s`): string {
  return n === 1 ? singular : plural;
}
