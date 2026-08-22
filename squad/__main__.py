"""CLI entrypoint: `uv run python -m squad recommend --entry-id <id>`.

Wires squad/live.py's live-data bridge to squad/optimize.py and prints a
plain-text recommendation report. Kept thin like analytics/__main__.py and
collector/__main__.py — no logic beyond argument parsing and formatting.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
from pathlib import Path

from collector.config import load_config
from squad.live import build_projections, fetch_live_data
from squad.optimize import OptimizationResult, optimize_squad, pair_transfers_by_position, template_risk_flags

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logging.getLogger("httpx").setLevel(logging.WARNING)  # one INFO line per request is noise, not signal, here
logger = logging.getLogger("squad.main")


def _price(now_cost: int) -> str:
    return f"£{now_cost / 10:.1f}m"


def _name(live, element_id: int) -> str:
    return live.web_names.get(element_id, f"#{element_id}")


def format_report(live, result: OptimizationResult, horizon: list[int]) -> str:
    lines: list[str] = []
    lines.append("=" * 70)
    lines.append("FPL SQUAD RECOMMENDATION")
    lines.append("=" * 70)
    lines.append("")
    lines.append(
        f"CAVEAT: gameweek {live.data_gw} of 2026/27 is this season's only "
        f"data so far, and only {live.teams_with_finished_data}/{live.teams_total} "
        "teams have a *finished* fixture in it as of this run -- an "
        "in-progress match's cumulative stats (e.g. 0 minutes because a "
        "fixture hasn't kicked off yet, not because of a blank) are "
        "deliberately excluded, so any team not yet counted here is "
        "projected from the position-level pooled average, not its own "
        "form. Every projection is otherwise a single-match trailing rate "
        "plus custom fixture difficulty (Elo carried over from 2025/26) -- "
        "not the validated model from Phase 2's backtest, which needed "
        "several seasons of data to clear its baselines. Treat this as a "
        "provisional, early-season read, not a settled recommendation "
        "(see squad/live.py's module docstring)."
    )
    lines.append("")

    pool_by_id = {p.element_id: p for p in live.pool}
    current_ids = {sp.element_id for sp in live.squad.players}

    lines.append(f"Free transfers available: {live.free_transfers}")
    lines.append(f"Bank: {_price(live.squad.bank)}")
    lines.append(f"Horizon: gameweeks {horizon}")
    lines.append("")

    if result.transfers_out or result.transfers_in:
        lines.append("-- Recommended transfers --")
        for out_id, in_id in pair_transfers_by_position(result.transfers_out, result.transfers_in, pool_by_id):
            out_p, in_p = pool_by_id[out_id], pool_by_id[in_id]
            lines.append(
                f"  OUT: {_name(live, out_id)} ({out_p.position}, {out_p.club}) "
                f"-> IN: {_name(live, in_id)} ({in_p.position}, {in_p.club}, {_price(in_p.now_cost)})"
            )
        if result.hits_taken:
            lines.append(f"  Hits taken: {result.hits_taken} (-{result.hits_taken * 4} pts)")
        lines.append(f"  Bank after: {_price(result.bank_after)}")
    else:
        lines.append("-- No transfer recommended: hold the squad. --")
    lines.append("")

    flags = template_risk_flags(result.transfers_out, live.ownership)
    if flags:
        lines.append("-- Template risk --")
        for eid, msg in flags.items():
            lines.append(f"  {_name(live, eid)}: {msg}")
        lines.append("")

    next_gw = horizon[0]
    xi = result.starting_xi[next_gw]
    captain_id = result.captain[next_gw]
    lines.append(f"-- Starting XI, gameweek {next_gw} --")
    for pos in ["GK", "DEF", "MID", "FWD"]:
        names_tagged = [
            f"{_name(live, eid)}{' (C)' if eid == captain_id else ''}"
            for eid in xi if pool_by_id[eid].position == pos
        ]
        if names_tagged:
            lines.append(f"  {pos}: {', '.join(names_tagged)}")
    lines.append("")
    lines.append(f"-- Bench order, gameweek {next_gw} (best-first) --")
    for eid in result.bench_order:
        lines.append(f"  {_name(live, eid)} ({pool_by_id[eid].position})")
    lines.append("")
    lines.append(f"Squad size: {len(result.squad)}, all currently-owned players not sold: {len(current_ids & result.squad)}")

    return "\n".join(lines)


async def cmd_recommend(args: argparse.Namespace) -> None:
    cfg = load_config(Path(args.config))
    entry_id = args.entry_id or cfg.own_entry_id
    if entry_id is None:
        raise SystemExit("no --entry-id given and config/collector.yaml's own_entry_id is unset")

    horizon = [int(x) for x in args.horizon.split(",")] if args.horizon else None
    live = await fetch_live_data(cfg, entry_id, horizon=horizon)
    horizon = horizon or [live.next_event, live.next_event + 1, live.next_event + 2]

    projections = build_projections(live.train_df, live.target_roster, live.scoring_config, live.difficulty_table, horizon)
    result = optimize_squad(
        live.squad, live.pool, projections, horizon=horizon, free_transfers=live.free_transfers,
        max_transfers=args.max_transfers, hit_cost=args.hit_cost,
    )
    print(format_report(live, result, horizon))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m squad")
    subparsers = parser.add_subparsers(dest="command", required=True)

    recommend_parser = subparsers.add_parser("recommend")
    recommend_parser.add_argument("--config", default="config/collector.yaml")
    recommend_parser.add_argument("--entry-id", type=int, default=None)
    recommend_parser.add_argument("--horizon", default=None, help="comma-separated gameweeks, e.g. 2,3,4")
    recommend_parser.add_argument("--max-transfers", type=int, default=None)
    recommend_parser.add_argument("--hit-cost", type=int, default=4)
    recommend_parser.set_defaults(func=cmd_recommend)

    args = parser.parse_args(argv)
    asyncio.run(args.func(args))


if __name__ == "__main__":
    main()
