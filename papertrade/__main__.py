"""CLI entrypoint: `uv run python -m papertrade <command>`.

- `freeze`: §6.1 -- freeze the shadow team's recommendation for the next
  gameweek. Run this once, before each deadline.
- `record-actuals [--gw N]`: append gw N's real results to the permanent
  actuals record. Omit --gw to auto-detect and catch up on every finished,
  unrecorded gameweek in one run — the mode the weekly automation uses.
- `evaluate`: §6.3-6.5 -- player-level and squad-level evaluation, plus the
  launch gate report, over every gameweek that has both a freeze and
  recorded actuals.

Kept thin like analytics/backtest/collector/squad's own __main__.py — no
logic beyond argument parsing, wiring, and formatting.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
from pathlib import Path

from collector.config import load_config
from papertrade.actuals import append_actuals, fetch_gw_actuals, fetch_missing_gw_actuals, load_actuals
from papertrade.evaluate import (
    evaluate_gw_player_level,
    evaluate_squad_level,
    fetch_real_gw_points,
    launch_gate_report,
    run_price_model_for_gate,
)
from papertrade.freeze import FREEZES_DIR, latest_frozen_gw, run_freeze

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger("papertrade.main")


async def cmd_freeze(args: argparse.Namespace) -> None:
    cfg = load_config(Path(args.config))
    entry_id = args.entry_id or cfg.own_entry_id
    if entry_id is None:
        raise SystemExit("no --entry-id given and config/collector.yaml's own_entry_id is unset")
    try:
        path = await run_freeze(cfg, entry_id, gw=args.gw)
    except FileExistsError as exc:
        # Expected steady state for a daily automated run: the current gw is
        # already frozen most days, and that's not a failure -- only the day
        # the next gw becomes freezable does this actually write something.
        logger.info("nothing new to freeze: %s", exc)
        return
    logger.info("froze gw%s -> %s", args.gw or "(next)", path)


async def cmd_record_actuals(args: argparse.Namespace) -> None:
    cfg = load_config(Path(args.config))
    if args.gw is not None:
        df = await fetch_gw_actuals(cfg, args.gw)
        path = append_actuals(df)
        logger.info("recorded %d players' actuals for gw%d -> %s", df.height, args.gw, path)
        return

    # auto-detect: catch up on every finished gameweek not yet recorded, so
    # the weekly automation doesn't need to know which gw to ask for.
    frames = await fetch_missing_gw_actuals(cfg)
    if not frames:
        logger.info("no newly-finished gameweeks to record")
        return
    for df in frames:
        gw = df["gw"][0]
        path = append_actuals(df)
        logger.info("recorded %d players' actuals for gw%d -> %s", df.height, gw, path)


async def cmd_evaluate(args: argparse.Namespace) -> None:
    cfg = load_config(Path(args.config))
    entry_id = args.entry_id or cfg.own_entry_id
    if entry_id is None:
        raise SystemExit("no --entry-id given and config/collector.yaml's own_entry_id is unset")

    actuals = load_actuals()
    evaluated_gws = sorted(set(actuals["gw"].to_list()) if actuals.height else [])
    latest_gw = latest_frozen_gw()
    candidate_gws = [gw for gw in evaluated_gws if gw <= (latest_gw or 0)]

    player_eval_by_gw = {}
    for gw in candidate_gws:
        try:
            player_eval_by_gw[gw] = evaluate_gw_player_level(gw, actuals=actuals)
        except (FileNotFoundError, ValueError) as exc:
            logger.info("skipping gw%d player-level eval: %s", gw, exc)

    real_points_by_gw = await fetch_real_gw_points(cfg, entry_id)
    squad_eval = evaluate_squad_level(real_points_by_gw, actuals=actuals)
    price_eval = run_price_model_for_gate()
    gate = launch_gate_report(player_eval_by_gw, squad_eval, price_eval=price_eval)

    report = {
        "player_level_by_gw": player_eval_by_gw,
        "squad_level": squad_eval,
        "launch_gate": gate,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

    print(f"Evaluated {len(player_eval_by_gw)} gameweek(s) at player level: {sorted(player_eval_by_gw)}")
    for gw, ev in sorted(player_eval_by_gw.items()):
        print(f"  gw{gw}: n={ev['n']} MAE={ev['mae']:.3f} spearman(mean)={ev['spearman_within_position'].get('mean', float('nan')):.3f}")
    print(f"\nSquad level ({squad_eval['n_gameweeks']} gameweek(s)): real={squad_eval['cumulative_real_points']} "
          f"shadow={squad_eval['cumulative_shadow_points']} (shadow - real = {squad_eval['shadow_minus_real']:+d})")
    print(f"  {squad_eval['warning']}")
    print(f"\nLaunch gate: {'READY' if gate['ready_to_launch'] else 'NOT READY'} ({gate['gameweeks_evaluated']}/13 gameweeks)")
    for name, c in gate["criteria"].items():
        print(f"  [{c['status']}] {name}: {c['detail']}")
    print(f"\nFull report written to {args.out}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m papertrade")
    subparsers = parser.add_subparsers(dest="command", required=True)

    freeze_parser = subparsers.add_parser("freeze")
    freeze_parser.add_argument("--config", default="config/collector.yaml")
    freeze_parser.add_argument("--entry-id", type=int, default=None)
    freeze_parser.add_argument("--gw", type=int, default=None)
    freeze_parser.set_defaults(func=cmd_freeze)

    actuals_parser = subparsers.add_parser("record-actuals")
    actuals_parser.add_argument("--config", default="config/collector.yaml")
    actuals_parser.add_argument("--gw", type=int, default=None, help="omit to auto-detect every finished, unrecorded gameweek")
    actuals_parser.set_defaults(func=cmd_record_actuals)

    evaluate_parser = subparsers.add_parser("evaluate")
    evaluate_parser.add_argument("--config", default="config/collector.yaml")
    evaluate_parser.add_argument("--entry-id", type=int, default=None)
    evaluate_parser.add_argument("--out", default="papertrade/report.json")
    evaluate_parser.set_defaults(func=cmd_evaluate)

    args = parser.parse_args(argv)
    asyncio.run(args.func(args))


if __name__ == "__main__":
    main()
