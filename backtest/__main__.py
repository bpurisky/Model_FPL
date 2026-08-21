"""CLI entrypoint: `uv run python -m backtest <command>`.

`run` is the single command §3.6 / §8 require: it regenerates the full
walk-forward report from the committed data/historical/*.parquet files with
no network access.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import polars as pl

from backtest.backfill import NORMALIZED_DIR, SEASONS, backfill_all
from backtest.harness import walk_forward_all_seasons
from backtest.report import build_report, write_report

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("backtest.main")


def cmd_backfill(_: argparse.Namespace) -> None:
    paths = backfill_all()
    logger.info("backfilled %d seasons: %s", len(paths), [str(p) for p in paths])


def cmd_run(args: argparse.Namespace) -> None:
    season_dfs = {}
    for season in SEASONS:
        path = NORMALIZED_DIR / f"{season}.parquet"
        if not path.exists():
            logger.error("%s missing; run `python -m backtest backfill` first", path)
            raise SystemExit(1)
        season_dfs[season] = pl.read_parquet(path)

    results = walk_forward_all_seasons(season_dfs)
    report = build_report(results)
    write_report(report, Path(args.out))
    logger.info("walk-forward complete: %d predictions -> %s", report.get("n_rows", 0), args.out)
    for baseline, metrics in report.get("pooled_baseline", {}).items():
        logger.info("pooled %-32s MAE=%.3f RMSE=%.3f spearman=%.3f", baseline, metrics["mae"], metrics["rmse"], metrics["spearman_within_position"]["mean"])


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m backtest")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("backfill").set_defaults(func=cmd_backfill)

    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--out", default="backtest/report.json")
    run_parser.set_defaults(func=cmd_run)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
