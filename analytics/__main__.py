"""CLI entrypoint: `uv run python -m analytics evaluate`.

The single command §4.4 / §8 need: regenerates the full model-vs-baselines
comparison, the per-component decomposition, and the minutes-head scorecard
from the committed data/historical/*.parquet and raw fixtures/teams cache.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from analytics.evaluate import run_comparison, run_component_decomposition
from backtest.report import build_report, component_decomposition_mae, minutes_head_metrics, write_report

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("analytics.main")


def cmd_evaluate(args: argparse.Namespace) -> None:
    results = run_comparison()
    report = build_report(results)

    decomposition = run_component_decomposition()
    report["component_decomposition"] = component_decomposition_mae(
        decomposition["predicted_components"], decomposition["actual_components"]
    )
    report["minutes_head"] = minutes_head_metrics(decomposition["predicted_minutes_dist"], decomposition["actual_minutes"])

    write_report(report, Path(args.out))
    logger.info("evaluation complete: %d predictions -> %s", report.get("n_rows", 0), args.out)
    for baseline, metrics in report.get("pooled_baseline", {}).items():
        logger.info(
            "pooled %-32s MAE=%.4f RMSE=%.4f spearman=%.4f",
            baseline, metrics["mae"], metrics["rmse"], metrics["spearman_within_position"]["mean"],
        )
    logger.info("component decomposition (MAE per bucket): %s", report["component_decomposition"])
    logger.info("minutes head: %s", report["minutes_head"])


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m analytics")
    subparsers = parser.add_subparsers(dest="command", required=True)

    evaluate_parser = subparsers.add_parser("evaluate")
    evaluate_parser.add_argument("--out", default="analytics/report.json")
    evaluate_parser.set_defaults(func=cmd_evaluate)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
