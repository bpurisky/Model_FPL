"""CLI entrypoint: `uv run python -m analytics evaluate`, and
`uv run python -m analytics fit-price-thresholds` (see
`analytics/price_model.py`'s module docstring for when to run it).

The single command §4.4 / §8 need: regenerates the full model-vs-baselines
comparison, the per-component decomposition, and the minutes-head scorecard
from the committed data/historical/*.parquet and raw fixtures/teams cache.
"""

from __future__ import annotations

import argparse
import logging
from datetime import date, datetime, timezone
from pathlib import Path

import polars as pl
import yaml

from analytics.evaluate import run_evaluation
from analytics.price_model import (
    PRICE_MODEL_CONFIG,
    RECENT_MOVE_WINDOW,
    cycle_observations,
    daily_cycles,
    evaluate_observations,
    fit_rise_threshold,
    load_distilled_history,
)
from backtest.report import build_report, component_decomposition_mae, minutes_head_metrics, write_report

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("analytics.main")


def cmd_evaluate(args: argparse.Namespace) -> None:
    results, decomposition = run_evaluation()
    report = build_report(results)

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


# Fit on this share of the available cycles by default; the rest, and
# every cycle completed afterwards, is what the launch gate scores.
DEFAULT_FIT_SHARE = 0.7


def cmd_fit_price_thresholds(args: argparse.Namespace) -> None:
    history = load_distilled_history(Path(args.distilled_dir))
    deadlines = pl.read_parquet(args.events)["deadline_time"].drop_nulls().to_list()
    cycles = daily_cycles(history["snapshot_ts"].unique().to_list(), deadlines)
    if not cycles:
        raise SystemExit("no complete price-change cycles in the distilled history")

    cutoff: date = date.fromisoformat(args.cutoff) if args.cutoff else cycles[int(len(cycles) * DEFAULT_FIT_SHARE)].prediction_ts.date()
    cutoff_ts = datetime.combine(cutoff, datetime.min.time(), tzinfo=timezone.utc)
    fit_cycles = [c for c in cycles if c.prediction_ts < cutoff_ts]
    held_out = [c for c in cycles if c.prediction_ts >= cutoff_ts]
    if not fit_cycles:
        raise SystemExit(f"no cycles before cutoff {cutoff}")

    fit_observations = cycle_observations(history, fit_cycles)
    fitted = fit_rise_threshold(fit_observations)
    in_sample = evaluate_observations(fit_observations, fitted["rise_threshold"])
    config = {
        "rise_threshold": round(float(fitted["rise_threshold"]), 4),
        "recent_move_hours": int(RECENT_MOVE_WINDOW.total_seconds() // 3600),
        "score": "net_transfers_24h",
        "fit_cutoff": cutoff,
        "fit_span": f"{fit_cycles[0].prediction_ts.date()} to {fit_cycles[-1].prediction_ts.date()}",
        "fit_cycles": len(fit_cycles),
        "fit_rise_f1": round(float(fitted["rise_f1"]), 4),
        "fit_rise_n_called": int(fitted["rise_n_called"]),
        "fit_in_sample_hit_rate": round(in_sample.hit_rate, 4) if in_sample.hit_rate is not None else None,
        "fitted_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    header = (
        "# Written by `uv run python -m analytics fit-price-thresholds` -- do not hand-edit.\n"
        "# The launch gate scores only cycles on or after fit_cutoff (see analytics/price_model.py).\n"
    )
    Path(args.out).write_text(header + yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    logger.info("fitted on %d cycle(s) before %s: %s", len(fit_cycles), cutoff, config)
    logger.info("in-sample: %s", in_sample)
    if held_out:
        logger.info(
            "out-of-sample so far (%d cycle(s), informational -- the gate recomputes this every run): %s",
            len(held_out),
            evaluate_observations(cycle_observations(history, held_out), fitted["rise_threshold"]),
        )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m analytics")
    subparsers = parser.add_subparsers(dest="command", required=True)

    evaluate_parser = subparsers.add_parser("evaluate")
    evaluate_parser.add_argument("--out", default="analytics/report.json")
    evaluate_parser.set_defaults(func=cmd_evaluate)

    fit_parser = subparsers.add_parser("fit-price-thresholds")
    fit_parser.add_argument("--distilled-dir", default="data/distilled")
    fit_parser.add_argument("--events", default="data/reference/events.parquet")
    fit_parser.add_argument("--cutoff", help="ISO date; fit on cycles before it (default: first 70%% of cycles)")
    fit_parser.add_argument("--out", default=str(PRICE_MODEL_CONFIG))
    fit_parser.set_defaults(func=cmd_fit_price_thresholds)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
