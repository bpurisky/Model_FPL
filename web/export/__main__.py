"""`uv run python -m web.export` — the 5A export CLI (§5.2.1).

Output goes to `data/web/v1/`. Which of those files are committed and
which are build artifacts is §5.3.4's split: `columns.json` is committed,
so a fresh clone renders the surfaces that depend on it with no pipeline
run at all.

Subcommands are added a file at a time as 5A lands. `all` runs every
exporter that is currently wired, so it stays honest about what exists
rather than pretending to a full contract.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from web.export.columns import REGISTRY
from web.export.contract import ColumnsFile, build_header
from web.export.correlations import build_correlations
from web.export.fixtures import build_fixtures
from web.export.golden import build_golden_spearman
from web.export.normalize import normalization_basis
from web.export.panel import build_panel, write_panel
from web.export.scorecard import build_scorecard

logger = logging.getLogger("web.export")

OUT_DIR = Path("data/web/v1")


def write_json(payload: str, name: str, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / name
    path.write_text(payload, encoding="utf-8")
    return path


def cmd_columns(args: argparse.Namespace) -> None:
    """The registry (§5.3.5), validated through the contract on the way
    out so a malformed entry cannot reach disk."""
    file = ColumnsFile(
        header=build_header(
            rows=len(REGISTRY),
            source_gameweek=None,  # the registry describes columns, not a gameweek
            normalization_basis=normalization_basis(),
        ),
        columns=REGISTRY,
    )
    path = write_json(file.model_dump_json(indent=2), "columns.json", Path(args.out))
    logger.info("wrote %d column entries -> %s", len(REGISTRY), path)


def cmd_panel(args: argparse.Namespace) -> None:
    """The tidy long table (§5.3.2). A build artifact, not committed."""
    df = build_panel()
    path = write_panel(df, Path(args.out))
    size_mb = path.stat().st_size / 1_048_576
    logger.info(
        "wrote panel: %d rows x %d cols, %.1f MB -> %s", df.height, df.width, size_mb, path
    )


def cmd_correlations(args: argparse.Namespace) -> None:
    """The Correlation Lab matrices (§5.4.1). Committed per §5.3.4, so a
    fresh clone renders the hero surface with no pipeline run — but built
    from `panel.parquet`, which is not committed, so this runs after
    `panel` rather than standing alone."""
    file = build_correlations(panel_path=Path(args.out) / "panel.parquet")
    path = write_json(file.model_dump_json(indent=2), "correlations.json", Path(args.out))
    hatched = sum(1 for c in file.cells if c.n < file.min_n_cell)
    logger.info(
        "wrote %d cells over %d metrics x %d groups (%d below n=%d) -> %s",
        len(file.cells), len(file.metrics), len(file.groups), hatched, file.min_n_cell, path,
    )


def cmd_scorecard(args: argparse.Namespace) -> None:
    """The backtest report made legible (§5.4.7). Committed per §5.3.4.

    Independent of `panel.parquet`: it rebuilds the walk-forward from
    `data/historical/`, which is committed, so this one stands alone.
    """
    file = build_scorecard()
    path = write_json(file.model_dump_json(indent=2), "scorecard.json", Path(args.out))
    logger.info(
        "wrote %d rows over %d models x %d seasons -> %s",
        len(file.rows), len(file.models), len(file.seasons), path,
    )


def cmd_fixtures(args: argparse.Namespace) -> None:
    """Elo difficulty beside FPL's static rating (§4.3). Committed per
    §5.3.4, and independent of `panel.parquet`."""
    file = build_fixtures()
    path = write_json(file.model_dump_json(indent=2), "fixtures.json", Path(args.out))
    played = sum(1 for f in file.fixtures if f.played)
    logger.info(
        "wrote %d fixtures (%d played) from %d elo matches seeded by %s -> %s",
        len(file.fixtures), played, file.elo_matches, file.elo_seeded_from or "nothing", path,
    )


def cmd_golden(args: argparse.Namespace) -> None:
    """The Spearman port's CI fixtures (§5.6.1). Committed per §5.3.4, and
    self-contained: it embeds the values its answers were computed over,
    because the TypeScript side cannot read the gitignored panel."""
    file = build_golden_spearman(panel_path=Path(args.out) / "panel.parquet")
    path = write_json(file.model_dump_json(indent=2), "golden_spearman.json", Path(args.out))
    computable = sum(1 for p in file.pairs if p.rho is not None)
    logger.info(
        "wrote %d golden pairs (%d computable) over %d samples -> %s",
        len(file.pairs), computable, len(file.samples), path,
    )


def cmd_all(args: argparse.Namespace) -> None:
    cmd_columns(args)
    cmd_panel(args)
    cmd_correlations(args)
    cmd_scorecard(args)
    cmd_fixtures(args)
    cmd_golden(args)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="web.export")
    parser.add_argument("--out", default=str(OUT_DIR))
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("columns").set_defaults(func=cmd_columns)
    subparsers.add_parser("panel").set_defaults(func=cmd_panel)
    subparsers.add_parser("correlations").set_defaults(func=cmd_correlations)
    subparsers.add_parser("scorecard").set_defaults(func=cmd_scorecard)
    subparsers.add_parser("fixtures").set_defaults(func=cmd_fixtures)
    subparsers.add_parser("golden").set_defaults(func=cmd_golden)
    subparsers.add_parser("all").set_defaults(func=cmd_all)
    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
