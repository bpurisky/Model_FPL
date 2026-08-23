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
from web.export.normalize import normalization_basis
from web.export.panel import build_panel, write_panel

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


def cmd_all(args: argparse.Namespace) -> None:
    cmd_columns(args)
    cmd_panel(args)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="web.export")
    parser.add_argument("--out", default=str(OUT_DIR))
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("columns").set_defaults(func=cmd_columns)
    subparsers.add_parser("panel").set_defaults(func=cmd_panel)
    subparsers.add_parser("all").set_defaults(func=cmd_all)
    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
