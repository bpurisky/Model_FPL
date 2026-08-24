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

from web.export.board import build_board
from web.export.columns import REGISTRY
from web.export.contract import ColumnsFile, build_header
from web.export.correlations import build_correlations
from web.export.fixtures import build_fixtures
from web.export.golden import build_golden_spearman
from web.export.normalize import normalization_basis
from web.export.observations import build_observations
from web.export.panel import build_panel, write_panel
from web.export.players import build_players
from web.export.timeseries import build_timeseries, write_timeseries
from web.export.scorecard import build_scorecard

logger = logging.getLogger("web.export")

OUT_DIR = Path("data/web/v1")


# Every header carries these and both move on every run regardless of what
# the numbers did (§5.3.1). Excluded when deciding whether a file changed.
_VOLATILE_HEADER = ("generated_at", "model_git_sha")


def _body(payload: str) -> str:
    """A file's content with the volatile header fields removed."""
    data = json.loads(payload)
    header = data.get("header")
    if isinstance(header, dict):
        data["header"] = {k: v for k, v in header.items() if k not in _VOLATILE_HEADER}
    return json.dumps(data, sort_keys=True)


def write_json(payload: str, name: str, out_dir: Path) -> tuple[Path, bool]:
    """Write only if the numbers actually changed.

    These files are committed (§5.3.4) and the deploy workflow regenerates
    them whenever the collector runs. `generated_at` and `model_git_sha`
    move every single run, so an unconditional write would commit all five
    files every hour and bury a real change in a year of noise.

    Leaving the older sha in place when the body is unchanged is the more
    truthful record, not a compromise: §5.3.1 wants every number traceable
    to the code that produced it, and if the numbers did not move then the
    earlier commit is the one that produced them.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / name
    if path.exists() and _body(path.read_text(encoding="utf-8")) == _body(payload):
        return path, False
    path.write_text(payload, encoding="utf-8")
    return path, True


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
    path, changed = write_json(file.model_dump_json(indent=2), "columns.json", Path(args.out))
    logger.info("%s %d column entries -> %s", "wrote" if changed else "unchanged:", len(REGISTRY), path)


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
    path, changed = write_json(file.model_dump_json(indent=2), "correlations.json", Path(args.out))
    hatched = sum(1 for c in file.cells if c.n < file.min_n_cell)
    logger.info(
        "%s %d cells over %d metrics x %d groups (%d below n=%d) -> %s",
        "wrote" if changed else "unchanged:", len(file.cells), len(file.metrics), len(file.groups), hatched, file.min_n_cell, path,
    )


def cmd_scorecard(args: argparse.Namespace) -> None:
    """The backtest report made legible (§5.4.7). Committed per §5.3.4.

    Independent of `panel.parquet`: it rebuilds the walk-forward from
    `data/historical/`, which is committed, so this one stands alone.
    """
    file = build_scorecard()
    path, changed = write_json(file.model_dump_json(indent=2), "scorecard.json", Path(args.out))
    logger.info(
        "%s %d rows over %d models x %d seasons -> %s",
        "wrote" if changed else "unchanged:", len(file.rows), len(file.models), len(file.seasons), path,
    )


def cmd_fixtures(args: argparse.Namespace) -> None:
    """Elo difficulty beside FPL's static rating (§4.3). Committed per
    §5.3.4, and independent of `panel.parquet`."""
    file = build_fixtures()
    path, changed = write_json(file.model_dump_json(indent=2), "fixtures.json", Path(args.out))
    played = sum(1 for f in file.fixtures if f.played)
    logger.info(
        "%s %d fixtures (%d played) from %d elo matches seeded by %s -> %s",
        "wrote" if changed else "unchanged:", len(file.fixtures), played, file.elo_matches, file.elo_seeded_from or "nothing", path,
    )


def cmd_golden(args: argparse.Namespace) -> None:
    """The Spearman port's CI fixtures (§5.6.1). Committed per §5.3.4, and
    self-contained: it embeds the values its answers were computed over,
    because the TypeScript side cannot read the gitignored panel."""
    file = build_golden_spearman(panel_path=Path(args.out) / "panel.parquet")
    path, changed = write_json(file.model_dump_json(indent=2), "golden_spearman.json", Path(args.out))
    computable = sum(1 for p in file.pairs if p.rho is not None)
    logger.info(
        "%s %d golden pairs (%d computable) over %d samples -> %s",
        "wrote" if changed else "unchanged:", len(file.pairs), computable, len(file.samples), path,
    )


def cmd_timeseries(args: argparse.Namespace) -> None:
    """Per-player market history (§5.4.8). A build artifact like the
    panel, and gitignored for the same reason."""
    df = build_timeseries()
    path = write_timeseries(df, Path(args.out))
    size_mb = path.stat().st_size / 1_048_576
    logger.info(
        "wrote timeseries: %d rows x %d cols over %d snapshot(s), %.2f MB -> %s",
        df.height, df.width, df["snapshot_ts"].n_unique(), size_mb, path,
    )


def cmd_board(args: argparse.Namespace) -> None:
    """The Model Board (§5.4.6): a ranked list and three buckets over one
    composite, with each bucket's measured worth attached."""
    file = build_board(panel_path=Path(args.out) / "panel.parquet")
    path, changed = write_json(file.model_dump_json(indent=2), "board.json", Path(args.out))
    logger.info(
        "%s %d players at %s gw%d -> %s",
        "wrote" if changed else "unchanged:", len(file.players), file.season, file.gameweek, path,
    )


def cmd_players(args: argparse.Namespace) -> None:
    """Per-player current state and projection (§5.3.2). Committed per
    §5.3.4, so a fresh clone renders Comparison and Explorer."""
    file = build_players(panel_path=Path(args.out) / "panel.parquet")
    path, changed = write_json(file.model_dump_json(indent=2), "players.json", Path(args.out))
    logger.info(
        "%s %d players at %s gw%d, projecting gw%d (%s) -> %s",
        "wrote" if changed else "unchanged:", len(file.players), file.season,
        file.gameweek, file.projected_gameweek, file.projection_basis, path,
    )


def cmd_observations(args: argparse.Namespace) -> None:
    """The values behind the matrix (§5.6.1). Committed, and loaded by the
    app only when the reader changes the season selection."""
    file = build_observations(panel_path=Path(args.out) / "panel.parquet")
    path, changed = write_json(file.model_dump_json(indent=2), "observations.json", Path(args.out))
    logger.info(
        "%s %d player-seasons over %d metrics (%s) -> %s",
        "wrote" if changed else "unchanged:", len(file.rows), len(file.metrics),
        ", ".join(f"{s.season} gw{s.gameweeks}" for s in file.seasons), path,
    )


def cmd_all(args: argparse.Namespace) -> None:
    cmd_columns(args)
    cmd_panel(args)
    cmd_correlations(args)
    cmd_scorecard(args)
    cmd_fixtures(args)
    cmd_golden(args)
    cmd_timeseries(args)
    cmd_board(args)
    cmd_players(args)
    cmd_observations(args)


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
    subparsers.add_parser("timeseries").set_defaults(func=cmd_timeseries)
    subparsers.add_parser("board").set_defaults(func=cmd_board)
    subparsers.add_parser("players").set_defaults(func=cmd_players)
    subparsers.add_parser("observations").set_defaults(func=cmd_observations)
    subparsers.add_parser("all").set_defaults(func=cmd_all)
    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
