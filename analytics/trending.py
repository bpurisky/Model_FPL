"""§7.2's market signals, built on `analytics/deltas.py`'s windowed deltas:
ownership acceleration, transfer velocity, price-change pressure.

FPL's price-change mechanism has never been officially published (§4.1's
own price-model note carries the same caveat) — every signal here is a
transparent, documented heuristic over publicly-tracked fields, not a
reverse-engineered replica of FPL's internal formula. `analytics/price_model.py`
is what actually validates these against real observed price changes; this
module doesn't assume they're correct by construction.
"""

from __future__ import annotations

import polars as pl


def _require_columns(df: pl.DataFrame, *columns: str) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"deltas frame is missing required column(s): {missing}")


def ownership_acceleration(
    deltas: pl.DataFrame, short_window: str = "1h", short_hours: float = 1.0, long_window: str = "24h", long_hours: float = 24.0
) -> pl.DataFrame:
    """element_id, both raw ownership deltas, and `ownership_acceleration`
    = the short-window hourly rate minus the long-window hourly rate.
    Positive: ownership is moving *faster* right now than its own recent
    average (a real breakout, not just sustained drift); negative: slowing.
    """
    short_col, long_col = f"selected_by_percent_delta_{short_window}", f"selected_by_percent_delta_{long_window}"
    _require_columns(deltas, short_col, long_col)
    return deltas.select(
        "element_id",
        pl.col(short_col).alias("ownership_delta_short"),
        pl.col(long_col).alias("ownership_delta_long"),
        ((pl.col(short_col) / short_hours) - (pl.col(long_col) / long_hours)).alias("ownership_acceleration"),
    )


def transfer_velocity(deltas: pl.DataFrame, window: str = "1h", hours: float = 1.0) -> pl.DataFrame:
    """element_id, net transfers (in minus out) within `window`, and that
    same count expressed as an hourly rate for comparison across windows."""
    in_col, out_col = f"transfers_in_event_delta_{window}", f"transfers_out_event_delta_{window}"
    _require_columns(deltas, in_col, out_col)
    net = pl.col(in_col) - pl.col(out_col)
    return deltas.select("element_id", net.alias("net_transfers"), (net / hours).alias("transfer_velocity"))


def price_change_pressure(deltas: pl.DataFrame, window: str = "1h") -> pl.DataFrame:
    """A transparent proxy for how close a player is to FPL's (unpublished)
    price-change threshold: net transfer flow in `window`, scaled down by
    current ownership. A highly-owned player needs proportionally more net
    transfers to move the same amount under FPL's real mechanism, so raw
    net-transfer-count alone isn't comparable across players — dividing by
    (ownership + 1) makes it roughly so. `+1` avoids a division blow-up for
    an essentially-unowned player, not a tuned constant.
    """
    in_col, out_col = f"transfers_in_event_delta_{window}", f"transfers_out_event_delta_{window}"
    _require_columns(deltas, in_col, out_col, "selected_by_percent")
    net = pl.col(in_col) - pl.col(out_col)
    return deltas.select(
        "element_id", net.alias("net_transfers"), (net / (pl.col("selected_by_percent") + 1.0)).alias("price_change_pressure")
    )
