"""§7.2 market signals over analytics/deltas.py's windowed output."""

from __future__ import annotations

import polars as pl
import pytest

from analytics.trending import ownership_acceleration, price_change_pressure, transfer_velocity


def _deltas_row(**overrides) -> dict:
    row = {
        "element_id": 101,
        "selected_by_percent": 20.0,
        "selected_by_percent_delta_1h": 0.0,
        "selected_by_percent_delta_24h": 0.0,
        "transfers_in_event_delta_1h": 0,
        "transfers_out_event_delta_1h": 0,
    }
    row.update(overrides)
    return row


def test_ownership_acceleration_positive_when_recent_rate_exceeds_average():
    df = pl.DataFrame([_deltas_row(selected_by_percent_delta_1h=2.0, selected_by_percent_delta_24h=2.4)])
    result = ownership_acceleration(df)
    row = result.row(0, named=True)
    # 1h rate = 2.0/hr; 24h average rate = 2.4/24 = 0.1/hr -> clearly accelerating
    assert row["ownership_acceleration"] == pytest.approx(2.0 - 0.1)


def test_ownership_acceleration_raises_on_missing_columns():
    df = pl.DataFrame([{"element_id": 101, "selected_by_percent_delta_1h": 1.0}])
    with pytest.raises(ValueError, match="missing"):
        ownership_acceleration(df)


def test_transfer_velocity_nets_in_minus_out():
    df = pl.DataFrame([_deltas_row(transfers_in_event_delta_1h=500, transfers_out_event_delta_1h=200)])
    result = transfer_velocity(df)
    row = result.row(0, named=True)
    assert row["net_transfers"] == 300
    assert row["transfer_velocity"] == pytest.approx(300.0)  # hours=1.0 default


def test_transfer_velocity_scales_by_hours():
    df = pl.DataFrame([_deltas_row(transfers_in_event_delta_1h=1200, transfers_out_event_delta_1h=0)])
    result = transfer_velocity(df, hours=2.0)
    assert result.row(0, named=True)["transfer_velocity"] == pytest.approx(600.0)


def test_price_change_pressure_scales_down_for_highly_owned_players():
    low_ownership = pl.DataFrame([_deltas_row(element_id=1, selected_by_percent=1.0, transfers_in_event_delta_1h=1000, transfers_out_event_delta_1h=0)])
    high_ownership = pl.DataFrame([_deltas_row(element_id=2, selected_by_percent=50.0, transfers_in_event_delta_1h=1000, transfers_out_event_delta_1h=0)])

    low_pressure = price_change_pressure(low_ownership).row(0, named=True)["price_change_pressure"]
    high_pressure = price_change_pressure(high_ownership).row(0, named=True)["price_change_pressure"]

    # same raw net transfers, but the highly-owned player needs more to move -> lower pressure per this proxy
    assert low_pressure > high_pressure


def test_price_change_pressure_negative_for_net_sells():
    df = pl.DataFrame([_deltas_row(transfers_in_event_delta_1h=100, transfers_out_event_delta_1h=900)])
    result = price_change_pressure(df)
    assert result.row(0, named=True)["price_change_pressure"] < 0
