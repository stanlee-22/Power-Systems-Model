"""Unit tests for the revenue engine."""

import pandas as pd
import pytest

from src.config import BatteryConfig, MarketConfig
from src.revenue_engine import run_revenue_engine


@pytest.fixture
def battery():
    return BatteryConfig(
        capacity_mwh=4.0,
        max_charge_mw=2.0,
        max_discharge_mw=2.0,
        round_trip_efficiency=1.0,
        initial_soc_mwh=4.0,
        min_soc_mwh=0.0,
        max_daily_cycles=10.0,  # unconstrained for basic tests
        interval_hours=0.5,
    )


def test_soc_decreases_after_sell(battery):
    market = MarketConfig(arbitrage_min_spread=10.0, strategy="solar_shift")
    idx = pd.date_range("2025-01-01", periods=5, freq="30min")
    df = pd.DataFrame({
        "solar_mw": [0.0] * 5,
        "dispatch_price": [200.0] * 5,
        "wholesale_price": [200.0] * 5,
    }, index=idx)
    result = run_revenue_engine(df, battery, market)
    assert (result["soc_end"] <= result["soc_after_solar"] + 1e-9).all()


def test_output_columns(battery):
    market = MarketConfig(strategy="arbitrage")
    idx = pd.date_range("2025-01-01", periods=4, freq="30min")
    df = pd.DataFrame({
        "solar_mw": [1.0] * 4,
        "dispatch_price": [100.0] * 4,
        "wholesale_price": [80.0] * 4,
    }, index=idx)
    result = run_revenue_engine(df, battery, market)
    expected = {
        "soc_before", "solar_charged", "soc_after_solar",
        "soc_end", "strategy", "soc_drawn", "energy_delivered",
        "grid_bought", "cost_basis", "revenue", "cumulative_revenue",
    }
    assert expected.issubset(result.columns)


def test_efficiency_reduces_delivered_energy(battery):
    """Discharge efficiency should reduce energy delivered to grid."""
    battery.round_trip_efficiency = 0.81  # 0.9 each way
    battery.initial_soc_mwh = 4.0
    market = MarketConfig(arbitrage_min_spread=0.0, strategy="solar_shift")
    idx = pd.date_range("2025-01-01", periods=1, freq="30min")
    df = pd.DataFrame({
        "solar_mw": [0.0],
        "dispatch_price": [200.0],
        "wholesale_price": [200.0],
    }, index=idx)
    result = run_revenue_engine(df, battery, market)
    row = result.iloc[0]
    assert row["soc_drawn"] > 0
    assert row["energy_delivered"] < row["soc_drawn"]
    assert abs(row["energy_delivered"] - row["soc_drawn"] * 0.9) < 1e-9


def test_arbitrage_uses_net_margin():
    """Arbitrage revenue should be net margin, not gross sale price."""
    battery = BatteryConfig(
        capacity_mwh=4.0, max_charge_mw=2.0, max_discharge_mw=2.0,
        round_trip_efficiency=1.0, initial_soc_mwh=0.0,
        max_daily_cycles=10.0, interval_hours=0.5,
    )
    market = MarketConfig(arbitrage_min_spread=10.0, strategy="arbitrage")

    # Buy at $20, then sell at $100 -> margin should be ~$80, not $100
    idx = pd.date_range("2025-01-01", periods=50, freq="30min")
    prices = [20.0] * 25 + [100.0] * 25
    df = pd.DataFrame({
        "solar_mw": [0.0] * 50,
        "dispatch_price": prices,
        "wholesale_price": prices,
    }, index=idx)
    result = run_revenue_engine(df, battery, market)

    sell_rows = result[result["strategy"] == "arbitrage"]
    if not sell_rows.empty:
        for _, row in sell_rows.iterrows():
            # Revenue should be less than gross (wp * delivered)
            gross = row["wholesale_price"] * row["energy_delivered"]
            assert row["revenue"] < gross, "Arbitrage should book net margin, not gross"


def test_daily_cycle_cap():
    """Battery should stop discharging after hitting daily cycle limit."""
    battery = BatteryConfig(
        capacity_mwh=4.0, max_charge_mw=2.0, max_discharge_mw=2.0,
        round_trip_efficiency=1.0, initial_soc_mwh=4.0,
        max_daily_cycles=0.5,  # 2 MWh max per day
        interval_hours=0.5,
    )
    market = MarketConfig(arbitrage_min_spread=0.0, strategy="solar_shift")

    idx = pd.date_range("2025-01-01", periods=10, freq="30min")
    df = pd.DataFrame({
        "solar_mw": [0.0] * 10,
        "dispatch_price": [200.0] * 10,
        "wholesale_price": [200.0] * 10,
    }, index=idx)
    result = run_revenue_engine(df, battery, market)
    total_drawn = result["soc_drawn"].sum()
    assert total_drawn <= 2.0 + 1e-9, f"Should cap at 2 MWh/day, got {total_drawn}"


def test_min_soc_floor():
    """Battery should never discharge below min_soc_mwh."""
    battery = BatteryConfig(
        capacity_mwh=4.0, max_charge_mw=2.0, max_discharge_mw=2.0,
        round_trip_efficiency=1.0, initial_soc_mwh=4.0,
        min_soc_mwh=1.0,  # 25% floor
        max_daily_cycles=10.0, interval_hours=0.5,
    )
    market = MarketConfig(arbitrage_min_spread=0.0, strategy="solar_shift")

    idx = pd.date_range("2025-01-01", periods=10, freq="30min")
    df = pd.DataFrame({
        "solar_mw": [0.0] * 10,
        "dispatch_price": [200.0] * 10,
        "wholesale_price": [200.0] * 10,
    }, index=idx)
    result = run_revenue_engine(df, battery, market)
    assert (result["soc_end"] >= 1.0 - 1e-9).all(), "SoC should never drop below min_soc_mwh"


def test_battery_recharges_after_sell():
    """Battery should recharge from solar after selling."""
    battery = BatteryConfig(
        capacity_mwh=4.0, max_charge_mw=2.0, max_discharge_mw=2.0,
        round_trip_efficiency=1.0, initial_soc_mwh=4.0,
        max_daily_cycles=10.0, interval_hours=0.5,
    )
    market = MarketConfig(arbitrage_min_spread=0.0, strategy="solar_shift")

    idx = pd.date_range("2025-01-01 17:00", periods=8, freq="30min")
    df = pd.DataFrame({
        "solar_mw":        [0, 0, 0, 0, 2.0, 2.0, 2.0, 0],
        "dispatch_price":  [300, 300, 300, 300, -10, -10, -10, 300],
        "wholesale_price": [300, 300, 300, 300, -10, -10, -10, 300],
    }, index=idx)
    result = run_revenue_engine(df, battery, market)

    soc_end = result["soc_end"].values
    min_soc = soc_end[:4].min()
    recharged_soc = soc_end[6]
    assert recharged_soc > min_soc, "SoC should recover after solar recharging"


def test_no_sell_at_negative_price():
    """Should not sell when wholesale price is negative."""
    battery = BatteryConfig(
        capacity_mwh=4.0, max_charge_mw=2.0, max_discharge_mw=2.0,
        round_trip_efficiency=1.0, initial_soc_mwh=4.0,
        max_daily_cycles=10.0, interval_hours=0.5,
    )
    market = MarketConfig(strategy="solar_shift")

    idx = pd.date_range("2025-01-01", periods=5, freq="30min")
    df = pd.DataFrame({
        "solar_mw": [0.0] * 5,
        "dispatch_price": [-50.0] * 5,
        "wholesale_price": [-50.0] * 5,
    }, index=idx)
    result = run_revenue_engine(df, battery, market)
    assert (result["soc_drawn"] == 0).all(), "Should never sell at negative prices"
