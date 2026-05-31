"""Integration tests for the VSR dispatch engine."""

import pandas as pd
import numpy as np
import pytest

from src.config import BatteryConfig, MarketConfig, VSRConfig
from src.vsr_engine import run_vsr_engine


@pytest.fixture
def battery():
    return BatteryConfig(
        capacity_mwh=0.03,
        max_charge_mw=0.015,
        max_discharge_mw=0.015,
        round_trip_efficiency=1.0,
        initial_soc_mwh=0.015,
        min_soc_mwh=0.006,
        max_daily_cycles=1.0,
        interval_hours=1 / 12,
    )


@pytest.fixture
def vsr():
    return VSRConfig(
        vsrp_margin=0.15,
        forecast_horizon=12,
        reoptimise_every=6,
        forecast_method="rolling_mean",
    )


@pytest.fixture
def market():
    return MarketConfig(strategy="vsr")


def _make_df(n, prices=None, solar=None):
    idx = pd.date_range("2025-01-01", periods=n, freq="5min")
    return pd.DataFrame({
        "solar_mw": solar if solar is not None else [0.0] * n,
        "dispatch_price": prices if prices is not None else [100.0] * n,
    }, index=idx)


def test_vsr_output_columns(battery, vsr, market):
    df = _make_df(24)
    result = run_vsr_engine(df, battery, vsr, market)
    expected = {
        "soc_before", "solar_charged", "soc_after_solar", "soc_end",
        "strategy", "soc_drawn", "energy_delivered", "grid_bought",
        "cost_basis", "revenue", "vsrp_fee", "bid_price", "cumulative_revenue",
    }
    assert expected.issubset(result.columns)


def test_vsr_soc_within_bounds(battery, vsr, market):
    prices = [20.0] * 50 + [500.0] * 50
    df = _make_df(100, prices=prices)
    result = run_vsr_engine(df, battery, vsr, market)
    assert (result["soc_end"] >= battery.min_soc_mwh - 1e-9).all()
    assert (result["soc_end"] <= battery.capacity_mwh + 1e-9).all()


def test_vsr_revenue_net_of_margin(battery, vsr, market):
    """Discharge revenue should have VSRP fee deducted."""
    battery.initial_soc_mwh = 0.03  # full
    prices = [500.0] * 24  # high prices, should discharge
    df = _make_df(24, prices=prices)
    result = run_vsr_engine(df, battery, vsr, market)

    discharge_rows = result[result["strategy"] == "vsr_discharge"]
    if not discharge_rows.empty:
        for _, row in discharge_rows.iterrows():
            gross = row["dispatch_price"] * row["energy_delivered"]
            assert row["vsrp_fee"] > 0, "VSRP fee should be positive on discharge"
            assert abs(row["vsrp_fee"] - gross * vsr.vsrp_margin) < 1e-6


def test_vsr_daily_cycle_cap(battery, vsr, market):
    battery.max_daily_cycles = 0.5  # 0.015 MWh/day limit
    battery.initial_soc_mwh = 0.03
    prices = [500.0] * 288  # full day, high prices
    df = _make_df(288, prices=prices)
    result = run_vsr_engine(df, battery, vsr, market)
    total_drawn = result["soc_drawn"].sum()
    assert total_drawn <= battery.daily_throughput_limit + 1e-9


def test_vsr_solar_zero_cost_basis(battery, vsr, market):
    """Solar-charged energy should enter at $0 cost basis, increasing margin."""
    battery.initial_soc_mwh = 0.006  # at min SoC
    solar = [0.005] * 12 + [0.0] * 12  # solar in first half
    prices = [10.0] * 12 + [500.0] * 12  # low then high
    df = _make_df(24, prices=prices, solar=solar)
    result = run_vsr_engine(df, battery, vsr, market)

    # After solar charging, some discharge should happen
    discharge_rows = result[result["strategy"] == "vsr_discharge"]
    if not discharge_rows.empty:
        for _, row in discharge_rows.iterrows():
            # Cost basis should be $0 or near-$0 for solar-charged energy
            assert row["cost_basis"] < row["dispatch_price"] * row["energy_delivered"]


def test_vsr_no_regression_on_existing_engine():
    """Existing revenue engine should still work at 30-min resolution."""
    from src.revenue_engine import run_revenue_engine

    battery = BatteryConfig(
        capacity_mwh=4.0, max_charge_mw=2.0, max_discharge_mw=2.0,
        round_trip_efficiency=1.0, initial_soc_mwh=4.0,
        max_daily_cycles=10.0, interval_hours=0.5,
    )
    market = MarketConfig(arbitrage_min_spread=10.0, strategy="solar_shift")
    idx = pd.date_range("2025-01-01", periods=5, freq="30min")
    df = pd.DataFrame({
        "solar_mw": [0.0] * 5,
        "dispatch_price": [200.0] * 5,
        "wholesale_price": [200.0] * 5,
    }, index=idx)
    result = run_revenue_engine(df, battery, market)
    assert "soc_end" in result.columns
    assert "revenue" in result.columns
