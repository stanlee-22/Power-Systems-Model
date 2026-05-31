"""Unit tests for the revenue engine."""

import pandas as pd
import pytest

from src.config import BatteryConfig, MarketConfig
from src.revenue_engine import run_revenue_engine, _dispatch_revenue, _arbitrage_revenue


@pytest.fixture
def battery():
    return BatteryConfig(
        capacity_mwh=4.0,
        max_charge_mw=2.0,
        max_discharge_mw=2.0,
        round_trip_efficiency=1.0,
        initial_soc_mwh=4.0,
        interval_hours=0.5,
    )


@pytest.fixture
def market():
    return MarketConfig(arbitrage_min_spread=20.0, dispatch_fraction=1.0)


def test_dispatch_revenue_positive_price(battery, market):
    rev, eng = _dispatch_revenue(4.0, 100.0, battery, market)
    assert rev > 0
    assert eng > 0


def test_dispatch_revenue_negative_price(battery, market):
    """Should not dispatch at negative prices."""
    rev, eng = _dispatch_revenue(4.0, -10.0, battery, market)
    assert rev == 0.0
    assert eng == 0.0


def test_arbitrage_below_spread(battery, market):
    """No trade when spread is below threshold."""
    rev, eng = _arbitrage_revenue(4.0, 50.0, battery, market, buy_price_threshold=40.0)
    # spread = 10, threshold = 20 -> no trade
    assert rev == 0.0


def test_arbitrage_above_spread(battery, market):
    rev, eng = _arbitrage_revenue(4.0, 150.0, battery, market, buy_price_threshold=50.0)
    assert rev > 0
    assert eng > 0


def test_soc_decreases_after_dispatch(battery, market):
    idx = pd.date_range("2025-01-01", periods=5, freq="30min")
    df = pd.DataFrame({
        "solar_mw": [0.0] * 5,
        "dispatch_price": [200.0] * 5,
        "wholesale_price": [50.0] * 5,
    }, index=idx)
    result = run_revenue_engine(df, battery, market)
    assert (result["soc_end"] <= result["soc_after_solar"] + 1e-9).all()


def test_revenue_engine_columns(battery, market):
    idx = pd.date_range("2025-01-01", periods=4, freq="30min")
    df = pd.DataFrame({
        "solar_mw": [1.0] * 4,
        "dispatch_price": [100.0] * 4,
        "wholesale_price": [80.0] * 4,
    }, index=idx)
    result = run_revenue_engine(df, battery, market)
    expected = {
        "soc_before", "solar_charged", "soc_after_solar",
        "soc_end", "strategy", "energy_dispatched", "revenue", "cumulative_revenue",
    }
    assert expected.issubset(result.columns)


def test_battery_recharges_after_dispatch(market):
    """Core regression test: battery should recharge from solar after dispatching."""
    battery = BatteryConfig(
        capacity_mwh=4.0,
        max_charge_mw=2.0,
        max_discharge_mw=2.0,
        round_trip_efficiency=1.0,
        initial_soc_mwh=4.0,
        interval_hours=0.5,
    )
    # Evening: high price, no solar -> dispatch drains battery
    # Daytime: low price, solar -> battery recharges
    # Next evening: high price again -> should dispatch again
    idx = pd.date_range("2025-01-01 17:00", periods=8, freq="30min")
    df = pd.DataFrame({
        "solar_mw":        [0, 0, 0, 0,  2.0, 2.0, 2.0, 0],
        "dispatch_price":  [300, 300, 300, 300, -10, -10, -10, 300],
        "wholesale_price": [300, 300, 300, 300, -10, -10, -10, 300],
    }, index=idx)
    result = run_revenue_engine(df, battery, market)

    # Battery should have dispatched early, recharged from solar, then dispatched again
    dispatches = result[result["strategy"] == "dispatch"]
    assert len(dispatches) >= 2, "Battery should dispatch more than once after recharging"

    # Verify SoC goes down then back up
    soc_end = result["soc_end"].values
    min_soc = soc_end[:4].min()
    recharged_soc = soc_end[6]
    assert recharged_soc > min_soc, "SoC should recover after solar recharging"
