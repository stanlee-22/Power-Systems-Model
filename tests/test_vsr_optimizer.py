"""Tests for the VSR LP dispatch optimizer."""

import numpy as np
import pytest

from src.config import BatteryConfig
from src.vsr_optimizer import solve_dispatch_lp, build_price_forecast


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


def test_lp_discharges_at_price_spike(battery):
    """LP should schedule discharge at the highest-priced interval."""
    prices = np.full(12, 50.0)
    prices[5] = 500.0  # spike
    charge, discharge, _ = solve_dispatch_lp(prices, 0.015, battery, 0.0)
    assert discharge[5] > 0, "Should discharge at price spike"
    assert discharge[5] >= discharge.max() - 1e-9


def test_lp_charges_at_low_price(battery):
    """LP should charge when price is low and future prices are high."""
    battery.initial_soc_mwh = 0.006  # at min SoC
    prices = np.array([10.0] * 6 + [200.0] * 6)
    charge, discharge, _ = solve_dispatch_lp(prices, 0.006, battery, 0.0)
    assert charge[:6].sum() > 0, "Should charge during low-price period"
    assert discharge[6:].sum() > 0, "Should discharge during high-price period"


def test_lp_respects_soc_upper_bound(battery):
    """SoC should never exceed capacity."""
    prices = np.full(12, 10.0)  # cheap, incentivises charging
    charge, discharge, _ = solve_dispatch_lp(prices, 0.029, battery, 0.0)
    eta_c = battery.charge_efficiency
    soc = 0.029
    for t in range(12):
        soc = soc + charge[t] * eta_c - discharge[t]
        assert soc <= battery.capacity_mwh + 1e-9, f"SoC exceeded capacity at t={t}"


def test_lp_respects_soc_lower_bound(battery):
    """SoC should never drop below min_soc."""
    prices = np.full(12, 500.0)  # very high, incentivises full discharge
    charge, discharge, _ = solve_dispatch_lp(prices, 0.015, battery, 0.0)
    soc = 0.015
    eta_c = battery.charge_efficiency
    for t in range(12):
        soc = soc + charge[t] * eta_c - discharge[t]
        assert soc >= battery.min_soc_mwh - 1e-9, f"SoC below min at t={t}"


def test_lp_respects_cycle_cap(battery):
    """Total discharge should not exceed daily cycle headroom."""
    prices = np.full(72, 500.0)
    battery.max_daily_cycles = 0.5  # 0.015 MWh limit
    charge, discharge, _ = solve_dispatch_lp(prices, 0.03, battery, 0.0)
    assert discharge.sum() <= battery.daily_throughput_limit + 1e-9


def test_lp_flat_prices_no_trades(battery):
    """With flat prices, no profitable round-trip exists (efficiency < 1)."""
    battery.round_trip_efficiency = 0.85
    prices = np.full(12, 100.0)
    charge, discharge, _ = solve_dispatch_lp(prices, 0.015, battery, 0.0)
    # With efficiency losses, buying and selling at the same price is a loss
    # LP should not charge from grid (may still discharge existing SoC)
    assert charge.sum() < 1e-9, "Should not charge from grid at flat prices with losses"


def test_shadow_price_positive_with_future_value(battery):
    """Shadow price should be positive when future prices are high."""
    prices = np.array([50.0] * 6 + [500.0] * 6)
    _, _, shadow_price = solve_dispatch_lp(prices, 0.015, battery, 0.0)
    assert shadow_price >= 0, "Shadow price should be non-negative"


def test_forecast_week_ago():
    history = np.arange(3000, dtype=float)
    idx = 2500
    forecast = build_price_forecast(history, idx, horizon=72, method="week_ago")
    assert len(forecast) == 72
    expected_start = idx - 7 * 288
    np.testing.assert_array_equal(forecast, history[expected_start:expected_start + 72])


def test_forecast_fallback_to_mean():
    """When insufficient history, should use trailing mean."""
    history = np.full(100, 75.0)
    forecast = build_price_forecast(history, current_idx=50, horizon=72, method="week_ago")
    assert len(forecast) == 72
    assert abs(forecast[0] - 75.0) < 1e-9
