"""Unit tests for the SoC engine."""

import pandas as pd
import pytest

from src.config import BatteryConfig
from src.soc_engine import compute_solar_charge, run_soc_engine


@pytest.fixture
def battery():
    return BatteryConfig(
        capacity_mwh=4.0,
        max_charge_mw=2.0,
        round_trip_efficiency=1.0,   # 100% for easy arithmetic
        initial_soc_mwh=0.0,
        interval_hours=0.5,
    )


def test_charge_limited_by_headroom(battery):
    """If battery is nearly full, only the remaining headroom is charged."""
    battery.initial_soc_mwh = 3.9
    charged, new_soc = compute_solar_charge(10.0, 3.9, battery)
    assert abs(new_soc - 4.0) < 1e-9
    assert abs(charged - 0.1) < 1e-9


def test_charge_limited_by_rate(battery):
    """If solar >> max rate, charge is capped at max_charge_per_interval."""
    charged, new_soc = compute_solar_charge(100.0, 0.0, battery)
    assert abs(charged - battery.max_charge_per_interval) < 1e-9


def test_no_negative_charge(battery):
    charged, _ = compute_solar_charge(-5.0, 2.0, battery)
    assert charged == 0.0


def test_run_soc_engine_columns(battery):
    idx = pd.date_range("2025-01-01", periods=10, freq="30min")
    df = pd.DataFrame({"solar_mw": [1.0] * 10, "dispatch_price": [100.0] * 10, "wholesale_price": [90.0] * 10}, index=idx)
    result = run_soc_engine(df, battery)
    assert {"soc_before", "solar_charged", "soc_after_solar"}.issubset(result.columns)
    # SoC should be monotonically non-decreasing (no discharge in SoC engine)
    assert (result["soc_after_solar"].diff().dropna() >= -1e-9).all()
