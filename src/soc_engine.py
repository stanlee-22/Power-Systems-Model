"""State-of-charge (SoC) engine.

Iterates chronologically over the master frame and tracks battery SoC after
absorbing solar generation each interval.
"""

import numpy as np
import pandas as pd

from src.config import BatteryConfig


def compute_solar_charge(
    solar_mw: float,
    current_soc: float,
    battery: BatteryConfig,
) -> tuple[float, float]:
    """
    Determine how much solar energy can be absorbed and the resulting SoC.

    Returns (energy_charged_mwh, new_soc).
    """
    available = solar_mw * battery.interval_hours * battery.charge_efficiency
    headroom = battery.capacity_mwh - current_soc
    max_per_interval = battery.max_charge_per_interval * battery.charge_efficiency

    charged = min(available, headroom, max_per_interval)
    charged = max(charged, 0.0)

    new_soc = current_soc + charged
    return charged, new_soc


def run_soc_engine(df: pd.DataFrame, battery: BatteryConfig) -> pd.DataFrame:
    """
    Run the solar-charging SoC loop over the full dataset.

    Adds columns:
      - soc_before   : SoC at the start of the interval (before solar charge)
      - solar_charged: energy absorbed from solar (MWh)
      - soc_after_solar: SoC after solar charging (before any market dispatch)

    NOTE: This is a standalone pre-pass. When used with the revenue engine,
    prefer run_combined_engine() from revenue_engine.py which interleaves
    charging and dispatch in a single pass for accurate SoC tracking.
    """
    n = len(df)
    soc_before = np.empty(n)
    solar_charged = np.empty(n)
    soc_after_solar = np.empty(n)

    current_soc = battery.initial_soc_mwh

    for i, (_, row) in enumerate(df.iterrows()):
        soc_before[i] = current_soc
        charged, new_soc = compute_solar_charge(row["solar_mw"], current_soc, battery)
        solar_charged[i] = charged
        soc_after_solar[i] = new_soc
        current_soc = new_soc

    df = df.copy()
    df["soc_before"] = soc_before
    df["solar_charged"] = solar_charged
    df["soc_after_solar"] = soc_after_solar
    return df
