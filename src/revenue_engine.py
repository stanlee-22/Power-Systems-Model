"""Revenue engine.

For each interval, evaluates two strategies and picks the more profitable one:

  1. Dispatch   — sell stored energy at the 5-min dispatch price.
  2. Arbitrage  — buy cheap / sell dear based on wholesale price spread
                  within the same session (simplified: if wholesale price
                  is above the rolling buy-price threshold, discharge).

The engine runs solar charging and dispatch decisions in a single pass so that
SoC is consistent throughout the full simulation — the battery recharges from
solar after each dispatch, enabling daily cycling.
"""

import numpy as np
import pandas as pd

from src.config import BatteryConfig, MarketConfig
from src.soc_engine import compute_solar_charge


# ---------------------------------------------------------------------------
# Strategy helpers
# ---------------------------------------------------------------------------

def _dispatch_revenue(
    soc: float,
    dispatch_price: float,
    battery: BatteryConfig,
    market: MarketConfig,
) -> tuple[float, float]:
    """
    Revenue from dispatching stored energy at the dispatch price.

    Returns (revenue_$, energy_discharged_mwh).
    """
    available = soc - battery.min_soc_mwh
    dischargeable = min(
        available,
        battery.max_discharge_per_interval * battery.discharge_efficiency,
    ) * market.dispatch_fraction
    dischargeable = max(dischargeable, 0.0)

    if dispatch_price <= 0 or dischargeable == 0:
        return 0.0, 0.0

    revenue = dispatch_price * dischargeable
    return revenue, dischargeable


def _arbitrage_revenue(
    soc: float,
    wholesale_price: float,
    battery: BatteryConfig,
    market: MarketConfig,
    buy_price_threshold: float,
) -> tuple[float, float]:
    """
    Revenue from arbitrage: discharge when price is above buy threshold + spread.

    Returns (revenue_$, energy_discharged_mwh).
    """
    spread = wholesale_price - buy_price_threshold
    if spread < market.arbitrage_min_spread:
        return 0.0, 0.0

    available = soc - battery.min_soc_mwh
    dischargeable = min(
        available,
        battery.max_discharge_per_interval * battery.discharge_efficiency,
    )
    dischargeable = max(dischargeable, 0.0)

    if dischargeable == 0:
        return 0.0, 0.0

    revenue = wholesale_price * dischargeable
    return revenue, dischargeable


# ---------------------------------------------------------------------------
# Main simulation loop — combined SoC + revenue in a single pass
# ---------------------------------------------------------------------------

def run_revenue_engine(
    df: pd.DataFrame,
    battery: BatteryConfig,
    market: MarketConfig | None = None,
) -> pd.DataFrame:
    """
    Simulate solar charging and dispatch/arbitrage decisions in one pass.

    Each interval:
      1. Record soc_before (carried from previous interval's soc_end)
      2. Charge from solar -> soc_after_solar
      3. Evaluate dispatch vs arbitrage vs hold
      4. Discharge if profitable -> soc_end

    This ensures the battery properly cycles: dispatch depletes SoC, then
    solar recharges it the next day, enabling repeated arbitrage.

    Adds columns:
      soc_before, solar_charged, soc_after_solar,
      soc_end, strategy, energy_dispatched, revenue, cumulative_revenue
    """
    if market is None:
        market = MarketConfig()

    n = len(df)
    soc_before = np.empty(n)
    solar_charged = np.empty(n)
    soc_after_solar = np.empty(n)
    soc_end = np.empty(n)
    strategies = np.empty(n, dtype=object)
    energy_dispatched = np.empty(n)
    revenues = np.empty(n)

    # Rolling minimum wholesale price as a proxy for "buy price" threshold.
    lookback = 48
    wholesale_arr = df["wholesale_price"].to_numpy()
    buy_price_threshold = np.empty(n)
    for i in range(n):
        start = max(0, i - lookback)
        buy_price_threshold[i] = np.min(wholesale_arr[start : i + 1])

    current_soc = battery.initial_soc_mwh

    for i, (_, row) in enumerate(df.iterrows()):
        # 1. Record SoC entering this interval
        soc_before[i] = current_soc

        # 2. Charge from solar
        charged, soc_post_solar = compute_solar_charge(
            row["solar_mw"], current_soc, battery,
        )
        solar_charged[i] = charged
        soc_after_solar[i] = soc_post_solar

        # 3. Evaluate dispatch strategies using post-solar SoC
        dp = row["dispatch_price"]
        wp = row["wholesale_price"]
        bpt = buy_price_threshold[i]

        rev_d, eng_d = _dispatch_revenue(soc_post_solar, dp, battery, market)
        rev_a, eng_a = _arbitrage_revenue(soc_post_solar, wp, battery, market, bpt)

        if rev_d >= rev_a and rev_d > 0:
            strategy = "dispatch"
            energy = eng_d
            revenue = rev_d
        elif rev_a > rev_d and rev_a > 0:
            strategy = "arbitrage"
            energy = eng_a
            revenue = rev_a
        else:
            strategy = "hold"
            energy = 0.0
            revenue = 0.0

        # 4. Update SoC after dispatch
        new_soc = max(soc_post_solar - energy, battery.min_soc_mwh)

        soc_end[i] = new_soc
        strategies[i] = strategy
        energy_dispatched[i] = energy
        revenues[i] = revenue
        current_soc = new_soc

    df = df.copy()
    df["soc_before"] = soc_before
    df["solar_charged"] = solar_charged
    df["soc_after_solar"] = soc_after_solar
    df["soc_end"] = soc_end
    df["strategy"] = strategies
    df["energy_dispatched"] = energy_dispatched
    df["revenue"] = revenues
    df["cumulative_revenue"] = df["revenue"].cumsum()
    return df
