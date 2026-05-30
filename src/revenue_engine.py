"""Revenue engine.

For each interval, evaluates two strategies and picks the more profitable one:

  1. Dispatch   — sell stored energy at the 5-min dispatch price.
  2. Arbitrage  — buy cheap / sell dear based on wholesale price spread
                  within the same session (simplified: if wholesale price
                  is above the rolling buy-price threshold, discharge).

The engine mutates the SoC after each interval's dispatch decision so that
SoC is consistent throughout the full simulation.
"""

import numpy as np
import pandas as pd

from src.config import BatteryConfig, MarketConfig


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

    # Only dispatch if price is positive (don't dump at negative prices)
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
# Main simulation loop
# ---------------------------------------------------------------------------

def run_revenue_engine(
    df: pd.DataFrame,
    battery: BatteryConfig,
    market: MarketConfig | None = None,
) -> pd.DataFrame:
    """
    Simulate dispatch/arbitrage decisions over the full horizon.

    Requires columns produced by run_soc_engine:
      soc_after_solar, dispatch_price, wholesale_price

    Adds columns:
      soc_end         : SoC at end of interval (after dispatch decision)
      strategy        : 'dispatch' | 'arbitrage' | 'hold'
      energy_dispatched: MWh discharged
      revenue         : $ earned this interval
      cumulative_revenue: running total $
    """
    if market is None:
        market = MarketConfig()

    n = len(df)
    soc_end = np.empty(n)
    strategies = np.empty(n, dtype=object)
    energy_dispatched = np.empty(n)
    revenues = np.empty(n)

    # Rolling minimum wholesale price as a proxy for "buy price" threshold.
    # Uses a 48-period (1 day) lookback so the model can detect intraday cycles.
    lookback = 48
    wholesale_arr = df["wholesale_price"].to_numpy()
    buy_price_threshold = np.empty(n)
    for i in range(n):
        start = max(0, i - lookback)
        buy_price_threshold[i] = np.min(wholesale_arr[start : i + 1])

    current_soc = df["soc_after_solar"].iloc[0]  # re-initialise from SoC engine output

    for i, (_, row) in enumerate(df.iterrows()):
        # SoC entering the market decision = post-solar SoC
        soc_in = row["soc_after_solar"] if i == 0 else current_soc

        # Ensure the SoC engine solar-charge is carried forward
        if i > 0:
            # apply solar charge from this interval on top of last interval's end SoC
            charged_delta = row["soc_after_solar"] - row["soc_before"]
            soc_in = min(current_soc + charged_delta, battery.capacity_mwh)
            soc_in = max(soc_in, battery.min_soc_mwh)

        dp = row["dispatch_price"]
        wp = row["wholesale_price"]
        bpt = buy_price_threshold[i]

        rev_d, eng_d = _dispatch_revenue(soc_in, dp, battery, market)
        rev_a, eng_a = _arbitrage_revenue(soc_in, wp, battery, market, bpt)

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

        new_soc = max(soc_in - energy, battery.min_soc_mwh)

        soc_end[i] = new_soc
        strategies[i] = strategy
        energy_dispatched[i] = energy
        revenues[i] = revenue
        current_soc = new_soc

    df = df.copy()
    df["soc_end"] = soc_end
    df["strategy"] = strategies
    df["energy_dispatched"] = energy_dispatched
    df["revenue"] = revenues
    df["cumulative_revenue"] = df["revenue"].cumsum()
    return df
