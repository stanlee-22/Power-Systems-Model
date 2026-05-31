"""Revenue engine — single-pass SoC + market simulation.

Supports two strategies:

  "solar_shift" — charge from solar only, sell at high prices. Revenue is the
                  full sale price (energy was free). This is the default and
                  what a solar+battery household actually does.

  "arbitrage"  — buy from the grid at low wholesale prices, sell at high
                 prices. Revenue is the net margin (sell - buy) after
                 accounting for round-trip efficiency losses. The battery
                 also absorbs free solar when available.

Physics modelled:
  - One-way charge/discharge efficiencies (sqrt of round-trip each)
  - Charging: grid energy stored = grid_mwh_bought × charge_efficiency
  - Discharging: grid energy delivered = soc_drawn × discharge_efficiency
  - Daily cycle cap (max equivalent full cycles per calendar day)
  - Min SoC floor
"""

import numpy as np
import pandas as pd

from src.config import BatteryConfig, MarketConfig
from src.soc_engine import compute_solar_charge


# ---------------------------------------------------------------------------
# Main simulation loop
# ---------------------------------------------------------------------------

def run_revenue_engine(
    df: pd.DataFrame,
    battery: BatteryConfig,
    market: MarketConfig | None = None,
) -> pd.DataFrame:
    """
    Simulate solar charging and market decisions in one chronological pass.

    Each interval:
      1. Reset daily cycle counter at midnight
      2. Charge from solar (free, limited by charge rate and headroom)
      3. Decide: sell / buy / hold based on strategy and price signals
      4. Apply efficiency losses and cycle cap
      5. Update SoC

    Columns added:
      soc_before, solar_charged, soc_after_solar, soc_end,
      strategy, soc_drawn, energy_delivered, grid_bought,
      cost_basis, revenue, cumulative_revenue
    """
    if market is None:
        market = MarketConfig()

    n = len(df)
    soc_before = np.empty(n)
    solar_charged = np.empty(n)
    soc_after_solar = np.empty(n)
    soc_end = np.empty(n)
    strategies = np.empty(n, dtype=object)
    soc_drawn = np.empty(n)
    energy_delivered = np.empty(n)
    grid_bought = np.empty(n)
    cost_basis_arr = np.empty(n)
    revenues = np.empty(n)

    is_arb = market.strategy == "arbitrage"

    # Pre-compute rolling price percentiles for buy/sell signals.
    lookback = 48  # 24 hours
    wholesale_arr = df["wholesale_price"].to_numpy()
    sell_threshold = np.empty(n)
    buy_ceiling = np.empty(n)
    for i in range(n):
        start = max(0, i - lookback)
        window = wholesale_arr[start : i + 1]
        sell_threshold[i] = np.percentile(window, 75)
        buy_ceiling[i] = np.percentile(window, 25)

    current_soc = battery.initial_soc_mwh
    daily_discharged = 0.0
    current_day = None

    # Weighted-average cost basis for energy in the battery ($/MWh of SoC).
    # Solar energy enters at $0; grid energy enters at buy_price/charge_eff.
    cost_basis = 0.0

    for i, (idx_val, row) in enumerate(df.iterrows()):
        # --- Reset daily cycle counter at midnight ---
        day = idx_val.date() if hasattr(idx_val, 'date') else None
        if day != current_day:
            daily_discharged = 0.0
            current_day = day

        # 1. Record SoC entering this interval
        soc_before[i] = current_soc

        # 2. Charge from solar (free energy)
        charged, soc_post_solar = compute_solar_charge(
            row["solar_mw"], current_soc, battery,
        )
        solar_charged[i] = charged
        soc_after_solar[i] = soc_post_solar

        # Update cost basis: solar energy enters at $0
        if charged > 0 and soc_post_solar > 0:
            cost_basis = cost_basis * (current_soc / soc_post_solar)

        wp = row["wholesale_price"]

        # 3. Decide action
        available_soc = soc_post_solar - battery.min_soc_mwh
        cycle_headroom = battery.daily_throughput_limit - daily_discharged

        strategy_out = "hold"
        drawn = 0.0
        delivered = 0.0
        bought = 0.0
        cost = 0.0
        rev = 0.0

        if is_arb:
            margin = wp - cost_basis
            if margin >= market.arbitrage_min_spread and available_soc > 1e-9 and cycle_headroom > 1e-9:
                # Sell: SoC drawn -> delivered to grid after discharge efficiency
                max_draw = min(available_soc, battery.max_discharge_per_interval, cycle_headroom)
                drawn = max(max_draw, 0.0)
                delivered = drawn * battery.discharge_efficiency
                cost = cost_basis * drawn
                rev = wp * delivered - cost
                strategy_out = "arbitrage"
            elif wp <= buy_ceiling[i] and wp >= 0 and soc_post_solar < battery.capacity_mwh:
                # Buy from grid: pay for grid MWh, store less due to charge efficiency
                headroom = battery.capacity_mwh - soc_post_solar
                max_grid_mwh = battery.max_charge_per_interval
                bought = min(max_grid_mwh, headroom / battery.charge_efficiency)
                stored = bought * battery.charge_efficiency
                cost = wp * bought
                rev = -cost  # outflow, negative revenue

                # Update cost basis with blended price
                new_soc = soc_post_solar + stored
                if new_soc > 0:
                    cost_basis = (cost_basis * soc_post_solar + (wp / battery.charge_efficiency) * stored) / new_soc

                strategy_out = "grid_charge"
        else:
            # Solar-shift: sell solar-charged energy at high prices, full price is profit
            if wp >= sell_threshold[i] and wp > 0 and available_soc > 1e-9 and cycle_headroom > 1e-9:
                max_draw = min(available_soc, battery.max_discharge_per_interval, cycle_headroom)
                drawn = max(max_draw, 0.0)
                delivered = drawn * battery.discharge_efficiency
                rev = wp * delivered
                strategy_out = "solar_shift"

        # 4. Update SoC
        if strategy_out == "grid_charge":
            new_soc = soc_post_solar + bought * battery.charge_efficiency
        else:
            new_soc = soc_post_solar - drawn
            daily_discharged += drawn

        new_soc = max(min(new_soc, battery.capacity_mwh), battery.min_soc_mwh)

        soc_end[i] = new_soc
        strategies[i] = strategy_out
        soc_drawn[i] = drawn
        energy_delivered[i] = delivered
        grid_bought[i] = bought
        cost_basis_arr[i] = cost
        revenues[i] = rev
        current_soc = new_soc

    df = df.copy()
    df["soc_before"] = soc_before
    df["solar_charged"] = solar_charged
    df["soc_after_solar"] = soc_after_solar
    df["soc_end"] = soc_end
    df["strategy"] = strategies
    df["soc_drawn"] = soc_drawn
    df["energy_delivered"] = energy_delivered
    df["grid_bought"] = grid_bought
    df["cost_basis"] = cost_basis_arr
    df["revenue"] = revenues
    df["cumulative_revenue"] = df["revenue"].cumsum()
    return df
