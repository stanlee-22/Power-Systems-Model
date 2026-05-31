"""VSR (Voluntarily Scheduled Resource) dispatch engine.

Simulates a residential battery participating in NEM dispatch under IPRR
"Scheduled Lite" via a VSR Provider (VSRP). Runs at 5-minute resolution.

Each re-optimisation window:
  1. Build a forward price forecast from historical data
  2. Solve a rolling-horizon LP for the optimal charge/discharge plan
  3. Execute the plan interval-by-interval, applying efficiency and tracking SoC

Revenue is net of VSRP margin (aggregator's cut) and energy cost basis.
"""

import numpy as np
import pandas as pd

from src.config import BatteryConfig, MarketConfig, VSRConfig
from src.soc_engine import compute_solar_charge
from src.vsr_optimizer import build_price_forecast, solve_dispatch_lp


def run_vsr_engine(
    df: pd.DataFrame,
    battery: BatteryConfig,
    vsr: VSRConfig,
    market: MarketConfig | None = None,
) -> pd.DataFrame:
    """
    Simulate VSR dispatch over the full dataset at 5-minute resolution.

    Each interval:
      1. Reset daily cycle counter at midnight
      2. Charge from solar (free)
      3. Every `reoptimise_every` intervals: solve LP for new dispatch plan
      4. Execute plan action (grid charge / discharge / hold)
      5. Apply VSRP margin, track cost basis
      6. Update SoC

    Columns added:
      soc_before, solar_charged, soc_after_solar, soc_end,
      strategy, soc_drawn, energy_delivered, grid_bought,
      cost_basis, revenue, vsrp_fee, bid_price, cumulative_revenue
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
    vsrp_fees = np.empty(n)
    bid_prices = np.empty(n)

    price_history = df["dispatch_price"].to_numpy()
    current_soc = battery.initial_soc_mwh
    daily_discharged = 0.0
    current_day = None
    cost_basis = 0.0

    charge_plan = None
    discharge_plan = None
    plan_offset = 0

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

        # 3. Re-optimise every N intervals (or on first interval)
        if charge_plan is None or plan_offset >= vsr.reoptimise_every:
            forecast = build_price_forecast(
                price_history, i, vsr.forecast_horizon, vsr.forecast_method,
            )
            charge_plan, discharge_plan, shadow = solve_dispatch_lp(
                forecast, soc_post_solar, battery, daily_discharged,
            )
            plan_offset = 0
            bid_price = shadow
        else:
            bid_price = bid_prices[i - 1] if i > 0 else 0.0

        # 4. Execute this interval's planned action
        planned_charge = charge_plan[plan_offset] if plan_offset < len(charge_plan) else 0.0
        planned_discharge = discharge_plan[plan_offset] if plan_offset < len(discharge_plan) else 0.0
        plan_offset += 1

        dp = row["dispatch_price"]

        strategy_out = "hold"
        drawn = 0.0
        delivered = 0.0
        bought = 0.0
        cost = 0.0
        rev = 0.0
        fee = 0.0

        available_soc = soc_post_solar - battery.min_soc_mwh
        cycle_headroom = battery.daily_throughput_limit - daily_discharged

        if planned_discharge > 1e-9 and available_soc > 1e-9 and cycle_headroom > 1e-9:
            drawn = min(planned_discharge, available_soc, battery.max_discharge_per_interval, cycle_headroom)
            delivered = drawn * battery.discharge_efficiency
            gross_rev = dp * delivered
            cost = cost_basis * drawn
            fee = gross_rev * vsr.vsrp_margin
            rev = gross_rev - cost - fee
            strategy_out = "vsr_discharge"

        elif planned_charge > 1e-9 and soc_post_solar < battery.capacity_mwh:
            headroom = battery.capacity_mwh - soc_post_solar
            bought = min(planned_charge, battery.max_charge_per_interval, headroom / battery.charge_efficiency)
            stored = bought * battery.charge_efficiency
            cost = dp * bought
            rev = -cost
            fee = 0.0

            # Update cost basis with blended price
            new_soc_after_buy = soc_post_solar + stored
            if new_soc_after_buy > 0:
                effective_cost = dp / battery.charge_efficiency
                cost_basis = (cost_basis * soc_post_solar + effective_cost * stored) / new_soc_after_buy

            strategy_out = "vsr_charge"

        # 5. Update SoC
        if strategy_out == "vsr_charge":
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
        vsrp_fees[i] = fee
        bid_prices[i] = bid_price
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
    df["vsrp_fee"] = vsrp_fees
    df["bid_price"] = bid_prices
    df["cumulative_revenue"] = df["revenue"].cumsum()
    return df
