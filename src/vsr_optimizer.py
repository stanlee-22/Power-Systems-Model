"""LP-based dispatch optimizer for VSR bidding.

Solves a rolling-horizon linear program to find the optimal charge/discharge
schedule over a forecast window. The shadow price on the SoC lower-bound
constraint at t=0 gives the battery's opportunity cost — its bid price.
"""

import numpy as np
from scipy.optimize import linprog

from src.config import BatteryConfig


# ---------------------------------------------------------------------------
# Price forecast
# ---------------------------------------------------------------------------

INTERVALS_PER_DAY = 288  # 24h × 12 intervals/h at 5-min


def build_price_forecast(
    price_history: np.ndarray,
    current_idx: int,
    horizon: int,
    method: str = "week_ago",
) -> np.ndarray:
    """
    Build a forward price forecast of length `horizon`.

    "week_ago": use prices from exactly 7 days prior at the same time of day.
    "rolling_mean": use the trailing 24h mean as a flat forecast.
    """
    week_offset = 7 * INTERVALS_PER_DAY  # 2016 intervals

    if method == "week_ago":
        start = current_idx - week_offset
        end = start + horizon
        if start >= 0 and end <= len(price_history):
            return price_history[start:end].copy()
        # Fallback: use trailing 24h mean
        lookback_start = max(0, current_idx - INTERVALS_PER_DAY)
        mean_price = np.mean(price_history[lookback_start:current_idx + 1])
        return np.full(horizon, mean_price)

    # rolling_mean
    lookback_start = max(0, current_idx - INTERVALS_PER_DAY)
    mean_price = np.mean(price_history[lookback_start:current_idx + 1])
    return np.full(horizon, mean_price)


# ---------------------------------------------------------------------------
# LP dispatch solver
# ---------------------------------------------------------------------------

def solve_dispatch_lp(
    forecast_prices: np.ndarray,
    current_soc: float,
    battery: BatteryConfig,
    daily_discharged: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Solve the optimal charge/discharge schedule over the forecast horizon.

    Variables layout: x = [charge_0 .. charge_{H-1}, discharge_0 .. discharge_{H-1}]

    Returns (charge_plan, discharge_plan, shadow_price).
    shadow_price is the opportunity cost of stored energy at t=0.
    """
    H = len(forecast_prices)
    eta_c = battery.charge_efficiency
    eta_d = battery.discharge_efficiency
    max_c = battery.max_charge_per_interval
    max_d = battery.max_discharge_per_interval
    cap = battery.capacity_mwh
    min_soc = battery.min_soc_mwh
    cycle_headroom = max(battery.daily_throughput_limit - daily_discharged, 0.0)

    # Objective: maximise sum(price[t] * discharge[t] * eta_d - price[t] * charge[t] / eta_c)
    # linprog minimises, so negate.
    c = np.zeros(2 * H)
    for t in range(H):
        c[t] = forecast_prices[t] / eta_c        # cost of charging (positive = bad)
        c[H + t] = -forecast_prices[t] * eta_d   # revenue from discharging (negative = good for min)

    # Variable bounds
    bounds = [(0, max_c)] * H + [(0, max_d)] * H

    # Inequality constraints: A_ub @ x <= b_ub
    # SoC at end of interval t:
    #   SoC(t) = soc_0 + sum_{j=0}^{t} (charge[j] * eta_c - discharge[j])
    # Upper bound: SoC(t) <= capacity
    # Lower bound: SoC(t) >= min_soc  =>  -SoC(t) <= -min_soc

    A_rows = []
    b_rows = []

    for t in range(H):
        # SoC upper bound: sum(charge[0:t+1]*eta_c) - sum(discharge[0:t+1]) <= cap - soc_0
        row_upper = np.zeros(2 * H)
        for j in range(t + 1):
            row_upper[j] = eta_c        # charge contributes to SoC
            row_upper[H + j] = -1.0     # discharge reduces SoC
        A_rows.append(row_upper)
        b_rows.append(cap - current_soc)

        # SoC lower bound: -sum(charge[0:t+1]*eta_c) + sum(discharge[0:t+1]) <= soc_0 - min_soc
        row_lower = np.zeros(2 * H)
        for j in range(t + 1):
            row_lower[j] = -eta_c
            row_lower[H + j] = 1.0
        A_rows.append(row_lower)
        b_rows.append(current_soc - min_soc)

    # Daily cycle cap: sum(discharge) <= cycle_headroom
    row_cycle = np.zeros(2 * H)
    row_cycle[H:] = 1.0
    A_rows.append(row_cycle)
    b_rows.append(cycle_headroom)

    A_ub = np.array(A_rows)
    b_ub = np.array(b_rows)

    result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")

    if not result.success:
        return np.zeros(H), np.zeros(H), 0.0

    charge_plan = result.x[:H]
    discharge_plan = result.x[H:]

    # Shadow price: dual on the SoC lower-bound constraint at t=0
    # The lower-bound constraints are at odd indices (1, 3, 5, ...)
    # Index 1 is the SoC lower bound for t=0
    marginals = result.ineqlin.marginals
    shadow_price = marginals[1] if len(marginals) > 1 else 0.0

    return charge_plan, discharge_plan, shadow_price
