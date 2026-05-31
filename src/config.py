"""Battery and model configuration."""

from dataclasses import dataclass
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

DATA_RAW = Path("data/raw")
DATA_PROCESSED = Path("data/processed")
OUTPUTS = Path("outputs")
CHARTS = OUTPUTS / "charts"


# ---------------------------------------------------------------------------
# Battery parameters
# ---------------------------------------------------------------------------

@dataclass
class BatteryConfig:
    capacity_mwh: float = 4.0          # usable energy capacity
    max_charge_mw: float = 2.0         # max charge power
    max_discharge_mw: float = 2.0      # max discharge power
    round_trip_efficiency: float = 0.85  # e.g. 85 %
    initial_soc_mwh: float = 2.0       # starting state of charge
    min_soc_mwh: float = 0.0           # floor (e.g. 20 % of capacity)
    max_daily_cycles: float = 1.0      # max equivalent full cycles per day
    interval_hours: float = 0.5        # 30-minute intervals

    @property
    def charge_efficiency(self) -> float:
        """One-way charge efficiency (sqrt of round-trip)."""
        return self.round_trip_efficiency ** 0.5

    @property
    def discharge_efficiency(self) -> float:
        """One-way discharge efficiency (sqrt of round-trip)."""
        return self.round_trip_efficiency ** 0.5

    @property
    def max_charge_per_interval(self) -> float:
        return self.max_charge_mw * self.interval_hours

    @property
    def max_discharge_per_interval(self) -> float:
        return self.max_discharge_mw * self.interval_hours

    @property
    def daily_throughput_limit(self) -> float:
        """Max MWh discharged per calendar day."""
        return self.capacity_mwh * self.max_daily_cycles


# ---------------------------------------------------------------------------
# Market / strategy settings
# ---------------------------------------------------------------------------

@dataclass
class MarketConfig:
    arbitrage_min_spread: float = 20.0
    strategy: str = "arbitrage"


# ---------------------------------------------------------------------------
# VSR (Voluntarily Scheduled Resource) dispatch settings
# ---------------------------------------------------------------------------

@dataclass
class VSRConfig:
    vsrp_margin: float = 0.15              # aggregator's cut (15 %)
    forecast_horizon: int = 72             # intervals to look ahead (72 × 5 min = 6 h)
    reoptimise_every: int = 6              # re-solve LP every N intervals (6 × 5 min = 30 min)
    forecast_method: str = "week_ago"      # "week_ago" or "rolling_mean"
