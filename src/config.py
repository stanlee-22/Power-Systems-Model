"""Battery and model configuration."""

from dataclasses import dataclass, field
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
    min_soc_mwh: float = 0.0           # floor (can set to e.g. 10 % of capacity)
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


# ---------------------------------------------------------------------------
# Arbitrage / dispatch settings
# ---------------------------------------------------------------------------

@dataclass
class MarketConfig:
    # Minimum price spread ($/MWh) required before executing an arbitrage trade
    arbitrage_min_spread: float = 20.0
    # Fraction of available SoC dispatched per interval (0-1)
    dispatch_fraction: float = 1.0
