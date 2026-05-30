"""
Generate synthetic CSVs for smoke-testing without real data.

Produces one full calendar year (2025) of plausible values:
  - data/raw/solar_5min.csv
  - data/raw/solar_30min.csv
  - data/raw/dispatch_prices.csv
  - data/raw/wholesale_prices.csv
"""

import numpy as np
import pandas as pd
from pathlib import Path

SEED = 42
rng = np.random.default_rng(SEED)

OUT = Path("data/raw")
OUT.mkdir(parents=True, exist_ok=True)

start = "2025-01-01"
end   = "2025-12-31 23:55"

# ------------------------------------------------------------------ #
# 5-minute index
# ------------------------------------------------------------------ #
idx5 = pd.date_range(start, end, freq="5min")

# Solar: zero at night, bell-curve during the day, seasonal amplitude
hour_of_day = idx5.hour + idx5.minute / 60
day_of_year = idx5.dayofyear
solar_amplitude = 2.5 + 1.0 * np.sin(2 * np.pi * (day_of_year - 80) / 365)  # 1.5–3.5 MW peak
solar_base = solar_amplitude * np.maximum(
    np.sin(np.pi * (hour_of_day - 6) / 12), 0
) ** 1.5
solar_noise = rng.normal(0, 0.05, len(idx5))
solar_5 = np.clip(solar_base + solar_noise, 0, None)

pd.DataFrame({"timestamp": idx5, "generation_mw": solar_5}).to_csv(
    OUT / "solar_5min.csv", index=False
)
print(f"Wrote {OUT / 'solar_5min.csv'}  ({len(idx5):,} rows)")

# ------------------------------------------------------------------ #
# 30-minute index
# ------------------------------------------------------------------ #
idx30 = pd.date_range(start, end, freq="30min")
hour_of_day30 = idx30.hour + idx30.minute / 60
day_of_year30 = idx30.dayofyear
solar_amplitude30 = 2.5 + 1.0 * np.sin(2 * np.pi * (day_of_year30 - 80) / 365)
solar_30 = np.clip(
    solar_amplitude30 * np.maximum(np.sin(np.pi * (hour_of_day30 - 6) / 12), 0) ** 1.5
    + rng.normal(0, 0.05, len(idx30)),
    0, None,
)
pd.DataFrame({"timestamp": idx30, "generation_mw": solar_30}).to_csv(
    OUT / "solar_30min.csv", index=False
)
print(f"Wrote {OUT / 'solar_30min.csv'}  ({len(idx30):,} rows)")

# ------------------------------------------------------------------ #
# 5-minute dispatch prices
# ------------------------------------------------------------------ #
# Base price with morning/evening peaks, random spikes
base_price = 80 + 40 * np.sin(2 * np.pi * (hour_of_day - 7) / 24)
spike_mask = rng.random(len(idx5)) < 0.005   # ~0.5% spike probability
spikes = rng.uniform(200, 800, len(idx5)) * spike_mask
dispatch_price = np.clip(base_price + rng.normal(0, 10, len(idx5)) + spikes, -50, 15000)

pd.DataFrame({"timestamp": idx5, "price_per_mwh": dispatch_price}).to_csv(
    OUT / "dispatch_prices.csv", index=False
)
print(f"Wrote {OUT / 'dispatch_prices.csv'}  ({len(idx5):,} rows)")

# ------------------------------------------------------------------ #
# 30-minute wholesale prices
# ------------------------------------------------------------------ #
base_w = 75 + 35 * np.sin(2 * np.pi * (hour_of_day30 - 7) / 24)
spike_mask_w = rng.random(len(idx30)) < 0.005
spikes_w = rng.uniform(150, 600, len(idx30)) * spike_mask_w
wholesale_price = np.clip(base_w + rng.normal(0, 8, len(idx30)) + spikes_w, -50, 15000)

pd.DataFrame({"timestamp": idx30, "price_per_mwh": wholesale_price}).to_csv(
    OUT / "wholesale_prices.csv", index=False
)
print(f"Wrote {OUT / 'wholesale_prices.csv'}  ({len(idx30):,} rows)")

print("\nSample data generation complete.")
