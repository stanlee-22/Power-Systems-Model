"""Load and align raw CSVs onto a common 30-minute time index.

Post-5MS note (AEMO, effective 1 Oct 2021):
  There is no separate 30-min TRADINGPRICE. wholesale_prices.csv sourced via
  fetch_aemo_data.py is the 30-min mean of DISPATCHPRICE. The model treats them
  identically regardless of source.

Solar source priority:
  1. solar_5min.csv  (DISPATCH_UNIT_SCADA output for a specific DUID)
  2. solar_30min.csv (ROOFTOP_PV_ACTUAL, or resampled SCADA)
  At least one must be present.
"""

import pandas as pd
from pathlib import Path

from src.config import DATA_RAW


def _read_csv(path: Path, time_col: str, value_col: str, freq: str) -> pd.Series:
    """Read a CSV, parse timestamps, set a DatetimeIndex, and return a Series."""
    df = pd.read_csv(path, parse_dates=[time_col])
    df = df.set_index(time_col).sort_index()
    series = df[value_col].rename(path.stem)
    series.index = series.index.round(freq)
    series = series[~series.index.duplicated(keep="first")]
    return series


def load_solar_5min(path: Path | None = None) -> pd.Series:
    """5-minute solar generation (MW)."""
    path = path or DATA_RAW / "solar_5min.csv"
    return _read_csv(path, time_col="timestamp", value_col="generation_mw", freq="5min")


def load_solar_30min(path: Path | None = None) -> pd.Series:
    """30-minute solar generation (MW)."""
    path = path or DATA_RAW / "solar_30min.csv"
    return _read_csv(path, time_col="timestamp", value_col="generation_mw", freq="30min")


def load_dispatch_prices(path: Path | None = None) -> pd.Series:
    """5-minute dispatch prices ($/MWh)."""
    path = path or DATA_RAW / "dispatch_prices.csv"
    return _read_csv(path, time_col="timestamp", value_col="price_per_mwh", freq="5min")


def load_wholesale_prices(path: Path | None = None) -> pd.Series:
    """30-minute wholesale market prices ($/MWh)."""
    path = path or DATA_RAW / "wholesale_prices.csv"
    return _read_csv(path, time_col="timestamp", value_col="price_per_mwh", freq="30min")


def build_master_frame(
    dispatch_prices: pd.Series,
    wholesale_prices: pd.Series,
    solar_5min: pd.Series | None = None,
    solar_30min: pd.Series | None = None,
) -> pd.DataFrame:
    """
    Align all inputs onto a common 30-minute DatetimeIndex.

    - 5-min solar  → resample to 30-min mean MW.
    - 5-min dispatch prices → resample to 30-min mean.
    - 30-min wholesale prices → use directly; forward-fill gaps ≤1 period.
    - solar_30min fills any gaps left after resampling solar_5min.
    """
    if solar_5min is None and solar_30min is None:
        raise ValueError("Provide at least one of solar_5min or solar_30min")

    # --- solar ---
    if solar_5min is not None:
        solar_mw_30 = solar_5min.resample("30min").mean()
        if solar_30min is not None:
            solar_mw_30 = solar_mw_30.combine_first(solar_30min)
    else:
        solar_mw_30 = solar_30min

    # --- dispatch price (5-min → 30-min) ---
    dispatch_30 = dispatch_prices.resample("30min").mean()

    df = pd.DataFrame({
        "solar_mw": solar_mw_30,
        "dispatch_price": dispatch_30,
        "wholesale_price": wholesale_prices,
    })

    df["wholesale_price"] = df["wholesale_price"].ffill(limit=1)
    df = df.dropna(subset=["dispatch_price", "wholesale_price"])
    df["solar_mw"] = df["solar_mw"].clip(lower=0).fillna(0)

    return df


def build_master_frame_5min(
    dispatch_prices: pd.Series,
    solar_5min: pd.Series | None = None,
    solar_30min: pd.Series | None = None,
) -> pd.DataFrame:
    """
    Build a 5-minute resolution master frame for VSR dispatch modelling.

    - Dispatch prices kept at native 5-min resolution (no resampling).
    - 30-min solar forward-filled across six 5-min sub-intervals.
    - No wholesale_price column — VSR uses dispatch prices directly.
    """
    if solar_5min is None and solar_30min is None:
        raise ValueError("Provide at least one of solar_5min or solar_30min")

    if solar_5min is not None:
        solar_mw = solar_5min
        if solar_30min is not None:
            solar_mw = solar_mw.combine_first(
                solar_30min.resample("5min").ffill()
            )
    else:
        solar_mw = solar_30min.resample("5min").ffill()

    df = pd.DataFrame({
        "solar_mw": solar_mw,
        "dispatch_price": dispatch_prices,
    })

    df = df.dropna(subset=["dispatch_price"])
    df["solar_mw"] = df["solar_mw"].clip(lower=0).fillna(0)
    return df


def load_all(
    solar_5min_path: Path | None = None,
    solar_30min_path: Path | None = None,
    dispatch_path: Path | None = None,
    wholesale_path: Path | None = None,
    resolution: str = "30min",
) -> pd.DataFrame:
    """
    Convenience wrapper: load raw files and return the aligned master frame.

    resolution="30min" (default): 30-min frame with dispatch, wholesale, solar.
    resolution="5min": 5-min frame with dispatch + solar only (for VSR).
    """
    path_5 = solar_5min_path or DATA_RAW / "solar_5min.csv"
    path_30 = solar_30min_path or DATA_RAW / "solar_30min.csv"

    solar_5: pd.Series | None = None
    solar_30: pd.Series | None = None

    if path_5.exists():
        solar_5 = load_solar_5min(path_5)
    if path_30.exists():
        solar_30 = load_solar_30min(path_30)

    if solar_5 is None and solar_30 is None:
        raise FileNotFoundError(
            f"No solar data found. Expected {path_5} or {path_30}."
        )

    dispatch = load_dispatch_prices(dispatch_path)

    if resolution == "5min":
        return build_master_frame_5min(dispatch, solar_5, solar_30)

    wholesale = load_wholesale_prices(wholesale_path)
    return build_master_frame(dispatch, wholesale, solar_5, solar_30)
