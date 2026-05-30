"""Load and align raw CSVs onto a common 30-minute time index."""

import pandas as pd
from pathlib import Path

from src.config import DATA_RAW


def _read_csv(path: Path, time_col: str, value_col: str, freq: str) -> pd.Series:
    """Read a CSV, parse timestamps, set a DatetimeIndex, and return a Series."""
    df = pd.read_csv(path, parse_dates=[time_col])
    df = df.set_index(time_col).sort_index()
    series = df[value_col].rename(path.stem)
    series.index = series.index.round(freq)   # snap to nearest interval boundary
    series = series[~series.index.duplicated(keep="first")]
    return series


def load_solar_5min(path: Path | None = None) -> pd.Series:
    """5-minute solar generation (MW)."""
    path = path or DATA_RAW / "solar_5min.csv"
    return _read_csv(path, time_col="timestamp", value_col="generation_mw", freq="5min")


def load_solar_30min(path: Path | None = None) -> pd.Series:
    """30-minute solar generation (MW). Used as fallback or additional input."""
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
    solar_5min: pd.Series,
    dispatch_prices: pd.Series,
    wholesale_prices: pd.Series,
    solar_30min: pd.Series | None = None,
) -> pd.DataFrame:
    """
    Align all inputs onto a common 30-minute DatetimeIndex.

    Strategy:
    - 5-min solar  → resample to 30-min by summing energy (MW × 5/60 h → MWh)
      then convert back to average MW for the half-hour.
    - 5-min dispatch prices → resample to 30-min (time-weighted mean).
    - 30-min wholesale prices → use directly; forward-fill any gaps up to 1 period.
    - 30-min solar (if provided) → use as override / fill for the resampled 5-min solar.
    """
    # --- solar ---
    solar_mw_30 = solar_5min.resample("30min").mean()
    if solar_30min is not None:
        solar_mw_30 = solar_mw_30.combine_first(solar_30min)

    # --- dispatch price ---
    dispatch_30 = dispatch_prices.resample("30min").mean()

    # --- build on the intersection of all three ---
    df = pd.DataFrame({
        "solar_mw": solar_mw_30,
        "dispatch_price": dispatch_30,
        "wholesale_price": wholesale_prices,
    })

    # Forward-fill wholesale prices up to one interval (handles NEM settlement lag)
    df["wholesale_price"] = df["wholesale_price"].ffill(limit=1)

    # Drop rows where critical columns are still missing
    df = df.dropna(subset=["dispatch_price", "wholesale_price"])

    # Clip negative solar readings to zero
    df["solar_mw"] = df["solar_mw"].clip(lower=0).fillna(0)

    return df


def load_all(
    solar_5min_path: Path | None = None,
    solar_30min_path: Path | None = None,
    dispatch_path: Path | None = None,
    wholesale_path: Path | None = None,
) -> pd.DataFrame:
    """Convenience wrapper: load raw files and return the aligned master frame."""
    solar_5 = load_solar_5min(solar_5min_path)
    dispatch = load_dispatch_prices(dispatch_path)
    wholesale = load_wholesale_prices(wholesale_path)

    solar_30 = None
    path_30 = solar_30min_path or DATA_RAW / "solar_30min.csv"
    if path_30.exists():
        solar_30 = load_solar_30min(path_30)

    return build_master_frame(solar_5, dispatch, wholesale, solar_30)
