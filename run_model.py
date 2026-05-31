"""
Entry point for the battery revenue optimisation model.

Usage:
    python run_model.py                          # uses defaults from src/config.py
    python run_model.py --capacity 6 --rate 3   # override battery size
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

from src.config import BatteryConfig, MarketConfig, DATA_PROCESSED, OUTPUTS
from src.data_loader import load_all
from src.revenue_engine import run_revenue_engine
from src.reporting import summarise, plot_results


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Battery revenue optimisation model")
    p.add_argument("--capacity", type=float, default=4.0, help="Battery capacity in MWh")
    p.add_argument("--rate", type=float, default=2.0, help="Max charge/discharge rate in MW")
    p.add_argument("--efficiency", type=float, default=0.85, help="Round-trip efficiency (0-1)")
    p.add_argument("--initial-soc", type=float, default=2.0, help="Initial SoC in MWh")
    p.add_argument("--min-soc-pct", type=float, default=0.0, help="Min SoC as %% of capacity (0-100)")
    p.add_argument("--min-spread", type=float, default=20.0, help="Min arbitrage spread $/MWh")
    p.add_argument("--solar-5min", type=Path, default=None)
    p.add_argument("--solar-30min", type=Path, default=None)
    p.add_argument("--dispatch-prices", type=Path, default=None)
    p.add_argument("--wholesale-prices", type=Path, default=None)
    p.add_argument("--output-dir", type=Path, default=OUTPUTS)
    p.add_argument("--no-charts", action="store_true", help="Skip chart generation")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    battery = BatteryConfig(
        capacity_mwh=args.capacity,
        max_charge_mw=args.rate,
        max_discharge_mw=args.rate,
        round_trip_efficiency=args.efficiency,
        initial_soc_mwh=args.initial_soc,
        min_soc_mwh=args.capacity * args.min_soc_pct / 100.0,
    )
    market = MarketConfig(arbitrage_min_spread=args.min_spread)

    print("Step 1 — Loading and aligning data …")
    df = load_all(
        solar_5min_path=args.solar_5min,
        solar_30min_path=args.solar_30min,
        dispatch_path=args.dispatch_prices,
        wholesale_path=args.wholesale_prices,
    )
    print(f"         {len(df)} intervals loaded ({df.index.min()} -> {df.index.max()})")

    print("Step 2 — Running combined SoC + revenue engine …")
    df = run_revenue_engine(df, battery, market)

    print("Step 3 — Building summary …")
    summary = summarise(df)
    print("\n" + summary.to_string())

    # Save detailed interval log
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    log_path = args.output_dir / "interval_log.csv"
    df.to_csv(log_path)
    print(f"\nInterval log saved → {log_path}")

    if not args.no_charts:
        print("Step 4 — Generating charts …")
        chart_paths = plot_results(df, args.output_dir / "charts")
        for p in chart_paths:
            print(f"         Chart saved → {p}")

    print("\nDone.")


if __name__ == "__main__":
    main()
