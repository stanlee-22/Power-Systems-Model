"""
Entry point for the battery revenue optimisation model.

Usage:
    python run_model.py                                    # defaults: 30 kWh arbitrage
    python run_model.py --strategy solar_shift             # sell free solar at peak
    python run_model.py --strategy vsr --vsrp-margin 0.15  # VSR dispatch (5-min LP)
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

from src.config import BatteryConfig, MarketConfig, VSRConfig, DATA_PROCESSED, OUTPUTS
from src.data_loader import load_all
from src.revenue_engine import run_revenue_engine
from src.vsr_engine import run_vsr_engine
from src.reporting import summarise, plot_results


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Battery revenue optimisation model")
    p.add_argument("--capacity", type=float, default=0.03, help="Battery capacity in MWh (default 0.03 = 30 kWh)")
    p.add_argument("--rate", type=float, default=0.015, help="Max charge/discharge rate in MW (default 0.015 = 15 kW)")
    p.add_argument("--efficiency", type=float, default=0.85, help="Round-trip efficiency (0-1)")
    p.add_argument("--initial-soc", type=float, default=None, help="Initial SoC in MWh (default: 50%% of capacity)")
    p.add_argument("--min-soc-pct", type=float, default=20.0, help="Min SoC as %% of capacity (default 20)")
    p.add_argument("--max-daily-cycles", type=float, default=1.0, help="Max equivalent full cycles per day (default 1)")
    p.add_argument("--min-spread", type=float, default=20.0, help="Min arbitrage spread $/MWh")
    p.add_argument("--strategy", choices=["solar_shift", "arbitrage", "vsr"], default="arbitrage",
                   help="'solar_shift' = sell free solar at peak, "
                        "'arbitrage' = buy low / sell high, "
                        "'vsr' = VSR dispatch via LP optimisation (5-min)")

    # VSR-specific options
    p.add_argument("--vsrp-margin", type=float, default=0.15, help="VSRP aggregator margin (default 0.15 = 15%%)")
    p.add_argument("--forecast-horizon", type=int, default=72, help="LP forecast horizon in intervals (default 72 = 6h)")
    p.add_argument("--reoptimise-every", type=int, default=6, help="Re-solve LP every N intervals (default 6 = 30min)")

    p.add_argument("--solar-5min", type=Path, default=None)
    p.add_argument("--solar-30min", type=Path, default=None)
    p.add_argument("--dispatch-prices", type=Path, default=None)
    p.add_argument("--wholesale-prices", type=Path, default=None)
    p.add_argument("--output-dir", type=Path, default=OUTPUTS)
    p.add_argument("--no-charts", action="store_true", help="Skip chart generation")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    initial_soc = args.initial_soc if args.initial_soc is not None else args.capacity * 0.5
    is_vsr = args.strategy == "vsr"

    battery = BatteryConfig(
        capacity_mwh=args.capacity,
        max_charge_mw=args.rate,
        max_discharge_mw=args.rate,
        round_trip_efficiency=args.efficiency,
        initial_soc_mwh=initial_soc,
        min_soc_mwh=args.capacity * args.min_soc_pct / 100.0,
        max_daily_cycles=args.max_daily_cycles,
        interval_hours=1 / 12 if is_vsr else 0.5,
    )
    market = MarketConfig(arbitrage_min_spread=args.min_spread, strategy=args.strategy)

    resolution = "5min" if is_vsr else "30min"
    print("Step 1 — Loading and aligning data ...")
    df = load_all(
        solar_5min_path=args.solar_5min,
        solar_30min_path=args.solar_30min,
        dispatch_path=args.dispatch_prices,
        wholesale_path=args.wholesale_prices,
        resolution=resolution,
    )
    print(f"         {len(df)} intervals loaded ({df.index.min()} -> {df.index.max()})")
    print(f"         Strategy: {args.strategy} @ {resolution} resolution")
    print(f"         Battery: {args.capacity*1000:.0f} kWh / {args.rate*1000:.0f} kW, "
          f"{args.efficiency*100:.0f}% RT eff, {args.min_soc_pct:.0f}% min SoC, "
          f"{args.max_daily_cycles:.1f} cycles/day")

    print("Step 2 — Running simulation ...")
    if is_vsr:
        vsr = VSRConfig(
            vsrp_margin=args.vsrp_margin,
            forecast_horizon=args.forecast_horizon,
            reoptimise_every=args.reoptimise_every,
        )
        print(f"         VSRP margin: {args.vsrp_margin*100:.0f}%, "
              f"horizon: {args.forecast_horizon} intervals, "
              f"re-optimise every {args.reoptimise_every}")
        df = run_vsr_engine(df, battery, vsr, market)
    else:
        df = run_revenue_engine(df, battery, market)

    print("Step 3 — Building summary ...")
    summary = summarise(df)
    print("\n" + summary.to_string())

    OUTPUTS.mkdir(parents=True, exist_ok=True)
    log_path = args.output_dir / "interval_log.csv"
    df.to_csv(log_path)
    print(f"\nInterval log saved -> {log_path}")

    if not args.no_charts:
        print("Step 4 — Generating charts ...")
        chart_paths = plot_results(df, args.output_dir / "charts")
        for p in chart_paths:
            print(f"         Chart saved -> {p}")

    print("\nDone.")


if __name__ == "__main__":
    main()
