"""Aggregation and charting."""

import matplotlib
matplotlib.use("Agg")   # headless rendering
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
from pathlib import Path

from src.config import CHARTS


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    """Return a strategy-level revenue summary."""
    agg_dict = {
        "intervals": ("revenue", "count"),
        "gross_revenue": ("revenue", lambda x: x[x > 0].sum()),
        "costs": ("revenue", lambda x: x[x < 0].sum()),
        "net_revenue": ("revenue", "sum"),
        "energy_delivered": ("energy_delivered", "sum"),
        "grid_bought": ("grid_bought", "sum"),
        "soc_drawn": ("soc_drawn", "sum"),
    }
    if "vsrp_fee" in df.columns:
        agg_dict["vsrp_fees"] = ("vsrp_fee", "sum")

    summary = (
        df.groupby("strategy")
        .agg(**agg_dict)
        .round(2)
    )
    totals = summary.sum(numeric_only=True).round(2)
    totals.name = "TOTAL"
    summary = pd.concat([summary, totals.to_frame().T])
    return summary


def plot_results(df: pd.DataFrame, output_dir: Path | None = None) -> list[Path]:
    """
    Produce two charts:
      1. Revenue over time (cumulative + per-interval bar)
      2. SoC over time with strategy colour-coding

    Returns list of saved file paths.
    """
    output_dir = output_dir or CHARTS
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []

    colour_map = {
        "solar_shift": "#FF9800",
        "arbitrage": "#2196F3",
        "grid_charge": "#9C27B0",
        "vsr_discharge": "#4CAF50",
        "vsr_charge": "#E91E63",
        "hold": "#9E9E9E",
    }
    colours = df["strategy"].map(colour_map).fillna("#9E9E9E")

    # ------------------------------------------------------------------ #
    # Chart 1: Revenue over time
    # ------------------------------------------------------------------ #
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    ax1, ax2 = axes

    ax1.bar(df.index, df["revenue"], color=colours, width=0.018, label="Interval revenue")
    ax1.set_ylabel("Revenue per interval ($)")
    ax1.set_title("Battery Revenue by Interval")
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=c, label=s) for s, c in colour_map.items()
                       if s in df["strategy"].values]
    ax1.legend(handles=legend_elements, loc="upper left")

    ax2.plot(df.index, df["cumulative_revenue"], color="#4CAF50", linewidth=1.5)
    ax2.set_ylabel("Cumulative Revenue ($)")
    ax2.set_title("Cumulative Revenue Over Time")
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax2.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
    fig.autofmt_xdate()
    plt.tight_layout()

    p1 = output_dir / "revenue_over_time.png"
    fig.savefig(p1, dpi=150)
    plt.close(fig)
    paths.append(p1)

    # ------------------------------------------------------------------ #
    # Chart 2: SoC over time
    # ------------------------------------------------------------------ #
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(df.index, df["soc_after_solar"], label="SoC after solar", color="#8BC34A", linewidth=1)
    ax.plot(df.index, df["soc_end"], label="SoC end of interval", color="#3F51B5", linewidth=1)
    ax.set_ylabel("State of Charge (MWh)")
    ax.set_title("Battery State of Charge Over Time")
    ax.legend()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
    fig.autofmt_xdate()
    plt.tight_layout()

    p2 = output_dir / "soc_over_time.png"
    fig.savefig(p2, dpi=150)
    plt.close(fig)
    paths.append(p2)

    return paths
