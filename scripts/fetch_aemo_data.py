"""
Download and clean AEMO market data from the NEMWeb MMSDM archive.

Covers a configurable year/month range (defaults to all of 2025).

Output CSVs written to data/raw/ — matching the schema in src/data_loader.py:

  dispatch_prices.csv   → timestamp (5-min, end of interval), price_per_mwh
  wholesale_prices.csv  → timestamp (30-min, derived by resampling dispatch), price_per_mwh
  solar_30min.csv       → timestamp (30-min), generation_mw   [ROOFTOP_PV_ACTUAL]
  solar_5min.csv        → timestamp (5-min), generation_mw    [DISPATCH_UNIT_SCADA, only if --duid given]

Post-5MS note (effective 1 Oct 2021):
  There is no longer a separate 30-minute TRADINGPRICE table.
  DISPATCHPRICE IS the settlement price. The wholesale_prices.csv produced
  here is a 30-min mean of the 5-min dispatch prices — used by the revenue
  engine as the arbitrage reference.

Network note:
  AEMO's NEMWeb blocks requests from cloud provider IP ranges.
  Run this script on your LOCAL machine, not inside a cloud shell.

  Two ways to run it:

  Option A — direct download (run locally):
    python scripts/fetch_aemo_data.py --region NSW1

  Option B — manual download + local parse:
    1. In your browser, go to:
         https://nemweb.com.au/Data_Archive/Wholesale_Electricity/MMSDM/
         2025/MMSDM_2025_01/MMSDM_Historical_Data_SQLLoader/DATA/
    2. Download the DISPATCHPRICE and ROOFTOP_PV_ACTUAL ZIPs
       (named PUBLIC_ARCHIVE#...# or PUBLIC_DVD_...)
       (repeat for each month)
    3. Place the ZIPs in any directory, then run:
         python scripts/fetch_aemo_data.py --region NSW1 --local-zip-dir /path/to/zips

Usage examples:
  python scripts/fetch_aemo_data.py --region NSW1
  python scripts/fetch_aemo_data.py --region VIC1 --year 2025 --months 1 2 3
  python scripts/fetch_aemo_data.py --region QLD1 --duid SOLARSF1  (utility solar)
  python scripts/fetch_aemo_data.py --region NSW1 --local-zip-dir ~/Downloads/aemo_zips

NEM regions: NSW1  VIC1  QLD1  SA1  TAS1
"""

import argparse
import io
import logging
import time
import zipfile
from pathlib import Path

import pandas as pd
import requests

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MMSDM_DIR = (
    "https://nemweb.com.au/Data_Archive/Wholesale_Electricity/MMSDM"
    "/{year}/MMSDM_{year}_{month:02d}"
    "/MMSDM_Historical_Data_SQLLoader/DATA"
)

# 2025+ naming: PUBLIC_ARCHIVE#TABLE#FILE01#YYYYMM010000.zip
# Pre-2025 naming: PUBLIC_DVD_TABLE_YYYYMM010000.zip
MMSDM_ARCHIVE_FMT = MMSDM_DIR + "/PUBLIC_ARCHIVE%2523{table}%2523FILE01%2523{year}{month:02d}010000.zip"
MMSDM_DVD_FMT = MMSDM_DIR + "/PUBLIC_DVD_{table}_{year}{month:02d}010000.zip"

NEM_REGIONS = {"NSW1", "VIC1", "QLD1", "SA1", "TAS1"}

OUT_DIR = Path("data/raw")

# AEMO's servers reject requests without a browser-like User-Agent.
_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Referer": "https://www.aemo.com.au/",
    "Accept": "application/zip,application/octet-stream,*/*",
}


# ---------------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------------

def _download(url: str, retries: int = 4) -> bytes:
    """GET with exponential backoff.  Returns raw bytes.  Raises immediately on 404."""
    delay = 2
    for attempt in range(retries + 1):
        try:
            log.info("  GET %s", url)
            r = requests.get(url, headers=_HEADERS, timeout=120)
            r.raise_for_status()
            return r.content
        except requests.HTTPError as exc:
            if exc.response is not None and exc.response.status_code == 404:
                raise
            if attempt == retries:
                raise
            log.warning("  Attempt %d failed (%s) — retrying in %ds", attempt + 1, exc, delay)
            time.sleep(delay)
            delay *= 2
        except requests.RequestException as exc:
            if attempt == retries:
                raise
            log.warning("  Attempt %d failed (%s) — retrying in %ds", attempt + 1, exc, delay)
            time.sleep(delay)
            delay *= 2
    raise RuntimeError("unreachable")


def _load_zip(
    table: str,
    year: int,
    month: int,
    local_zip_dir: Path | None,
) -> bytes:
    """
    Return raw ZIP bytes — from a local file if --local-zip-dir was given,
    otherwise download from NEMWeb.

    Checks both naming conventions (ARCHIVE# and DVD_) for local files.
    """
    candidates = [
        f"PUBLIC_ARCHIVE#DISPATCHPRICE#FILE01#{year}{month:02d}010000.zip"
        if table == "DISPATCHPRICE" else
        f"PUBLIC_ARCHIVE#{table}#FILE01#{year}{month:02d}010000.zip",
        f"PUBLIC_DVD_{table}_{year}{month:02d}010000.zip",
    ]
    if local_zip_dir is not None:
        for filename in candidates:
            path = local_zip_dir / filename
            if path.exists():
                log.info("  Reading local file %s", path)
                return path.read_bytes()
        raise FileNotFoundError(
            f"Expected local ZIP not found in {local_zip_dir}\n"
            f"Tried: {', '.join(candidates)}"
        )
    return _download_with_fallback(table, year, month)


def _download_with_fallback(table: str, year: int, month: int) -> bytes:
    """Try each candidate URL; return bytes from the first that succeeds."""
    urls = _mmsdm_urls(table, year, month)
    last_exc = None
    for url in urls:
        try:
            return _download(url)
        except requests.HTTPError as exc:
            if exc.response is not None and exc.response.status_code == 404:
                log.info("  404 for %s — trying next format", url.split("/")[-1])
                last_exc = exc
                continue
            raise
    raise last_exc


def _mmsdm_urls(table: str, year: int, month: int) -> list[str]:
    """Return candidate URLs to try — new ARCHIVE format first, then legacy DVD."""
    return [
        MMSDM_ARCHIVE_FMT.format(table=table, year=year, month=month),
        MMSDM_DVD_FMT.format(table=table, year=year, month=month),
    ]


# ---------------------------------------------------------------------------
# AEMO CSV parser
# ---------------------------------------------------------------------------

def _parse_aemo_zip(raw_bytes: bytes, pkg: str, tbl: str) -> pd.DataFrame:
    """
    Extract one logical table (pkg, tbl) from an AEMO MMSDM ZIP file.

    AEMO CSV format:
      C,…          version/comment rows  → skip
      I,PKG,TBL,v,col1,col2,…           → column header for that table
      D,PKG,TBL,v,val1,val2,…           → data rows
      C,END OF REPORT,…                 → skip

    Returns a DataFrame with columns taken from the matching I-row.
    Raises ValueError if the table is not found.
    """
    frames: list[pd.DataFrame] = []

    with zipfile.ZipFile(io.BytesIO(raw_bytes)) as zf:
        csv_names = [n for n in zf.namelist() if n.lower().endswith(".csv")]
        if not csv_names:
            raise ValueError("No CSV files found inside ZIP")

        for csv_name in csv_names:
            text = zf.read(csv_name).decode("utf-8", errors="replace")
            lines = text.splitlines()

            header: list[str] | None = None
            rows: list[list[str]] = []

            for line in lines:
                parts = line.split(",")
                if not parts:
                    continue
                row_type = parts[0].strip().upper()

                if row_type == "I" and len(parts) > 3:
                    # I,PKG,TBL,version,col1,col2,…
                    if parts[1].strip().upper() == pkg and parts[2].strip().upper() == tbl:
                        header = [c.strip() for c in parts[4:]]

                elif row_type == "D" and len(parts) > 3:
                    if parts[1].strip().upper() == pkg and parts[2].strip().upper() == tbl:
                        if header is not None:
                            rows.append([c.strip().strip('"') for c in parts[4:]])

            if header and rows:
                # Truncate/pad rows to match header length
                n = len(header)
                trimmed = [r[:n] + [""] * max(0, n - len(r)) for r in rows]
                frames.append(pd.DataFrame(trimmed, columns=header))

    if not frames:
        raise ValueError(f"Table {pkg},{tbl} not found in ZIP")

    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# Per-table fetch & clean functions
# ---------------------------------------------------------------------------

def fetch_dispatch_prices(
    year: int,
    months: list[int],
    region: str,
    out_dir: Path,
    local_zip_dir: Path | None = None,
) -> pd.DataFrame:
    """
    Download DISPATCHPRICE for each month, filter by region and
    INTERVENTION=0, return (and save) a clean DataFrame.

    Columns returned: timestamp (end of 5-min interval), price_per_mwh
    """
    chunks: list[pd.DataFrame] = []

    for m in months:
        try:
            raw = _load_zip("DISPATCHPRICE", year, m, local_zip_dir)
        except (requests.HTTPError, FileNotFoundError) as e:
            log.warning("  Skipping %d-%02d dispatch prices: %s", year, m, e)
            continue

        df = _parse_aemo_zip(raw, "DISPATCH", "PRICE")

        # Keep non-intervention rows for the requested region only
        df = df[df["REGIONID"] == region]
        if "INTERVENTION" in df.columns:
            df = df[df["INTERVENTION"].astype(str) == "0"]

        df["timestamp"] = pd.to_datetime(df["SETTLEMENTDATE"], dayfirst=False)
        df["price_per_mwh"] = pd.to_numeric(df["RRP"], errors="coerce")
        chunks.append(df[["timestamp", "price_per_mwh"]])
        log.info("  %d-%02d: %d dispatch price rows", year, m, len(df))

    if not chunks:
        raise RuntimeError("No dispatch price data downloaded")

    combined = (
        pd.concat(chunks)
        .sort_values("timestamp")
        .drop_duplicates("timestamp")
        .reset_index(drop=True)
    )

    out_path = out_dir / "dispatch_prices.csv"
    combined.to_csv(out_path, index=False)
    log.info("Saved %s  (%d rows)", out_path, len(combined))
    return combined


def derive_wholesale_prices(dispatch_df: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    """
    Derive 30-min wholesale prices by resampling 5-min dispatch prices.

    Post-5MS (Oct 2021) there is no separate TRADINGPRICE.
    The 30-min mean of the dispatch price is a close proxy.
    """
    ts = dispatch_df.set_index("timestamp")["price_per_mwh"]
    ts.index = pd.to_datetime(ts.index)
    wholesale = ts.resample("30min").mean().reset_index()
    wholesale.columns = ["timestamp", "price_per_mwh"]
    wholesale = wholesale.dropna()

    out_path = out_dir / "wholesale_prices.csv"
    wholesale.to_csv(out_path, index=False)
    log.info("Saved %s  (%d rows)", out_path, len(wholesale))
    return wholesale


def fetch_rooftop_solar(
    year: int,
    months: list[int],
    region: str,
    out_dir: Path,
    local_zip_dir: Path | None = None,
) -> pd.DataFrame:
    """
    Download ROOFTOP_PV_ACTUAL (30-min aggregate rooftop solar by region).

    The table uses a BANDWIDTH column to separate confidence bands.
    We select the MEASUREMENT type where available, falling back to SATELLITE.

    Columns returned: timestamp, generation_mw
    """
    chunks: list[pd.DataFrame] = []

    for m in months:
        try:
            raw = _load_zip("ROOFTOP_PV_ACTUAL", year, m, local_zip_dir)
        except (requests.HTTPError, FileNotFoundError) as e:
            log.warning("  Skipping %d-%02d rooftop solar: %s", year, m, e)
            continue

        try:
            df = _parse_aemo_zip(raw, "ROOFTOP_PV", "ACTUAL")
        except ValueError:
            df = _parse_aemo_zip(raw, "ROOFTOP", "ACTUAL")
        df = df[df["REGIONID"] == region]

        # Prefer MEASUREMENT type; fall back to SATELLITE
        if "TYPE" in df.columns:
            meas = df[df["TYPE"] == "MEASUREMENT"]
            df = meas if not meas.empty else df

        ts_col = "INTERVAL_DATETIME" if "INTERVAL_DATETIME" in df.columns else "SETTLEMENTDATE"
        df["timestamp"] = pd.to_datetime(df[ts_col], dayfirst=False)
        pwr_col = "POWER" if "POWER" in df.columns else "POWER_MW"
        df["generation_mw"] = pd.to_numeric(df[pwr_col], errors="coerce").clip(lower=0)
        chunks.append(df[["timestamp", "generation_mw"]])
        log.info("  %d-%02d: %d rooftop solar rows", year, m, len(df))

    if not chunks:
        raise RuntimeError("No rooftop solar data downloaded")

    combined = (
        pd.concat(chunks)
        .sort_values("timestamp")
        .drop_duplicates("timestamp")
        .reset_index(drop=True)
    )

    out_path = out_dir / "solar_30min.csv"
    combined.to_csv(out_path, index=False)
    log.info("Saved %s  (%d rows)", out_path, len(combined))
    return combined


def fetch_unit_scada(
    year: int,
    months: list[int],
    duid: str,
    out_dir: Path,
    local_zip_dir: Path | None = None,
) -> pd.DataFrame:
    """
    Download DISPATCH_UNIT_SCADA for a specific DUID (utility-scale solar plant).

    Writes both solar_5min.csv and solar_30min.csv (resampled).
    Warning: these monthly ZIP files are large (~100–300 MB each).
    """
    chunks: list[pd.DataFrame] = []

    for m in months:
        try:
            raw = _load_zip("DISPATCH_UNIT_SCADA", year, m, local_zip_dir)
        except (requests.HTTPError, FileNotFoundError) as e:
            log.warning("  Skipping %d-%02d unit SCADA: %s", year, m, e)
            continue

        df = _parse_aemo_zip(raw, "DISPATCH", "UNIT_SCADA")
        df = df[df["DUID"] == duid]
        if df.empty:
            log.warning("  %d-%02d: DUID %s not found in SCADA data", year, m, duid)
            continue

        df["timestamp"] = pd.to_datetime(df["SETTLEMENTDATE"], dayfirst=False)
        df["generation_mw"] = pd.to_numeric(df["SCADAVALUE"], errors="coerce").clip(lower=0)
        chunks.append(df[["timestamp", "generation_mw"]])
        log.info("  %d-%02d: %d SCADA rows for %s", year, m, len(df), duid)

    if not chunks:
        raise RuntimeError(f"No SCADA data found for DUID {duid}")

    df5 = (
        pd.concat(chunks)
        .sort_values("timestamp")
        .drop_duplicates("timestamp")
        .reset_index(drop=True)
    )

    out5 = out_dir / "solar_5min.csv"
    df5.to_csv(out5, index=False)
    log.info("Saved %s  (%d rows)", out5, len(df5))

    # Also write a 30-min resampled version
    df30 = (
        df5.set_index("timestamp")["generation_mw"]
        .resample("30min").mean()
        .clip(lower=0)
        .reset_index()
    )
    df30.columns = ["timestamp", "generation_mw"]
    out30 = out_dir / "solar_30min.csv"
    df30.to_csv(out30, index=False)
    log.info("Saved %s  (%d rows)", out30, len(df30))

    return df5


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description="Fetch AEMO market data from NEMWeb MMSDM archive")
    p.add_argument(
        "--region",
        required=True,
        choices=sorted(NEM_REGIONS),
        help="NEM region, e.g. NSW1, VIC1, QLD1, SA1, TAS1",
    )
    p.add_argument("--year", type=int, default=2025, help="Calendar year (default 2025)")
    p.add_argument(
        "--months",
        nargs="+",
        type=int,
        default=list(range(1, 13)),
        metavar="M",
        help="Months to fetch (default: 1–12).  Example: --months 1 2 3",
    )
    p.add_argument(
        "--duid",
        default=None,
        help=(
            "DUID of a specific utility-scale solar unit (e.g. SOLARSF1). "
            "If omitted, rooftop PV aggregate is used instead. "
            "WARNING: DISPATCH_UNIT_SCADA files are ~100–300 MB per month."
        ),
    )
    p.add_argument("--out-dir", type=Path, default=OUT_DIR, help="Output directory for CSVs")
    p.add_argument(
        "--local-zip-dir",
        type=Path,
        default=None,
        metavar="DIR",
        help=(
            "Parse ZIPs from a local directory instead of downloading. "
            "Files must follow AEMO naming: PUBLIC_DVD_{TABLE}_{YYYYMM}010000.zip. "
            "Use this if NEMWeb is inaccessible (e.g. from a cloud environment)."
        ),
    )
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    local = args.local_zip_dir

    if local:
        log.info("=== Local ZIP mode — reading from %s ===", local)

    log.info("=== Fetching dispatch prices (%s) ===", args.region)
    dispatch_df = fetch_dispatch_prices(args.year, args.months, args.region, args.out_dir, local)

    log.info("=== Deriving 30-min wholesale prices from dispatch prices ===")
    derive_wholesale_prices(dispatch_df, args.out_dir)

    if args.duid:
        log.info("=== Fetching DISPATCH_UNIT_SCADA for DUID=%s ===", args.duid)
        fetch_unit_scada(args.year, args.months, args.duid, args.out_dir, local)
    else:
        log.info("=== Fetching rooftop PV actual (%s) ===", args.region)
        fetch_rooftop_solar(args.year, args.months, args.region, args.out_dir, local)

    log.info("=== All done — CSVs written to %s ===", args.out_dir)


if __name__ == "__main__":
    main()
