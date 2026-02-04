"""
Download historical aggTrades data from Binance (data.binance.vision)
for BTCUSDT and ETHUSDT.
"""

import os
import io
import zipfile
import requests
import pandas as pd
from datetime import datetime, timedelta

BASE_URL = "https://data.binance.vision/data/spot/daily/aggTrades"
SYMBOLS = ["BTCUSDT", "ETHUSDT"]
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")

COLUMNS = [
    "agg_trade_id",
    "price",
    "quantity",
    "first_trade_id",
    "last_trade_id",
    "timestamp",
    "is_buyer_maker",
    "is_best_match",
]


def download_day(symbol: str, date: str) -> pd.DataFrame | None:
    """Download aggTrades for one symbol and one day (YYYY-MM-DD)."""
    filename = f"{symbol}-aggTrades-{date}.zip"
    url = f"{BASE_URL}/{symbol}/{filename}"
    print(f"  Downloading {url} ...", end=" ")

    resp = requests.get(url, timeout=60)
    if resp.status_code == 404:
        print("not available (404)")
        return None
    resp.raise_for_status()

    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        csv_name = zf.namelist()[0]
        with zf.open(csv_name) as f:
            df = pd.read_csv(f, header=None, names=COLUMNS)

    print(f"{len(df):,} trades")
    return df


def collect(symbol: str, start_date: datetime, num_days: int) -> pd.DataFrame:
    """Collect num_days of aggTrades starting from start_date."""
    frames = []
    for i in range(num_days):
        date_str = (start_date + timedelta(days=i)).strftime("%Y-%m-%d")
        df = download_day(symbol, date_str)
        if df is not None:
            frames.append(df)

    if not frames:
        raise RuntimeError(f"No data downloaded for {symbol}")

    return pd.concat(frames, ignore_index=True)


def save(df: pd.DataFrame, symbol: str):
    """Save raw data to CSV with standardized column names."""
    # Map to required output format
    out = pd.DataFrame(
        {
            "timestamp": df["timestamp"],
            "price": df["price"].astype(float),
            "quantity": df["quantity"].astype(float),
            "side": df["is_buyer_maker"].map({True: "sell", False: "buy"}),
            "trade_id": df["agg_trade_id"],
        }
    )
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    name = "btc_ticks.csv" if "BTC" in symbol else "eth_ticks.csv"
    path = os.path.join(OUTPUT_DIR, name)
    out.to_csv(path, index=False)
    print(f"Saved {len(out):,} rows to {path}")


def main():
    # Download 7 days of recent data.
    # Binance publishes daily files with a ~1-day lag, so start 8 days ago.
    num_days = 7
    start = datetime.utcnow() - timedelta(days=num_days + 1)

    for symbol in SYMBOLS:
        print(f"\n=== {symbol} ===")
        df = collect(symbol, start, num_days)
        save(df, symbol)

    print("\nDone. Raw files saved to", os.path.abspath(OUTPUT_DIR))


if __name__ == "__main__":
    main()
