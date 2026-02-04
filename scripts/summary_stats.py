"""
Generate summary statistics for the cleaned tick data.
"""

import os
import pandas as pd
from datetime import datetime

PROC_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
FILES = {
    "BTCUSDT": "btc_ticks_clean.parquet",
    "ETHUSDT": "eth_ticks_clean.parquet",
}


def summarize(symbol: str, path: str):
    df = pd.read_parquet(path)
    ts_min = df["timestamp"].min()
    ts_max = df["timestamp"].max()
    # Timestamps are in microseconds (16 digits)
    divisor = 1_000_000
    dt_min = datetime.utcfromtimestamp(ts_min / divisor)
    dt_max = datetime.utcfromtimestamp(ts_max / divisor)
    duration_sec = (ts_max - ts_min) / divisor

    print(f"\n{'='*50}")
    print(f"  {symbol}")
    print(f"{'='*50}")
    print(f"  Data source:         Binance aggTrades (data.binance.vision)")
    print(f"  Date range:          {dt_min:%Y-%m-%d %H:%M:%S} to {dt_max:%Y-%m-%d %H:%M:%S} UTC")
    print(f"  Duration:            {duration_sec/86400:.1f} days")
    print(f"  Total trades:        {len(df):,}")
    print(f"  Avg trades/second:   {len(df)/duration_sec:.1f}")
    print(f"  Price range:         {df['price'].min():.2f} - {df['price'].max():.2f}")
    print(f"  Avg trade size:      {df['quantity'].mean():.6f}")
    print(f"  Timestamp precision: microseconds")

    # Side distribution
    if "side" in df.columns:
        sides = df["side"].value_counts()
        print(f"  Buy trades:          {sides.get('buy', 0):,}")
        print(f"  Sell trades:         {sides.get('sell', 0):,}")

    # Find gaps > 60 seconds
    diffs = df["timestamp"].diff().dropna()
    gaps = diffs[diffs > 60_000_000]  # 60 seconds in microseconds
    if len(gaps) > 0:
        print(f"\n  Gaps > 1 minute:     {len(gaps)}")
        for idx in gaps.index[:10]:  # show first 10
            gap_ts = df.loc[idx, "timestamp"]
            gap_dt = datetime.utcfromtimestamp(gap_ts / divisor)
            gap_sec = gaps[idx] / divisor
            print(f"    {gap_dt:%Y-%m-%d %H:%M:%S} UTC — gap of {gap_sec:.1f}s")
        if len(gaps) > 10:
            print(f"    ... and {len(gaps) - 10} more")
    else:
        print(f"\n  Gaps > 1 minute:     None")

    print()


def main():
    print("\n" + "=" * 50)
    print("  TICK DATA SUMMARY REPORT")
    print("=" * 50)

    for symbol, filename in FILES.items():
        path = os.path.join(PROC_DIR, filename)
        if not os.path.exists(path):
            print(f"\n  {symbol}: {filename} not found, skipping")
            continue
        summarize(symbol, path)


if __name__ == "__main__":
    main()
