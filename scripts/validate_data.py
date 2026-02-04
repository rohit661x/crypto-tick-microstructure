"""
Validate and clean raw tick data, output to processed Parquet files.
"""

import os
import pandas as pd

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
PROC_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")

FILES = {
    "btc_ticks.csv": "btc_ticks_clean.parquet",
    "eth_ticks.csv": "eth_ticks_clean.parquet",
}


def validate_and_clean(path: str) -> pd.DataFrame:
    print(f"\nValidating {path}")
    df = pd.read_csv(path)
    n_raw = len(df)
    print(f"  Raw rows: {n_raw:,}")

    # Drop nulls in critical columns
    before = len(df)
    df = df.dropna(subset=["timestamp", "price", "quantity"])
    dropped_null = before - len(df)
    if dropped_null:
        print(f"  Dropped {dropped_null} rows with null timestamp/price/quantity")

    # Drop duplicate trade_ids
    before = len(df)
    df = df.drop_duplicates(subset=["trade_id"])
    dropped_dup = before - len(df)
    if dropped_dup:
        print(f"  Dropped {dropped_dup} duplicate trade_ids")

    # Sort by timestamp
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Price sanity: remove rows where price <= 0
    before = len(df)
    df = df[df["price"] > 0]
    dropped_price = before - len(df)
    if dropped_price:
        print(f"  Dropped {dropped_price} rows with price <= 0")

    # Quantity sanity
    before = len(df)
    df = df[df["quantity"] > 0]
    dropped_qty = before - len(df)
    if dropped_qty:
        print(f"  Dropped {dropped_qty} rows with quantity <= 0")

    # Verify timestamps are monotonically non-decreasing
    ts = df["timestamp"].values
    non_mono = (ts[1:] < ts[:-1]).sum()
    if non_mono:
        print(f"  WARNING: {non_mono} non-monotonic timestamps found (after sort, should be 0)")
    else:
        print("  Timestamps are monotonically non-decreasing")

    print(f"  Clean rows: {len(df):,} ({len(df)/n_raw*100:.1f}% retained)")
    return df


def main():
    os.makedirs(PROC_DIR, exist_ok=True)

    for raw_name, proc_name in FILES.items():
        raw_path = os.path.join(RAW_DIR, raw_name)
        if not os.path.exists(raw_path):
            print(f"Skipping {raw_name} (not found)")
            continue

        df = validate_and_clean(raw_path)
        out_path = os.path.join(PROC_DIR, proc_name)
        df.to_parquet(out_path, index=False)
        print(f"  Saved to {out_path}")

    print("\nValidation complete.")


if __name__ == "__main__":
    main()
