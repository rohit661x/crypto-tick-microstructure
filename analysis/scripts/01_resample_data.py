"""
01_resample_data.py
Resample tick data to regular grids at various intervals using last-tick interpolation.
Compute log returns for each resampled series.
"""

import os
import numpy as np
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "processed")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")

INTERVALS_MS = [100, 250, 500, 1_000, 2_000, 5_000, 10_000, 30_000, 60_000, 300_000]
INTERVAL_LABELS = {
    100: "100ms", 250: "250ms", 500: "500ms", 1_000: "1s",
    2_000: "2s", 5_000: "5s", 10_000: "10s", 30_000: "30s",
    60_000: "1min", 300_000: "5min",
}


def load_ticks(symbol: str) -> pd.DataFrame:
    name = "btc_ticks_clean.parquet" if symbol == "BTC" else "eth_ticks_clean.parquet"
    df = pd.read_parquet(os.path.join(DATA_DIR, name))
    # Convert microsecond timestamps to datetime
    df["dt"] = pd.to_datetime(df["timestamp"], unit="us", utc=True)
    df = df.sort_values("dt").reset_index(drop=True)
    return df


def resample_to_grid(df: pd.DataFrame, interval_ms: int) -> pd.Series:
    """Resample prices to regular grid using last-tick (forward-fill) interpolation."""
    freq_str = f"{interval_ms}ms"
    ts = df.set_index("dt")["price"]
    # Use resample().last() to get the last tick in each bucket, then ffill gaps
    resampled = ts.resample(freq_str).last().ffill()
    resampled = resampled.dropna()
    return resampled


def compute_log_returns(prices: pd.Series) -> pd.Series:
    return np.log(prices).diff().dropna()


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    btc = load_ticks("BTC")
    eth = load_ticks("ETH")
    print(f"BTC ticks: {len(btc):,}, ETH ticks: {len(eth):,}")

    # Find common time range
    t_start = max(btc["dt"].min(), eth["dt"].min())
    t_end = min(btc["dt"].max(), eth["dt"].max())
    print(f"Common range: {t_start} to {t_end}")

    btc = btc[(btc["dt"] >= t_start) & (btc["dt"] <= t_end)]
    eth = eth[(eth["dt"] >= t_start) & (eth["dt"] <= t_end)]

    for interval_ms in INTERVALS_MS:
        label = INTERVAL_LABELS[interval_ms]
        print(f"\nResampling at Δ = {label}...")

        btc_prices = resample_to_grid(btc, interval_ms)
        eth_prices = resample_to_grid(eth, interval_ms)

        # Align indices
        common_idx = btc_prices.index.intersection(eth_prices.index)
        btc_prices = btc_prices.loc[common_idx]
        eth_prices = eth_prices.loc[common_idx]

        btc_ret = compute_log_returns(btc_prices)
        eth_ret = compute_log_returns(eth_prices)

        # Align returns
        common_ret_idx = btc_ret.index.intersection(eth_ret.index)
        btc_ret = btc_ret.loc[common_ret_idx]
        eth_ret = eth_ret.loc[common_ret_idx]

        # Save
        out = pd.DataFrame({
            "timestamp": common_ret_idx,
            "btc_price": btc_prices.loc[common_ret_idx].values,
            "eth_price": eth_prices.loc[common_ret_idx].values,
            "btc_return": btc_ret.values,
            "eth_return": eth_ret.values,
        })
        outpath = os.path.join(RESULTS_DIR, f"resampled_{interval_ms}ms.parquet")
        out.to_parquet(outpath, index=False)
        print(f"  {len(out):,} observations saved to {outpath}")

    print("\nResampling complete.")


if __name__ == "__main__":
    main()
