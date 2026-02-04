"""
15_granger_causality.py
Granger causality tests at tick-level using binned returns at multiple windows.
"""

import os
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import grangercausalitytests

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "processed")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")

WINDOW_MS_LIST = [100, 200, 500, 1000]
MAX_LAGS_MAP = {100: 10, 200: 10, 500: 10, 1000: 10}


def load_ticks(symbol):
    name = "btc_ticks_clean.parquet" if symbol == "BTC" else "eth_ticks_clean.parquet"
    df = pd.read_parquet(os.path.join(DATA_DIR, name), columns=["timestamp", "price"])
    return df["timestamp"].values, df["price"].values


def bin_returns(times, prices, t_start, t_end, window_us):
    """Compute returns in fixed-width time bins."""
    log_p = np.log(prices.astype(np.float64))
    bins = np.arange(t_start, t_end, window_us)
    idx = np.searchsorted(times, bins, side="right") - 1
    idx = np.clip(idx, 0, len(log_p) - 1)
    binned_lp = log_p[idx]
    returns = np.diff(binned_lp)
    return returns


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    btc_t, btc_p = load_ticks("BTC")
    eth_t, eth_p = load_ticks("ETH")

    t_start = max(btc_t[0], eth_t[0])
    # Use 2 days to keep Granger test tractable
    two_days = int(2 * 86400 * 1e6)
    t_end = min(min(btc_t[-1], eth_t[-1]), t_start + two_days)

    rows = []

    for window_ms in WINDOW_MS_LIST:
        window_us = window_ms * 1000
        max_lags = MAX_LAGS_MAP[window_ms]

        btc_r = bin_returns(btc_t, btc_p, t_start, t_end, window_us)
        eth_r = bin_returns(eth_t, eth_p, t_start, t_end, window_us)

        n = min(len(btc_r), len(eth_r))
        btc_r = btc_r[:n]
        eth_r = eth_r[:n]

        print(f"\nWindow = {window_ms}ms, n = {n:,}")

        # BTC -> ETH: does BTC Granger-cause ETH?
        df_be = pd.DataFrame({"eth": eth_r, "btc": btc_r})
        try:
            gc_be = grangercausalitytests(df_be, maxlag=max_lags, verbose=False)
        except Exception as e:
            print(f"  BTC->ETH failed: {e}")
            gc_be = None

        # ETH -> BTC
        df_eb = pd.DataFrame({"btc": btc_r, "eth": eth_r})
        try:
            gc_eb = grangercausalitytests(df_eb, maxlag=max_lags, verbose=False)
        except Exception as e:
            print(f"  ETH->BTC failed: {e}")
            gc_eb = None

        for lag in range(1, max_lags + 1):
            lag_ms = lag * window_ms

            if gc_be is not None:
                f_stat, p_val, _, _ = gc_be[lag][0]["ssr_ftest"]
                sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
                print(f"  BTC->ETH lag={lag} ({lag_ms}ms): F={f_stat:.2f} p={p_val:.4f} {sig}")
                rows.append({
                    "window_ms": window_ms,
                    "direction": "BTC->ETH",
                    "lag": lag,
                    "lag_ms": lag_ms,
                    "f_statistic": round(f_stat, 4),
                    "p_value": p_val,
                    "significant_5pct": p_val < 0.05,
                    "n_observations": n,
                })

            if gc_eb is not None:
                f_stat, p_val, _, _ = gc_eb[lag][0]["ssr_ftest"]
                sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
                print(f"  ETH->BTC lag={lag} ({lag_ms}ms): F={f_stat:.2f} p={p_val:.4f} {sig}")
                rows.append({
                    "window_ms": window_ms,
                    "direction": "ETH->BTC",
                    "lag": lag,
                    "lag_ms": lag_ms,
                    "f_statistic": round(f_stat, 4),
                    "p_value": p_val,
                    "significant_5pct": p_val < 0.05,
                    "n_observations": n,
                })

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS_DIR, "granger_results.csv"), index=False)
    print(f"\nSaved granger_results.csv ({len(df)} rows)")


if __name__ == "__main__":
    main()
