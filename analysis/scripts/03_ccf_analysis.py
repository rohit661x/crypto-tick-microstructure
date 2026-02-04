"""
03_ccf_analysis.py
Cross-correlation function (CCF) at multiple sampling frequencies.
Positive lag means BTC leads ETH.
"""

import os
import numpy as np
import pandas as pd

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")

CCF_INTERVALS_MS = [500, 1_000, 2_000, 5_000, 10_000, 30_000]
MAX_LAG = 20
N_BOOTSTRAP = 500
BLOCK_SIZE = 100


def compute_ccf(x: np.ndarray, y: np.ndarray, max_lag: int) -> np.ndarray:
    """Compute normalized cross-correlation at lags -max_lag to +max_lag.
    CCF(τ) = Corr(x_t, y_{t+τ}).  Positive τ: x leads y."""
    x = (x - x.mean()) / x.std()
    y = (y - y.mean()) / y.std()
    n = len(x)
    ccf = np.empty(2 * max_lag + 1)
    for i, lag in enumerate(range(-max_lag, max_lag + 1)):
        if lag >= 0:
            ccf[i] = np.dot(x[:n - lag], y[lag:]) / n
        else:
            ccf[i] = np.dot(x[-lag:], y[:n + lag]) / n
    return ccf


def bootstrap_ccf_peak(x: np.ndarray, y: np.ndarray, max_lag: int,
                        n_boot: int, block_size: int):
    """Bootstrap the peak lag location."""
    n = len(x)
    n_blocks = (n + block_size - 1) // block_size
    peak_lags = np.empty(n_boot)

    for b in range(n_boot):
        starts = np.random.randint(0, n - block_size + 1, size=n_blocks)
        indices = np.concatenate([np.arange(s, s + block_size) for s in starts])[:n]
        ccf_b = compute_ccf(x[indices], y[indices], max_lag)
        lags = np.arange(-max_lag, max_lag + 1)
        peak_lags[b] = lags[np.argmax(np.abs(ccf_b))]

    return peak_lags


def main():
    ccf_rows = []
    peak_rows = []

    for interval_ms in CCF_INTERVALS_MS:
        path = os.path.join(RESULTS_DIR, f"resampled_{interval_ms}ms.parquet")
        if not os.path.exists(path):
            print(f"Skipping {interval_ms}ms")
            continue

        df = pd.read_parquet(path)
        btc_r = df["btc_return"].values
        eth_r = df["eth_return"].values
        mask = np.isfinite(btc_r) & np.isfinite(eth_r)
        btc_r, eth_r = btc_r[mask], eth_r[mask]

        n = len(btc_r)
        ccf = compute_ccf(btc_r, eth_r, MAX_LAG)
        lags = np.arange(-MAX_LAG, MAX_LAG + 1)
        sig_threshold = 1.96 / np.sqrt(n)

        # Store full curve
        for lag, val in zip(lags, ccf):
            ccf_rows.append({
                "sampling_interval_ms": interval_ms,
                "lag": int(lag),
                "lag_ms": int(lag * interval_ms),
                "ccf_value": round(val, 6),
            })

        # Find peak
        peak_idx = np.argmax(np.abs(ccf))
        peak_lag = int(lags[peak_idx])
        peak_ccf = ccf[peak_idx]
        peak_significant = abs(peak_ccf) > sig_threshold

        # Bootstrap peak for error bars
        bs = min(BLOCK_SIZE, n // 10)
        boot_peaks = bootstrap_ccf_peak(btc_r, eth_r, MAX_LAG, N_BOOTSTRAP, max(bs, 2))
        peak_lag_ci_lower = np.percentile(boot_peaks, 2.5)
        peak_lag_ci_upper = np.percentile(boot_peaks, 97.5)

        print(f"Δ={interval_ms:>5}ms  peak_lag={peak_lag:+d} ({peak_lag*interval_ms:+d}ms)  "
              f"CCF={peak_ccf:.4f}  sig={peak_significant}  "
              f"boot_CI=[{peak_lag_ci_lower:.1f}, {peak_lag_ci_upper:.1f}]")

        peak_rows.append({
            "sampling_interval_ms": interval_ms,
            "peak_lag": peak_lag,
            "peak_lag_ms": peak_lag * interval_ms,
            "peak_ccf": round(peak_ccf, 6),
            "peak_significant": peak_significant,
            "significance_threshold": round(sig_threshold, 6),
            "boot_peak_lag_ci_lower": round(peak_lag_ci_lower, 2),
            "boot_peak_lag_ci_upper": round(peak_lag_ci_upper, 2),
            "boot_peak_lag_ms_ci_lower": round(peak_lag_ci_lower * interval_ms, 0),
            "boot_peak_lag_ms_ci_upper": round(peak_lag_ci_upper * interval_ms, 0),
            "n_observations": n,
        })

    # Save
    pd.DataFrame(ccf_rows).to_csv(os.path.join(RESULTS_DIR, "ccf_curves.csv"), index=False)
    pd.DataFrame(peak_rows).to_csv(os.path.join(RESULTS_DIR, "ccf_peaks.csv"), index=False)
    print(f"\nSaved ccf_curves.csv and ccf_peaks.csv")


if __name__ == "__main__":
    main()
