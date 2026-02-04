"""
02_correlation_analysis.py
Compute Pearson correlation between BTC and ETH returns at each sampling frequency.
Block bootstrap for 95% confidence intervals.
"""

import os
import numpy as np
import pandas as pd

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")

INTERVALS_MS = [100, 250, 500, 1_000, 2_000, 5_000, 10_000, 30_000, 60_000, 300_000]
N_BOOTSTRAP = 1000
BLOCK_SIZE = 100


def block_bootstrap_correlation(x: np.ndarray, y: np.ndarray,
                                 n_boot: int, block_size: int) -> np.ndarray:
    """Block bootstrap for correlation, preserving serial dependence."""
    n = len(x)
    correlations = np.empty(n_boot)
    n_blocks = (n + block_size - 1) // block_size

    for b in range(n_boot):
        # Sample block start indices with replacement
        starts = np.random.randint(0, n - block_size + 1, size=n_blocks)
        indices = np.concatenate([np.arange(s, s + block_size) for s in starts])[:n]
        correlations[b] = np.corrcoef(x[indices], y[indices])[0, 1]

    return correlations


def main():
    rows = []

    for interval_ms in INTERVALS_MS:
        path = os.path.join(RESULTS_DIR, f"resampled_{interval_ms}ms.parquet")
        if not os.path.exists(path):
            print(f"Skipping {interval_ms}ms (file not found)")
            continue

        df = pd.read_parquet(path)
        btc_r = df["btc_return"].values
        eth_r = df["eth_return"].values

        # Drop any NaN
        mask = np.isfinite(btc_r) & np.isfinite(eth_r)
        btc_r, eth_r = btc_r[mask], eth_r[mask]

        corr = np.corrcoef(btc_r, eth_r)[0, 1]

        # Block bootstrap
        bs = min(BLOCK_SIZE, len(btc_r) // 10)
        boot_corrs = block_bootstrap_correlation(btc_r, eth_r, N_BOOTSTRAP, max(bs, 2))
        ci_lower = np.percentile(boot_corrs, 2.5)
        ci_upper = np.percentile(boot_corrs, 97.5)

        print(f"Δ={interval_ms:>6}ms  ρ={corr:.4f}  95%CI=[{ci_lower:.4f}, {ci_upper:.4f}]  n={len(btc_r):,}")

        rows.append({
            "sampling_interval_ms": interval_ms,
            "correlation": round(corr, 6),
            "ci_lower": round(ci_lower, 6),
            "ci_upper": round(ci_upper, 6),
            "n_observations": len(btc_r),
        })

    out = pd.DataFrame(rows)
    outpath = os.path.join(RESULTS_DIR, "correlation_vs_frequency.csv")
    out.to_csv(outpath, index=False)
    print(f"\nSaved to {outpath}")


if __name__ == "__main__":
    main()
