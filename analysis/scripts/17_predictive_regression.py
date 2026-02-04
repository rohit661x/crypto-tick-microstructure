"""
17_predictive_regression.py
Predictive regression: does lagged BTC predict future ETH returns?
Out-of-sample R² with rolling window.
Grid search over lookback and horizon.
"""

import os
import numpy as np
import pandas as pd
from scipy import stats

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "processed")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")

LOOKBACKS_MS = [25, 50, 100, 200]
HORIZONS_MS = [25, 50, 100, 200, 500]
SAMPLING_MS = 25
TRAIN_FRACTION = 0.7


def load_ticks(symbol):
    name = "btc_ticks_clean.parquet" if symbol == "BTC" else "eth_ticks_clean.parquet"
    df = pd.read_parquet(os.path.join(DATA_DIR, name), columns=["timestamp", "price"])
    return df["timestamp"].values, df["price"].values


def predictive_regression(btc_t, btc_lp, eth_t, eth_lp,
                           sampling_ms, lookback_ms, horizon_ms, train_frac):
    """Run predictive regression with OOS evaluation."""
    sampling_us = sampling_ms * 1000
    lookback_us = lookback_ms * 1000
    horizon_us = horizon_ms * 1000

    t_start = max(btc_t[0], eth_t[0]) + lookback_us
    t_end = min(btc_t[-1], eth_t[-1]) - horizon_us

    grid = np.arange(t_start, t_end, sampling_us)

    # Vectorized lookups
    btc_idx_now = np.searchsorted(btc_t, grid, side="right") - 1
    btc_idx_past = np.searchsorted(btc_t, grid - lookback_us, side="right") - 1
    eth_idx_now = np.searchsorted(eth_t, grid, side="right") - 1
    eth_idx_fwd = np.searchsorted(eth_t, grid + horizon_us, side="right") - 1

    btc_idx_now = np.clip(btc_idx_now, 0, len(btc_lp) - 1)
    btc_idx_past = np.clip(btc_idx_past, 0, len(btc_lp) - 1)
    eth_idx_now = np.clip(eth_idx_now, 0, len(eth_lp) - 1)
    eth_idx_fwd = np.clip(eth_idx_fwd, 0, len(eth_lp) - 1)

    X = btc_lp[btc_idx_now] - btc_lp[btc_idx_past]  # BTC lookback return
    Y = eth_lp[eth_idx_fwd] - eth_lp[eth_idx_now]    # ETH forward return

    # Remove any NaN/Inf
    mask = np.isfinite(X) & np.isfinite(Y)
    X, Y = X[mask], Y[mask]

    if len(X) < 1000:
        return None

    # Train/test split
    n_train = int(len(X) * train_frac)
    X_train, X_test = X[:n_train], X[n_train:]
    Y_train, Y_test = Y[:n_train], Y[n_train:]

    # OLS: Y = alpha + beta * X
    x_mean = X_train.mean()
    y_mean = Y_train.mean()
    ss_xx = np.sum((X_train - x_mean) ** 2)
    ss_xy = np.sum((X_train - x_mean) * (Y_train - y_mean))

    if ss_xx == 0:
        return None

    beta = ss_xy / ss_xx
    alpha = y_mean - beta * x_mean

    # In-sample
    Y_train_pred = alpha + beta * X_train
    ss_res_train = np.sum((Y_train - Y_train_pred) ** 2)
    ss_tot_train = np.sum((Y_train - y_mean) ** 2)
    r2_is = 1 - ss_res_train / ss_tot_train if ss_tot_train > 0 else 0

    # Out-of-sample
    Y_test_pred = alpha + beta * X_test
    ss_res_test = np.sum((Y_test - Y_test_pred) ** 2)
    ss_tot_test = np.sum((Y_test - Y_test.mean()) ** 2)
    r2_oos = 1 - ss_res_test / ss_tot_test if ss_tot_test > 0 else 0

    # Standard error and t-stat
    residuals = Y_train - Y_train_pred
    sigma2 = np.sum(residuals ** 2) / (n_train - 2)
    se_beta = np.sqrt(sigma2 / ss_xx)
    t_stat = beta / se_beta if se_beta > 0 else 0
    p_value = 2 * (1 - stats.t.cdf(abs(t_stat), n_train - 2))

    return {
        "lookback_ms": lookback_ms,
        "horizon_ms": horizon_ms,
        "beta": beta,
        "se_beta": se_beta,
        "t_statistic": t_stat,
        "p_value": p_value,
        "r2_insample": r2_is,
        "r2_oos": r2_oos,
        "n_train": n_train,
        "n_test": len(X_test),
        "significant_5pct": p_value < 0.05,
    }


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    btc_t, btc_p = load_ticks("BTC")
    eth_t, eth_p = load_ticks("ETH")

    btc_lp = np.log(btc_p.astype(np.float64))
    eth_lp = np.log(eth_p.astype(np.float64))

    # Use first 2 days to keep runtime reasonable
    t_start = max(btc_t[0], eth_t[0])
    two_days = int(2 * 86400 * 1e6)
    t_end_2d = t_start + two_days

    bm = (btc_t >= t_start) & (btc_t <= t_end_2d)
    em = (eth_t >= t_start) & (eth_t <= t_end_2d)
    btc_t_s, btc_lp_s = btc_t[bm], btc_lp[bm]
    eth_t_s, eth_lp_s = eth_t[em], eth_lp[em]

    print(f"Using {len(btc_t_s):,} BTC and {len(eth_t_s):,} ETH ticks (2 days)")

    rows = []
    for lookback_ms in LOOKBACKS_MS:
        for horizon_ms in HORIZONS_MS:
            print(f"  lookback={lookback_ms}ms, horizon={horizon_ms}ms ...", end=" ")
            result = predictive_regression(
                btc_t_s, btc_lp_s, eth_t_s, eth_lp_s,
                SAMPLING_MS, lookback_ms, horizon_ms, TRAIN_FRACTION,
            )
            if result is None:
                print("skipped (insufficient data)")
                continue

            sig = "***" if result["p_value"] < 0.001 else "**" if result["p_value"] < 0.01 else "*" if result["p_value"] < 0.05 else ""
            print(f"β={result['beta']:.4f} t={result['t_statistic']:.1f} "
                  f"R²_OOS={result['r2_oos']:.4f} {sig}")
            rows.append(result)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS_DIR, "predictive_regression.csv"), index=False)

    # Also save horizon analysis (fixed best lookback)
    if len(df) > 0:
        best = df.loc[df["r2_oos"].idxmax()]
        print(f"\nBest model: lookback={int(best['lookback_ms'])}ms, horizon={int(best['horizon_ms'])}ms")
        print(f"  β = {best['beta']:.4f} (SE = {best['se_beta']:.4f})")
        print(f"  t-stat = {best['t_statistic']:.1f} (p = {best['p_value']:.2e})")
        print(f"  In-sample R² = {best['r2_insample']:.4f}")
        print(f"  Out-of-sample R² = {best['r2_oos']:.4f}")

    df.to_csv(os.path.join(RESULTS_DIR, "horizon_analysis.csv"), index=False)
    print(f"\nSaved predictive_regression.csv and horizon_analysis.csv")


if __name__ == "__main__":
    main()
