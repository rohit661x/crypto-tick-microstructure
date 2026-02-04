"""
13_hayashi_yoshida.py
Non-synchronous covariance estimation:
  - Refresh-time synchronization
  - Hayashi-Yoshida estimator
  - Pre-averaged HY estimator
With asymptotic confidence intervals.
"""

import os
import numpy as np
import pandas as pd
from numba import njit

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "processed")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")


def load_ticks(symbol):
    name = "btc_ticks_clean.parquet" if symbol == "BTC" else "eth_ticks_clean.parquet"
    df = pd.read_parquet(os.path.join(DATA_DIR, name), columns=["timestamp", "price"])
    return df["timestamp"].values, df["price"].values


@njit
def hayashi_yoshida_core(btc_t, btc_log_p, eth_t, eth_log_p):
    """Efficient O(N_A + N_B) HY estimator using sorted scan."""
    n_a = len(btc_t) - 1
    n_b = len(eth_t) - 1

    btc_ret = np.empty(n_a)
    for i in range(n_a):
        btc_ret[i] = btc_log_p[i + 1] - btc_log_p[i]

    eth_ret = np.empty(n_b)
    for j in range(n_b):
        eth_ret[j] = eth_log_p[j + 1] - eth_log_p[j]

    cov_hy = 0.0
    var_hy = 0.0  # for asymptotic variance
    n_overlaps = 0

    j_start = 0
    for i in range(n_a):
        t_i0 = btc_t[i]
        t_i1 = btc_t[i + 1]

        for j in range(j_start, n_b):
            s_j0 = eth_t[j]
            s_j1 = eth_t[j + 1]

            if s_j0 >= t_i1:
                break
            if s_j1 <= t_i0:
                j_start = j + 1
                continue

            # Overlap exists
            product = btc_ret[i] * eth_ret[j]
            cov_hy += product
            var_hy += (btc_ret[i] ** 2) * (eth_ret[j] ** 2)
            n_overlaps += 1

    var_btc = 0.0
    for i in range(n_a):
        var_btc += btc_ret[i] ** 2

    var_eth = 0.0
    for j in range(n_b):
        var_eth += eth_ret[j] ** 2

    return cov_hy, var_btc, var_eth, var_hy, n_overlaps


def preaverage_series(times, log_prices, K):
    """Pre-average a tick series with weight g(x) = min(x, 1-x)."""
    returns = np.diff(log_prices)
    n = len(returns)
    if n <= K:
        return times[:1], log_prices[:1]

    weights = np.array([min(k / K, 1 - k / K) for k in range(1, K + 1)])
    psi2 = np.sum(weights ** 2) / K ** 2

    n_out = n - K
    pa_returns = np.empty(n_out)
    for i in range(n_out):
        pa_returns[i] = np.dot(weights, returns[i:i + K])

    # Midpoint times
    pa_times = (times[:-1][: n_out] + times[1:][: n_out]) / 2
    # Reconstruct cumulative log prices for HY
    pa_log_prices = np.concatenate([[0.0], np.cumsum(pa_returns)])

    return pa_times, pa_log_prices, psi2, K


def refresh_time_sync(btc_t, btc_p, eth_t, eth_p, interval_us=100_000):
    """Synchronize via last-tick at regular grid (refresh-time approximation)."""
    t_start = max(btc_t[0], eth_t[0])
    t_end = min(btc_t[-1], eth_t[-1])
    grid = np.arange(t_start, t_end, interval_us)

    btc_idx = np.searchsorted(btc_t, grid, side="right") - 1
    eth_idx = np.searchsorted(eth_t, grid, side="right") - 1

    btc_idx = np.clip(btc_idx, 0, len(btc_p) - 1)
    eth_idx = np.clip(eth_idx, 0, len(eth_p) - 1)

    btc_sync = np.log(btc_p[btc_idx])
    eth_sync = np.log(eth_p[eth_idx])

    btc_r = np.diff(btc_sync)
    eth_r = np.diff(eth_sync)

    cov = np.sum(btc_r * eth_r)
    var_b = np.sum(btc_r ** 2)
    var_e = np.sum(eth_r ** 2)
    n = len(btc_r)

    return cov, var_b, var_e, n


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    btc_t, btc_p = load_ticks("BTC")
    eth_t, eth_p = load_ticks("ETH")

    # Common time range
    t_start = max(btc_t[0], eth_t[0])
    t_end = min(btc_t[-1], eth_t[-1])
    btc_mask = (btc_t >= t_start) & (btc_t <= t_end)
    eth_mask = (eth_t >= t_start) & (eth_t <= t_end)
    btc_t, btc_p = btc_t[btc_mask], btc_p[btc_mask]
    eth_t, eth_p = eth_t[eth_mask], eth_p[eth_mask]

    duration_days = (t_end - t_start) / 1e6 / 86400
    print(f"BTC ticks: {len(btc_t):,}, ETH ticks: {len(eth_t):,}")
    print(f"Duration: {duration_days:.1f} days")

    # Process day by day and aggregate
    day_us = int(86400 * 1e6)
    day_starts = np.arange(t_start, t_end, day_us)

    results_rows = []

    for method in ["refresh_100ms", "hayashi_yoshida", "preavg_hy"]:
        total_cov = 0.0
        total_var_btc = 0.0
        total_var_eth = 0.0
        total_avar = 0.0
        total_n = 0
        day_covs = []

        for d_start in day_starts:
            d_end = d_start + day_us
            bm = (btc_t >= d_start) & (btc_t < d_end)
            em = (eth_t >= d_start) & (eth_t < d_end)
            bt, bp = btc_t[bm], btc_p[bm]
            et, ep = eth_t[em], eth_p[em]

            if len(bt) < 100 or len(et) < 100:
                continue

            if method == "refresh_100ms":
                cov, vb, ve, n = refresh_time_sync(bt, bp, et, ep, 100_000)
                avar = 0.0
            elif method == "hayashi_yoshida":
                cov, vb, ve, avar, n = hayashi_yoshida_core(
                    bt, np.log(bp.astype(np.float64)),
                    et, np.log(ep.astype(np.float64)),
                )
            elif method == "preavg_hy":
                K = 50
                bt2, blp, psi2_b, _ = preaverage_series(bt, np.log(bp.astype(np.float64)), K)
                et2, elp, psi2_e, _ = preaverage_series(et, np.log(ep.astype(np.float64)), K)
                if len(bt2) < 10 or len(et2) < 10:
                    continue
                cov, vb, ve, avar, n = hayashi_yoshida_core(bt2, blp, et2, elp)
                # Bias correction
                cov = cov / (psi2_b * K)
                vb = vb / (psi2_b * K)
                ve = ve / (psi2_e * K)

            total_cov += cov
            total_var_btc += vb
            total_var_eth += ve
            total_avar += avar
            total_n += n
            day_covs.append(cov)

        n_days = len(day_covs)
        cov_daily = total_cov / n_days if n_days > 0 else 0
        var_btc_daily = total_var_btc / n_days if n_days > 0 else 0
        var_eth_daily = total_var_eth / n_days if n_days > 0 else 0

        corr = total_cov / np.sqrt(total_var_btc * total_var_eth) if total_var_btc > 0 and total_var_eth > 0 else 0

        # Std error from day-to-day variation
        if n_days > 1:
            se_cov = np.std(day_covs, ddof=1) / np.sqrt(n_days)
        else:
            se_cov = np.sqrt(2 * total_avar / total_n) if total_n > 0 else 0

        ci_lo = total_cov - 1.96 * se_cov * n_days
        ci_hi = total_cov + 1.96 * se_cov * n_days

        # Correlation CI via Fisher z
        if abs(corr) < 1:
            z = np.arctanh(corr)
            se_z = 1.0 / np.sqrt(total_n - 3) if total_n > 3 else 0
            corr_ci_lo = np.tanh(z - 1.96 * se_z)
            corr_ci_hi = np.tanh(z + 1.96 * se_z)
        else:
            corr_ci_lo = corr_ci_hi = corr

        print(f"\n{method}:")
        print(f"  Covariance (total): {total_cov:.8f}")
        print(f"  Covariance (daily): {cov_daily:.8f}")
        print(f"  Correlation:        {corr:.4f} [{corr_ci_lo:.4f}, {corr_ci_hi:.4f}]")
        print(f"  RV BTC (daily):     {var_btc_daily:.8f}")
        print(f"  RV ETH (daily):     {var_eth_daily:.8f}")
        print(f"  Overlaps/obs:       {total_n:,}")

        results_rows.append({
            "method": method,
            "covariance_total": total_cov,
            "covariance_daily": cov_daily,
            "correlation": corr,
            "corr_ci_lower": corr_ci_lo,
            "corr_ci_upper": corr_ci_hi,
            "rv_btc_daily": var_btc_daily,
            "rv_eth_daily": var_eth_daily,
            "se_cov_daily": se_cov,
            "n_observations": total_n,
            "n_days": n_days,
        })

    df = pd.DataFrame(results_rows)
    df.to_csv(os.path.join(RESULTS_DIR, "covariance_estimates.csv"), index=False)

    # Attenuation factor
    sync_corr = df.loc[df["method"] == "refresh_100ms", "correlation"].values[0]
    hy_corr = df.loc[df["method"] == "hayashi_yoshida", "correlation"].values[0]
    print(f"\nAttenuation factor (sync/HY): {sync_corr / hy_corr:.4f}")

    print("\nSaved covariance_estimates.csv")


if __name__ == "__main__":
    main()
