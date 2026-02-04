"""
14_lead_lag_contrast.py
Lead-lag estimation via Hoffmann-Rosenbaum-Yoshida contrast function.
Bootstrap inference and hypothesis test for θ = 0.
"""

import os
import numpy as np
import pandas as pd
from numba import njit

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "processed")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")

# Grid: -500ms to +500ms in 5ms steps (positive = BTC leads ETH)
THETA_GRID_MS = np.arange(-500, 501, 5)
BANDWIDTH_MS = 50  # kernel bandwidth
N_BOOTSTRAP = 200
BLOCK_SIZE_MS = 5000  # 5-second blocks for bootstrap


def load_ticks(symbol):
    name = "btc_ticks_clean.parquet" if symbol == "BTC" else "eth_ticks_clean.parquet"
    df = pd.read_parquet(os.path.join(DATA_DIR, name), columns=["timestamp", "price"])
    return df["timestamp"].values, df["price"].values


@njit
def contrast_function(btc_mid, btc_ret, eth_mid, eth_ret, theta_us, h_us):
    """Compute U_n(theta) using Gaussian kernel, with efficient scanning."""
    n_a = len(btc_ret)
    n_b = len(eth_ret)
    U = 0.0
    inv_2h2 = 1.0 / (2.0 * h_us * h_us)
    norm = 1.0 / (np.sqrt(2.0 * np.pi) * h_us)

    # For each BTC return, find nearby ETH returns
    j_lo = 0
    cutoff = 5.0 * h_us  # 5 sigma cutoff

    for i in range(n_a):
        target = btc_mid[i] - theta_us

        # Advance j_lo
        while j_lo < n_b and eth_mid[j_lo] < target - cutoff:
            j_lo += 1

        for j in range(j_lo, n_b):
            diff = target - eth_mid[j]
            if diff < -cutoff:
                break
            w = norm * np.exp(-diff * diff * inv_2h2)
            U += btc_ret[i] * eth_ret[j] * w

    return U


def compute_contrast_curve(btc_t, btc_p, eth_t, eth_p, theta_grid_ms, bandwidth_ms):
    """Compute contrast function over theta grid."""
    btc_log_p = np.log(btc_p.astype(np.float64))
    eth_log_p = np.log(eth_p.astype(np.float64))

    btc_ret = np.diff(btc_log_p)
    eth_ret = np.diff(eth_log_p)
    btc_mid = (btc_t[:-1] + btc_t[1:]) / 2.0
    eth_mid = (eth_t[:-1] + eth_t[1:]) / 2.0

    h_us = bandwidth_ms * 1000.0
    values = np.empty(len(theta_grid_ms))

    for k, theta_ms in enumerate(theta_grid_ms):
        theta_us = theta_ms * 1000.0
        values[k] = contrast_function(btc_mid, btc_ret, eth_mid, eth_ret, theta_us, h_us)

    return values


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    btc_t, btc_p = load_ticks("BTC")
    eth_t, eth_p = load_ticks("ETH")

    t_start = max(btc_t[0], eth_t[0])
    t_end = min(btc_t[-1], eth_t[-1])

    # Process one day at a time for the contrast function (full dataset too large for O(N²))
    day_us = int(86400 * 1e6)
    day_starts = np.arange(t_start, t_end, day_us)

    all_contrast = np.zeros(len(THETA_GRID_MS))
    day_thetas = []

    for d_idx, d_start in enumerate(day_starts):
        d_end = d_start + day_us
        bm = (btc_t >= d_start) & (btc_t < d_end)
        em = (eth_t >= d_start) & (eth_t < d_end)
        bt, bp = btc_t[bm], btc_p[bm]
        et, ep = eth_t[em], eth_p[em]

        if len(bt) < 1000 or len(et) < 1000:
            continue

        # Subsample to keep O(N²) manageable: take every 10th tick
        step = max(1, len(bt) // 100_000)
        bt_s, bp_s = bt[::step], bp[::step]
        step_e = max(1, len(et) // 100_000)
        et_s, ep_s = et[::step_e], ep[::step_e]

        print(f"  Day {d_idx + 1}: BTC={len(bt_s):,} ETH={len(et_s):,} ticks (subsampled)")
        cv = compute_contrast_curve(bt_s, bp_s, et_s, ep_s, THETA_GRID_MS, BANDWIDTH_MS)
        all_contrast += cv

        # Day-level peak
        peak_idx = np.argmax(cv)
        day_thetas.append(THETA_GRID_MS[peak_idx])

    # Overall peak
    peak_idx = np.argmax(all_contrast)
    theta_hat = THETA_GRID_MS[peak_idx]
    print(f"\nContrast function peak: θ = {theta_hat}ms")

    # Normalize contrast for output
    contrast_max = np.max(all_contrast)
    contrast_norm = all_contrast / contrast_max if contrast_max != 0 else all_contrast

    # Save contrast curve
    contrast_df = pd.DataFrame({
        "theta_ms": THETA_GRID_MS,
        "contrast": all_contrast,
        "contrast_normalized": contrast_norm,
    })
    contrast_df.to_csv(os.path.join(RESULTS_DIR, "contrast_function.csv"), index=False)

    # Bootstrap: resample days
    print(f"\nBootstrapping ({N_BOOTSTRAP} iterations, day-level block bootstrap)...")
    day_thetas = np.array(day_thetas)
    n_days = len(day_thetas)
    boot_thetas = np.empty(N_BOOTSTRAP)

    for b in range(N_BOOTSTRAP):
        sampled = day_thetas[np.random.randint(0, n_days, size=n_days)]
        boot_thetas[b] = np.mean(sampled)

    se_theta = np.std(boot_thetas)
    ci_lo = np.percentile(boot_thetas, 2.5)
    ci_hi = np.percentile(boot_thetas, 97.5)

    # Hypothesis test: H0: θ = 0
    if se_theta > 0:
        t_stat = theta_hat / se_theta
        from scipy import stats
        p_value = 2 * (1 - stats.norm.cdf(abs(t_stat)))
    else:
        t_stat = np.inf
        p_value = 0.0

    print(f"θ_hat = {theta_hat}ms, SE = {se_theta:.1f}ms, 95% CI = [{ci_lo:.1f}, {ci_hi:.1f}]ms")
    print(f"H0: θ=0  t={t_stat:.2f}  p={p_value:.4f}")

    # Save estimates
    est_df = pd.DataFrame([{
        "theta_hat_ms": theta_hat,
        "se_ms": se_theta,
        "ci_lower_ms": ci_lo,
        "ci_upper_ms": ci_hi,
        "t_statistic": t_stat,
        "p_value": p_value,
        "significant_5pct": p_value < 0.05,
        "bandwidth_ms": BANDWIDTH_MS,
        "n_days": n_days,
        "n_bootstrap": N_BOOTSTRAP,
    }])
    est_df.to_csv(os.path.join(RESULTS_DIR, "lead_lag_estimates.csv"), index=False)

    # Save day-level bootstrap distribution
    boot_df = pd.DataFrame({
        "bootstrap_iteration": np.arange(N_BOOTSTRAP),
        "theta_ms": boot_thetas,
    })
    boot_df.to_csv(os.path.join(RESULTS_DIR, "lead_lag_bootstrap.csv"), index=False)

    print("\nSaved contrast_function.csv, lead_lag_estimates.csv, lead_lag_bootstrap.csv")


if __name__ == "__main__":
    main()
