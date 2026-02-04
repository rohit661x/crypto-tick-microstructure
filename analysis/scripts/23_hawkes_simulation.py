"""
23_hawkes_simulation.py
Simulate from fitted Hawkes model using Ogata's thinning algorithm.
Generate N replications, compute summary statistics (HY cov, CCF, correlations).
"""

import os
import numpy as np
import pandas as pd
from numba import njit

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")

N_REPLICATIONS = 200
T_SIM = 3600.0  # 1 hour per replication


def load_pooled_params():
    df = pd.read_csv(os.path.join(RESULTS_DIR, "hawkes_parameter_estimates.csv"))
    pooled = df[df["day"] == "pooled"].iloc[0]
    params = np.array([
        float(pooled["mu_BTC"]), float(pooled["mu_ETH"]),
        float(pooled["alpha_BTC_from_BTC"]), float(pooled["alpha_BTC_from_ETH"]),
        float(pooled["alpha_ETH_from_BTC"]), float(pooled["alpha_ETH_from_ETH"]),
        float(pooled["beta_BTC_from_BTC"]), float(pooled["beta_BTC_from_ETH"]),
        float(pooled["beta_ETH_from_BTC"]), float(pooled["beta_ETH_from_ETH"]),
    ])
    return params, float(pooled["sigma_BTC"]), float(pooled["sigma_ETH"])


@njit
def simulate_hawkes(params, sigma_BTC, sigma_ETH, T, seed):
    """Ogata's thinning algorithm for bivariate marked Hawkes."""
    np.random.seed(seed)
    mu_B, mu_E = params[0], params[1]
    a_BB, a_BE, a_EB, a_EE = params[2], params[3], params[4], params[5]
    b_BB, b_BE, b_EB, b_EE = params[6], params[7], params[8], params[9]

    max_events = 2_000_000
    btc_times = np.empty(max_events)
    btc_marks = np.empty(max_events)
    eth_times = np.empty(max_events)
    eth_marks = np.empty(max_events)
    n_btc = 0
    n_eth = 0

    R_BB = 0.0
    R_BE = 0.0
    R_EB = 0.0
    R_EE = 0.0
    t = 0.0

    while t < T and (n_btc + n_eth) < max_events - 1:
        lam_B = mu_B + a_BB * R_BB + a_BE * R_BE
        lam_E = mu_E + a_EB * R_EB + a_EE * R_EE
        lam_bar = lam_B + lam_E
        if lam_bar <= 0:
            break

        dt = -np.log(np.random.random()) / lam_bar
        t += dt
        if t >= T:
            break

        R_BB *= np.exp(-b_BB * dt)
        R_BE *= np.exp(-b_BE * dt)
        R_EB *= np.exp(-b_EB * dt)
        R_EE *= np.exp(-b_EE * dt)

        lam_B = mu_B + a_BB * R_BB + a_BE * R_BE
        lam_E = mu_E + a_EB * R_EB + a_EE * R_EE

        u = np.random.random() * lam_bar
        if u <= lam_B:
            btc_times[n_btc] = t
            u2 = np.random.random() - 0.5
            btc_marks[n_btc] = -sigma_BTC * np.sign(u2) * np.log(1 - 2 * np.abs(u2))
            n_btc += 1
            R_BB += b_BB
            R_EB += b_EB
        elif u <= lam_B + lam_E:
            eth_times[n_eth] = t
            u2 = np.random.random() - 0.5
            eth_marks[n_eth] = -sigma_ETH * np.sign(u2) * np.log(1 - 2 * np.abs(u2))
            n_eth += 1
            R_BE += b_BE
            R_EE += b_EE

    return btc_times[:n_btc], btc_marks[:n_btc], eth_times[:n_eth], eth_marks[:n_eth]


@njit
def hy_cov_sim(btc_t, btc_lp, eth_t, eth_lp):
    """HY covariance on simulated log-price paths."""
    btc_ret = np.empty(len(btc_lp) - 1)
    for i in range(len(btc_ret)):
        btc_ret[i] = btc_lp[i + 1] - btc_lp[i]
    eth_ret = np.empty(len(eth_lp) - 1)
    for i in range(len(eth_ret)):
        eth_ret[i] = eth_lp[i + 1] - eth_lp[i]

    cov = 0.0
    var_b = 0.0
    var_e = 0.0
    j_start = 0
    for i in range(len(btc_ret)):
        t_i0, t_i1 = btc_t[i], btc_t[i + 1]
        var_b += btc_ret[i] ** 2
        for j in range(j_start, len(eth_ret)):
            s_j0, s_j1 = eth_t[j], eth_t[j + 1]
            if s_j0 >= t_i1:
                break
            if s_j1 <= t_i0:
                j_start = j + 1
                continue
            cov += btc_ret[i] * eth_ret[j]
    for j in range(len(eth_ret)):
        var_e += eth_ret[j] ** 2
    corr = cov / np.sqrt(var_b * var_e) if var_b > 0 and var_e > 0 else 0.0
    return cov, corr, var_b, var_e


def sync_corr_sim(btc_t, btc_lp, eth_t, eth_lp, interval_s=1.0):
    """Synchronized correlation at given interval."""
    t0 = max(btc_t[0], eth_t[0])
    t1 = min(btc_t[-1], eth_t[-1])
    grid = np.arange(t0, t1, interval_s)
    if len(grid) < 3:
        return 0.0

    bi = np.searchsorted(btc_t, grid, side="right") - 1
    ei = np.searchsorted(eth_t, grid, side="right") - 1
    bi = np.clip(bi, 0, len(btc_lp) - 1)
    ei = np.clip(ei, 0, len(eth_lp) - 1)
    br = np.diff(btc_lp[bi])
    er = np.diff(eth_lp[ei])
    n = min(len(br), len(er))
    br, er = br[:n], er[:n]
    if np.std(br) == 0 or np.std(er) == 0:
        return 0.0
    return np.corrcoef(br, er)[0, 1]


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    params, sigma_BTC, sigma_ETH = load_pooled_params()
    print(f"Simulating {N_REPLICATIONS} paths of {T_SIM:.0f}s each...")

    rows = []
    for s in range(N_REPLICATIONS):
        bt, bm, et, em = simulate_hawkes(params, sigma_BTC, sigma_ETH, T_SIM, seed=1000 + s)

        if len(bt) < 50 or len(et) < 50:
            continue

        # Construct log-price paths
        btc_lp = np.concatenate([[0.0], np.cumsum(bm)])
        eth_lp = np.concatenate([[0.0], np.cumsum(em)])
        btc_t_full = np.concatenate([[0.0], bt])
        eth_t_full = np.concatenate([[0.0], et])

        # HY covariance
        hy_cov, hy_corr, rv_b, rv_e = hy_cov_sim(btc_t_full, btc_lp, eth_t_full, eth_lp)

        # Sync correlation at 1s
        sync_r = sync_corr_sim(btc_t_full, btc_lp, eth_t_full, eth_lp, 1.0)

        rows.append({
            "replication": s,
            "n_btc": len(bt),
            "n_eth": len(et),
            "hy_covariance": hy_cov,
            "hy_correlation": hy_corr,
            "rv_btc": rv_b,
            "rv_eth": rv_e,
            "sync_correlation_1s": sync_r,
        })

        if (s + 1) % 50 == 0:
            print(f"  {s + 1}/{N_REPLICATIONS} done")

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS_DIR, "simulation_summary_stats.csv"), index=False)

    print(f"\nSimulation Summary ({len(df)} successful paths):")
    print(f"  BTC events/path: {df['n_btc'].mean():.0f} ± {df['n_btc'].std():.0f}")
    print(f"  ETH events/path: {df['n_eth'].mean():.0f} ± {df['n_eth'].std():.0f}")
    print(f"  HY correlation:  {df['hy_correlation'].mean():.4f} ± {df['hy_correlation'].std():.4f}")
    print(f"  Sync corr (1s):  {df['sync_correlation_1s'].mean():.4f} ± {df['sync_correlation_1s'].std():.4f}")
    print(f"  RV BTC:          {df['rv_btc'].mean():.6f}")
    print(f"  RV ETH:          {df['rv_eth'].mean():.6f}")

    print(f"\nSaved simulation_summary_stats.csv")


if __name__ == "__main__":
    main()
