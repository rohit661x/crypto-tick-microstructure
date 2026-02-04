"""
21_hawkes_diagnostics.py
Diagnostics for fitted Hawkes model:
  - Time-rescaling theorem: QQ plot and KS test on rescaled inter-arrivals
  - Simulation-based HY goodness-of-fit
  - Simulation-based CCF comparison
"""

import os
import numpy as np
import pandas as pd
from numba import njit
from scipy import stats

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "processed")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
FIGURES_DIR = os.path.join(os.path.dirname(__file__), "..", "figures")

N_SIM_PATHS = 100


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
    sigma_BTC = float(pooled["sigma_BTC"])
    sigma_ETH = float(pooled["sigma_ETH"])
    return params, sigma_BTC, sigma_ETH


def load_ticks(symbol):
    name = "btc_ticks_clean.parquet" if symbol == "BTC" else "eth_ticks_clean.parquet"
    df = pd.read_parquet(os.path.join(DATA_DIR, name), columns=["timestamp", "price"])
    return df["timestamp"].values, df["price"].values


@njit
def compute_intensities(params, times, assets, T):
    """Compute intensity at each event time (for time-rescaling)."""
    mu_B, mu_E = params[0], params[1]
    a_BB, a_BE, a_EB, a_EE = params[2], params[3], params[4], params[5]
    b_BB, b_BE, b_EB, b_EE = params[6], params[7], params[8], params[9]

    n = len(times)
    intensities = np.empty(n)
    R_BB = 0.0
    R_BE = 0.0
    R_EB = 0.0
    R_EE = 0.0
    prev_t = 0.0

    for k in range(n):
        t = times[k]
        a = assets[k]
        dt = t - prev_t
        if dt > 0:
            R_BB *= np.exp(-b_BB * dt)
            R_BE *= np.exp(-b_BE * dt)
            R_EB *= np.exp(-b_EB * dt)
            R_EE *= np.exp(-b_EE * dt)

        if a == 0:
            intensities[k] = mu_B + a_BB * R_BB + a_BE * R_BE
        else:
            intensities[k] = mu_E + a_EB * R_EB + a_EE * R_EE

        if a == 0:
            R_BB += b_BB
            R_EB += b_EB
        else:
            R_BE += b_BE
            R_EE += b_EE

        prev_t = t

    return intensities


@njit
def compute_compensator_increments(params, times, assets):
    """Compute Λ(t_{k-1}, t_k) for time-rescaling, per-asset."""
    mu_B, mu_E = params[0], params[1]
    a_BB, a_BE, a_EB, a_EE = params[2], params[3], params[4], params[5]
    b_BB, b_BE, b_EB, b_EE = params[6], params[7], params[8], params[9]

    n = len(times)
    # Track running R values at each event time
    R_BB = 0.0
    R_BE = 0.0
    R_EB = 0.0
    R_EE = 0.0
    prev_t = 0.0

    # For BTC events: compute compensator increments between consecutive BTC events
    btc_times_list = np.empty(n)
    btc_R_BB_list = np.empty(n)
    btc_R_BE_list = np.empty(n)
    n_btc = 0

    eth_times_list = np.empty(n)
    eth_R_EB_list = np.empty(n)
    eth_R_EE_list = np.empty(n)
    n_eth = 0

    for k in range(n):
        t = times[k]
        a = assets[k]
        dt = t - prev_t
        if dt > 0:
            R_BB *= np.exp(-b_BB * dt)
            R_BE *= np.exp(-b_BE * dt)
            R_EB *= np.exp(-b_EB * dt)
            R_EE *= np.exp(-b_EE * dt)

        if a == 0:
            btc_times_list[n_btc] = t
            btc_R_BB_list[n_btc] = R_BB
            btc_R_BE_list[n_btc] = R_BE
            n_btc += 1
            R_BB += b_BB
            R_EB += b_EB
        else:
            eth_times_list[n_eth] = t
            eth_R_EB_list[n_eth] = R_EB
            eth_R_EE_list[n_eth] = R_EE
            n_eth += 1
            R_BE += b_BE
            R_EE += b_EE

        prev_t = t

    # Compensator increments for BTC between consecutive BTC events
    # Λ_BTC(t_{k-1}, t_k) = μ_B·Δt + Σ (excitation integrals)
    # Simplified: use trapezoidal with the intensity at event times
    # More accurate: closed-form with exponential kernels is complex for interleaved events
    # Approximate: intensity at event time × inter-event time
    btc_Lambda = np.empty(n_btc - 1)
    for i in range(1, n_btc):
        dt = btc_times_list[i] - btc_times_list[i - 1]
        lam_start = mu_B + a_BB * btc_R_BB_list[i - 1] + a_BE * btc_R_BE_list[i - 1]
        lam_end = mu_B + a_BB * btc_R_BB_list[i] + a_BE * btc_R_BE_list[i]
        btc_Lambda[i - 1] = 0.5 * (lam_start + lam_end) * dt

    eth_Lambda = np.empty(n_eth - 1)
    for i in range(1, n_eth):
        dt = eth_times_list[i] - eth_times_list[i - 1]
        lam_start = mu_E + a_EB * eth_R_EB_list[i - 1] + a_EE * eth_R_EE_list[i - 1]
        lam_end = mu_E + a_EB * eth_R_EB_list[i] + a_EE * eth_R_EE_list[i]
        eth_Lambda[i - 1] = 0.5 * (lam_start + lam_end) * dt

    return btc_Lambda, eth_Lambda


@njit
def simulate_hawkes(params, sigma_BTC, sigma_ETH, T, seed):
    """Ogata's thinning algorithm for bivariate marked Hawkes."""
    np.random.seed(seed)
    mu_B, mu_E = params[0], params[1]
    a_BB, a_BE, a_EB, a_EE = params[2], params[3], params[4], params[5]
    b_BB, b_BE, b_EB, b_EE = params[6], params[7], params[8], params[9]

    max_events = 500000
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

        # Next candidate event time
        dt = -np.log(np.random.random()) / lam_bar
        t += dt

        if t >= T:
            break

        # Decay
        R_BB *= np.exp(-b_BB * dt)
        R_BE *= np.exp(-b_BE * dt)
        R_EB *= np.exp(-b_EB * dt)
        R_EE *= np.exp(-b_EE * dt)

        # Recompute intensities after decay
        lam_B = mu_B + a_BB * R_BB + a_BE * R_BE
        lam_E = mu_E + a_EB * R_EB + a_EE * R_EE

        # Accept/reject
        u = np.random.random() * lam_bar
        if u <= lam_B:
            # BTC event
            btc_times[n_btc] = t
            # Laplace mark
            u2 = np.random.random() - 0.5
            btc_marks[n_btc] = -sigma_BTC * np.sign(u2) * np.log(1 - 2 * np.abs(u2))
            n_btc += 1
            R_BB += b_BB
            R_EB += b_EB
        elif u <= lam_B + lam_E:
            # ETH event
            eth_times[n_eth] = t
            u2 = np.random.random() - 0.5
            eth_marks[n_eth] = -sigma_ETH * np.sign(u2) * np.log(1 - 2 * np.abs(u2))
            n_eth += 1
            R_BE += b_BE
            R_EE += b_EE
        # else: thinned (reject)

    return btc_times[:n_btc], btc_marks[:n_btc], eth_times[:n_eth], eth_marks[:n_eth]


def hayashi_yoshida_sim(btc_t, btc_lp, eth_t, eth_lp):
    """Simple HY covariance for simulated data."""
    btc_ret = np.diff(btc_lp)
    eth_ret = np.diff(eth_lp)
    n_a = len(btc_ret)
    n_b = len(eth_ret)
    cov = 0.0
    j_start = 0
    for i in range(n_a):
        t_i0, t_i1 = btc_t[i], btc_t[i + 1]
        for j in range(j_start, n_b):
            s_j0, s_j1 = eth_t[j], eth_t[j + 1]
            if s_j0 >= t_i1:
                break
            if s_j1 <= t_i0:
                j_start = j + 1
                continue
            cov += btc_ret[i] * eth_ret[j]
    var_b = np.sum(btc_ret ** 2)
    var_e = np.sum(eth_ret ** 2)
    corr = cov / np.sqrt(var_b * var_e) if var_b > 0 and var_e > 0 else 0
    return cov, corr


def ccf_from_sim(btc_t, btc_marks, eth_t, eth_marks, interval_ms=500, max_lag=20):
    """Compute CCF on simulated path after resampling."""
    if len(btc_t) < 10 or len(eth_t) < 10:
        return np.zeros(2 * max_lag + 1)

    t0 = max(btc_t[0], eth_t[0])
    t1 = min(btc_t[-1], eth_t[-1])
    interval_s = interval_ms / 1000.0
    grid = np.arange(t0, t1, interval_s)
    if len(grid) < 2 * max_lag + 2:
        return np.zeros(2 * max_lag + 1)

    btc_cum = np.cumsum(btc_marks)
    eth_cum = np.cumsum(eth_marks)
    btc_cum = np.concatenate([[0], btc_cum])
    eth_cum = np.concatenate([[0], eth_cum])

    btc_idx = np.searchsorted(btc_t, grid, side="right") - 1
    eth_idx = np.searchsorted(eth_t, grid, side="right") - 1
    btc_idx = np.clip(btc_idx, 0, len(btc_cum) - 1)
    eth_idx = np.clip(eth_idx, 0, len(eth_cum) - 1)

    btc_r = np.diff(btc_cum[btc_idx])
    eth_r = np.diff(eth_cum[eth_idx])

    n = min(len(btc_r), len(eth_r))
    btc_r = btc_r[:n]
    eth_r = eth_r[:n]

    if np.std(btc_r) == 0 or np.std(eth_r) == 0:
        return np.zeros(2 * max_lag + 1)

    x = (btc_r - btc_r.mean()) / btc_r.std()
    y = (eth_r - eth_r.mean()) / eth_r.std()

    ccf = np.empty(2 * max_lag + 1)
    for idx, lag in enumerate(range(-max_lag, max_lag + 1)):
        if lag >= 0:
            ccf[idx] = np.dot(x[:n - lag], y[lag:]) / n
        else:
            ccf[idx] = np.dot(x[-lag:], y[:n + lag]) / n
    return ccf


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    params, sigma_BTC, sigma_ETH = load_pooled_params()
    print("Pooled parameters loaded")
    print(f"  μ_BTC={params[0]:.2f}  μ_ETH={params[1]:.2f}")
    print(f"  α_BB={params[2]:.4f}  α_BE={params[3]:.4f}  α_EB={params[4]:.4f}  α_EE={params[5]:.4f}")

    # Load one day of real data for diagnostics
    btc_t_all, btc_p_all = load_ticks("BTC")
    eth_t_all, eth_p_all = load_ticks("ETH")
    t_start = max(btc_t_all[0], eth_t_all[0])
    day_us = int(86400 * 1e6)
    d_end = t_start + day_us

    bm = (btc_t_all >= t_start) & (btc_t_all < d_end)
    em = (eth_t_all >= t_start) & (eth_t_all < d_end)
    btc_t_d = btc_t_all[bm]
    btc_p_d = btc_p_all[bm]
    eth_t_d = eth_t_all[em]
    eth_p_d = eth_p_all[em]

    # Prepare interleaved for time-rescaling
    bt_sec = (btc_t_d - t_start) / 1e6
    et_sec = (eth_t_d - t_start) / 1e6
    bt_marks = np.diff(np.log(btc_p_d.astype(np.float64)))
    et_marks = np.diff(np.log(eth_p_d.astype(np.float64)))
    bt_sec = bt_sec[1:]
    et_sec = et_sec[1:]

    n_b, n_e = len(bt_sec), len(et_sec)
    times = np.concatenate([bt_sec, et_sec])
    marks = np.concatenate([bt_marks, et_marks])
    assets = np.concatenate([np.zeros(n_b, dtype=np.int32), np.ones(n_e, dtype=np.int32)])
    order = np.argsort(times, kind="mergesort")
    times = times[order]
    marks = marks[order]
    assets = assets[order]

    # Subsample for tractability
    max_ev = 200_000
    if len(times) > max_ev:
        step = len(times) // max_ev
        times = times[::step]
        marks = marks[::step]
        assets = assets[::step]

    T = 86400.0

    # ── 1. Time-rescaling diagnostic ──
    print("\n1. Time-rescaling diagnostic")
    btc_Lambda, eth_Lambda = compute_compensator_increments(params, times, assets)

    diag_rows = []
    for label, Lambda in [("BTC", btc_Lambda), ("ETH", eth_Lambda)]:
        # Under correct model, Lambda should be ~Exp(1)
        Lambda_pos = Lambda[Lambda > 0]
        if len(Lambda_pos) < 100:
            print(f"  {label}: too few compensator increments")
            continue
        ks_stat, ks_p = stats.kstest(Lambda_pos, "expon", args=(0, 1))
        mean_L = np.mean(Lambda_pos)
        var_L = np.var(Lambda_pos)
        print(f"  {label}: KS stat={ks_stat:.4f}, p={ks_p:.4f}, mean={mean_L:.4f}, var={var_L:.4f}")
        diag_rows.append({
            "asset": label,
            "ks_statistic": ks_stat,
            "ks_p_value": ks_p,
            "mean_lambda": mean_L,
            "var_lambda": var_L,
            "n_increments": len(Lambda_pos),
        })

    # ── 2. Simulation-based goodness-of-fit ──
    print(f"\n2. Simulating {N_SIM_PATHS} paths for goodness-of-fit...")

    # Empirical HY on day 1 (subsampled)
    step_b = max(1, len(btc_t_d) // 50_000)
    step_e = max(1, len(eth_t_d) // 50_000)
    emp_cov, emp_corr = hayashi_yoshida_sim(
        (btc_t_d[::step_b] - t_start) / 1e6,
        np.log(btc_p_d[::step_b].astype(np.float64)),
        (eth_t_d[::step_e] - t_start) / 1e6,
        np.log(eth_p_d[::step_e].astype(np.float64)),
    )
    print(f"  Empirical HY correlation (day 1, subsampled): {emp_corr:.4f}")

    sim_corrs = []
    sim_ccfs = []

    for s in range(N_SIM_PATHS):
        bt_sim, bm_sim, et_sim, em_sim = simulate_hawkes(
            params, sigma_BTC, sigma_ETH, T, seed=42 + s
        )
        if len(bt_sim) < 100 or len(et_sim) < 100:
            continue

        # HY on simulated
        btc_lp_sim = np.cumsum(bm_sim)
        eth_lp_sim = np.cumsum(em_sim)
        btc_lp_sim = np.concatenate([[0], btc_lp_sim])
        eth_lp_sim = np.concatenate([[0], eth_lp_sim])

        _, corr_sim = hayashi_yoshida_sim(
            bt_sim, btc_lp_sim[:-1], et_sim, eth_lp_sim[:-1]
        )
        sim_corrs.append(corr_sim)

        # CCF on simulated
        ccf_sim = ccf_from_sim(bt_sim, bm_sim, et_sim, em_sim)
        sim_ccfs.append(ccf_sim)

        if (s + 1) % 20 == 0:
            print(f"    {s + 1}/{N_SIM_PATHS} paths done")

    sim_corrs = np.array(sim_corrs)
    sim_ccfs = np.array(sim_ccfs)

    print(f"  Simulated HY correlation: mean={np.mean(sim_corrs):.4f}, "
          f"std={np.std(sim_corrs):.4f}")
    print(f"  Empirical vs simulated: empirical={emp_corr:.4f}, "
          f"sim 95% CI=[{np.percentile(sim_corrs, 2.5):.4f}, {np.percentile(sim_corrs, 97.5):.4f}]")

    in_ci = np.percentile(sim_corrs, 2.5) <= emp_corr <= np.percentile(sim_corrs, 97.5)
    diag_rows.append({
        "asset": "HY_GOF",
        "ks_statistic": emp_corr,
        "ks_p_value": np.mean(sim_corrs),
        "mean_lambda": np.percentile(sim_corrs, 2.5),
        "var_lambda": np.percentile(sim_corrs, 97.5),
        "n_increments": len(sim_corrs),
    })

    pd.DataFrame(diag_rows).to_csv(os.path.join(RESULTS_DIR, "hawkes_diagnostics.csv"), index=False)

    # Save simulated CCFs for figure 12
    if len(sim_ccfs) > 0:
        lags = np.arange(-20, 21)
        ccf_summary = pd.DataFrame({
            "lag": lags,
            "sim_mean": np.mean(sim_ccfs, axis=0),
            "sim_lower": np.percentile(sim_ccfs, 2.5, axis=0),
            "sim_upper": np.percentile(sim_ccfs, 97.5, axis=0),
        })
        ccf_summary.to_csv(os.path.join(RESULTS_DIR, "hawkes_sim_ccf.csv"), index=False)

    print(f"\nSaved hawkes_diagnostics.csv, hawkes_sim_ccf.csv")


if __name__ == "__main__":
    main()
