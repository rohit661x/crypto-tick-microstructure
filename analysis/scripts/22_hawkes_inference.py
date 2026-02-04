"""
22_hawkes_inference.py
Formal inference on Hawkes model:
  - Wald tests for off-diagonal α (mutual excitation)
  - Asymmetry test: α_{ETH←BTC} - α_{BTC←ETH}
  - Comparison with VECM information shares
  - Figures 9-12
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from numba import njit
from scipy import stats

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
FIGURES_DIR = os.path.join(os.path.dirname(__file__), "..", "figures")
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "processed")

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.color": "gray",
    "figure.dpi": 300,
    "savefig.bbox": "tight",
})

COLORS = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#F0E442", "#56B4E9"]


def save_fig(fig, name):
    for ext in ["png", "pdf"]:
        fig.savefig(os.path.join(FIGURES_DIR, f"{name}.{ext}"))
    print(f"  Saved {name}")
    plt.close(fig)


def load_estimates():
    df = pd.read_csv(os.path.join(RESULTS_DIR, "hawkes_parameter_estimates.csv"))
    pooled = df[df["day"] == "pooled"].iloc[0]
    days = df[df["day"] != "pooled"]
    return pooled, days


def load_ticks(symbol):
    name = "btc_ticks_clean.parquet" if symbol == "BTC" else "eth_ticks_clean.parquet"
    df = pd.read_parquet(os.path.join(DATA_DIR, name), columns=["timestamp", "price"])
    return df["timestamp"].values, df["price"].values


@njit
def compute_intensities(params, times, assets, T):
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
    mu_B, mu_E = params[0], params[1]
    a_BB, a_BE, a_EB, a_EE = params[2], params[3], params[4], params[5]
    b_BB, b_BE, b_EB, b_EE = params[6], params[7], params[8], params[9]
    n = len(times)
    R_BB = 0.0
    R_BE = 0.0
    R_EB = 0.0
    R_EE = 0.0
    prev_t = 0.0

    btc_times_arr = np.empty(n)
    btc_R_BB = np.empty(n)
    btc_R_BE = np.empty(n)
    n_btc = 0
    eth_times_arr = np.empty(n)
    eth_R_EB = np.empty(n)
    eth_R_EE = np.empty(n)
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
            btc_times_arr[n_btc] = t
            btc_R_BB[n_btc] = R_BB
            btc_R_BE[n_btc] = R_BE
            n_btc += 1
            R_BB += b_BB
            R_EB += b_EB
        else:
            eth_times_arr[n_eth] = t
            eth_R_EB[n_eth] = R_EB
            eth_R_EE[n_eth] = R_EE
            n_eth += 1
            R_BE += b_BE
            R_EE += b_EE
        prev_t = t

    btc_Lambda = np.empty(n_btc - 1)
    for i in range(1, n_btc):
        dt = btc_times_arr[i] - btc_times_arr[i - 1]
        lam_s = mu_B + a_BB * btc_R_BB[i - 1] + a_BE * btc_R_BE[i - 1]
        lam_e = mu_B + a_BB * btc_R_BB[i] + a_BE * btc_R_BE[i]
        btc_Lambda[i - 1] = 0.5 * (lam_s + lam_e) * dt

    eth_Lambda = np.empty(n_eth - 1)
    for i in range(1, n_eth):
        dt = eth_times_arr[i] - eth_times_arr[i - 1]
        lam_s = mu_E + a_EB * eth_R_EB[i - 1] + a_EE * eth_R_EE[i - 1]
        lam_e = mu_E + a_EB * eth_R_EB[i] + a_EE * eth_R_EE[i]
        eth_Lambda[i - 1] = 0.5 * (lam_s + lam_e) * dt

    return btc_Lambda, eth_Lambda


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    pooled, days = load_estimates()
    params = np.array([
        float(pooled["mu_BTC"]), float(pooled["mu_ETH"]),
        float(pooled["alpha_BTC_from_BTC"]), float(pooled["alpha_BTC_from_ETH"]),
        float(pooled["alpha_ETH_from_BTC"]), float(pooled["alpha_ETH_from_ETH"]),
        float(pooled["beta_BTC_from_BTC"]), float(pooled["beta_BTC_from_ETH"]),
        float(pooled["beta_ETH_from_BTC"]), float(pooled["beta_ETH_from_ETH"]),
    ])

    results = []

    # ── Wald tests ──
    print("Wald tests on cross-excitation parameters:")
    for param_name, label in [
        ("alpha_ETH_from_BTC", "H0: α_{ETH←BTC}=0 (BTC does not excite ETH)"),
        ("alpha_BTC_from_ETH", "H0: α_{BTC←ETH}=0 (ETH does not excite BTC)"),
    ]:
        val = float(pooled[param_name])
        se = float(pooled[f"{param_name}_se"])
        if se > 0 and not np.isnan(se):
            wald = (val / se) ** 2
            p_val = 1 - stats.chi2.cdf(wald, df=1)
        else:
            wald = np.inf
            p_val = 0.0
        sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
        print(f"  {label}")
        print(f"    Estimate = {val:.4f}, SE = {se:.4f}, Wald χ²(1) = {wald:.2f}, p = {p_val:.2e} {sig}")
        results.append({
            "test": f"Wald: {param_name}=0",
            "estimate": val, "se": se,
            "test_statistic": wald, "df": 1,
            "p_value": p_val, "significant_5pct": p_val < 0.05,
        })

    # ── Asymmetry test ──
    a_EB = float(pooled["alpha_ETH_from_BTC"])
    a_BE = float(pooled["alpha_BTC_from_ETH"])
    diff = a_EB - a_BE
    day_diffs = days["alpha_ETH_from_BTC"].astype(float).values - days["alpha_BTC_from_ETH"].astype(float).values
    n_days = len(day_diffs)
    boot_diffs = np.array([np.mean(day_diffs[np.random.randint(0, n_days, n_days)]) for _ in range(2000)])
    ci_lo, ci_hi = np.percentile(boot_diffs, [2.5, 97.5])
    se_diff = np.std(boot_diffs)
    t_asym = diff / se_diff if se_diff > 0 else np.inf
    p_asym = 2 * (1 - stats.norm.cdf(abs(t_asym)))

    print(f"\nAsymmetry test: α_EB - α_BE = {diff:.4f}")
    print(f"  Bootstrap SE = {se_diff:.4f}, 95% CI = [{ci_lo:.4f}, {ci_hi:.4f}]")
    print(f"  t = {t_asym:.2f}, p = {p_asym:.4f}")
    direction = "BTC excites ETH more" if diff > 0 else "ETH excites BTC more"
    print(f"  → {direction}")
    results.append({
        "test": "Asymmetry: α_EB - α_BE",
        "estimate": diff, "se": se_diff,
        "test_statistic": t_asym, "df": n_days - 1,
        "p_value": p_asym, "significant_5pct": p_asym < 0.05,
    })

    # ── Comparison table ──
    print("\n" + "=" * 60)
    print("Hawkes branching vs VECM information shares")
    print("=" * 60)
    print(f"  α_EB (BTC→ETH excitation): {a_EB:.4f}")
    print(f"  α_BE (ETH→BTC excitation): {a_BE:.4f}")

    b_EB = float(pooled["beta_ETH_from_BTC"])
    b_BE = float(pooled["beta_BTC_from_ETH"])
    b_BB = float(pooled["beta_BTC_from_BTC"])
    b_EE = float(pooled["beta_ETH_from_ETH"])
    print(f"\n  Decay timescales:")
    print(f"    BTC→ETH: 1/β = {1000/b_EB:.1f}ms")
    print(f"    ETH→BTC: 1/β = {1000/b_BE:.1f}ms")
    print(f"    BTC self: 1/β = {1000/b_BB:.1f}ms")
    print(f"    ETH self: 1/β = {1000/b_EE:.1f}ms")

    try:
        info_df = pd.read_csv(os.path.join(RESULTS_DIR, "information_shares.csv"))
        coint = info_df[info_df["cointegrated"] == True]
        if len(coint) > 0:
            r = coint.iloc[0]
            print(f"\n  VECM IS (Δ={int(r['sampling_interval_ms'])}ms): BTC={r['is_btc']:.1%}, ETH={r['is_eth']:.1%}")
    except Exception:
        pass

    pd.DataFrame(results).to_csv(os.path.join(RESULTS_DIR, "hawkes_inference_results.csv"), index=False)

    # ── Figures ──
    print("\nGenerating figures...")

    # Fig 9: Impulse response functions
    t_ms = np.linspace(0, 500, 1000)
    t_s = t_ms / 1000.0

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(t_ms, a_EB * b_EB * np.exp(-b_EB * t_s), "-", color=COLORS[0], linewidth=2,
            label=f"BTC → ETH (α={a_EB:.3f}, 1/β={1000/b_EB:.0f}ms)")
    ax.plot(t_ms, a_BE * b_BE * np.exp(-b_BE * t_s), "-", color=COLORS[1], linewidth=2,
            label=f"ETH → BTC (α={a_BE:.3f}, 1/β={1000/b_BE:.0f}ms)")
    ax.plot(t_ms, float(pooled["alpha_BTC_from_BTC"]) * b_BB * np.exp(-b_BB * t_s),
            "--", color=COLORS[0], linewidth=1, alpha=0.5, label="BTC self")
    ax.plot(t_ms, float(pooled["alpha_ETH_from_ETH"]) * b_EE * np.exp(-b_EE * t_s),
            "--", color=COLORS[1], linewidth=1, alpha=0.5, label="ETH self")
    ax.set_xlabel("Time since event (ms)")
    ax.set_ylabel("Excitation rate α·φ(t)")
    ax.set_title("Hawkes Impulse Response Functions")
    ax.legend(fontsize=8)
    save_fig(fig, "fig9_impulse_response")

    # Fig 10: Intensity over 60-second window
    btc_t_all, btc_p_all = load_ticks("BTC")
    eth_t_all, eth_p_all = load_ticks("ETH")
    t0 = max(btc_t_all[0], eth_t_all[0])
    window_us = int(60 * 1e6)

    bm = (btc_t_all >= t0) & (btc_t_all < t0 + window_us)
    em = (eth_t_all >= t0) & (eth_t_all < t0 + window_us)
    bt_w = (btc_t_all[bm] - t0) / 1e6
    et_w = (eth_t_all[em] - t0) / 1e6
    bt_m = np.diff(np.log(btc_p_all[bm].astype(np.float64)))
    et_m = np.diff(np.log(eth_p_all[em].astype(np.float64)))
    bt_w, et_w = bt_w[1:], et_w[1:]

    n_b, n_e = len(bt_w), len(et_w)
    w_times = np.concatenate([bt_w, et_w])
    w_assets = np.concatenate([np.zeros(n_b, dtype=np.int32), np.ones(n_e, dtype=np.int32)])
    w_order = np.argsort(w_times, kind="mergesort")
    w_times = w_times[w_order]
    w_assets = w_assets[w_order]

    w_int = compute_intensities(params, w_times, w_assets, 60.0)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 5), sharex=True,
                                    gridspec_kw={"height_ratios": [3, 1]})
    bm2 = w_assets == 0
    em2 = w_assets == 1
    ax1.plot(w_times[bm2] * 1000, w_int[bm2], ".", color=COLORS[0], markersize=0.5, alpha=0.4, label="λ_BTC")
    ax1.plot(w_times[em2] * 1000, w_int[em2], ".", color=COLORS[1], markersize=0.5, alpha=0.4, label="λ_ETH")
    ax1.set_ylabel("Intensity (events/sec)")
    ax1.set_title("Estimated Intensity — 60-Second Window")
    ax1.legend(fontsize=9, markerscale=10)

    ax2.eventplot([w_times[bm2] * 1000], lineoffsets=1, colors=COLORS[0], linewidths=0.2)
    ax2.eventplot([w_times[em2] * 1000], lineoffsets=0, colors=COLORS[1], linewidths=0.2)
    ax2.set_yticks([0, 1])
    ax2.set_yticklabels(["ETH", "BTC"])
    ax2.set_xlabel("Time (ms)")
    fig.tight_layout()
    save_fig(fig, "fig10_intensity_window")

    # Fig 11: QQ plot
    btc_Lambda, eth_Lambda = compute_compensator_increments(params, w_times, w_assets)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    for ax, Lambda, label in [(ax1, btc_Lambda, "BTC"), (ax2, eth_Lambda, "ETH")]:
        Lambda_pos = Lambda[Lambda > 0]
        n_pts = min(len(Lambda_pos), 5000)
        if n_pts > 10:
            theoretical = stats.expon.ppf(np.linspace(0.01, 0.99, n_pts))
            empirical = np.sort(np.random.choice(Lambda_pos, n_pts, replace=False))
            ax.plot(theoretical, empirical, ".", markersize=1, color=COLORS[0])
            lim = max(theoretical.max(), empirical.max()) * 1.05
            ax.plot([0, lim], [0, lim], "r--", linewidth=0.8)
        ax.set_xlabel("Theoretical (Exp(1))")
        ax.set_ylabel("Empirical")
        ax.set_title(f"QQ — {label} Rescaled Inter-arrivals")
    fig.tight_layout()
    save_fig(fig, "fig11_residual_qq")

    # Fig 12: Empirical CCF vs simulated band
    try:
        sim_ccf = pd.read_csv(os.path.join(RESULTS_DIR, "hawkes_sim_ccf.csv"))
        emp_ccf = pd.read_csv(os.path.join(RESULTS_DIR, "ccf_curves.csv"))
        emp_500 = emp_ccf[emp_ccf["sampling_interval_ms"] == 500]

        fig, ax = plt.subplots(figsize=(6, 4))
        lags_ms = sim_ccf["lag"].values * 500
        ax.fill_between(lags_ms, sim_ccf["sim_lower"], sim_ccf["sim_upper"],
                        alpha=0.3, color=COLORS[2], label="Simulated 95% band")
        ax.plot(lags_ms, sim_ccf["sim_mean"], "-", color=COLORS[2], linewidth=1, label="Simulated mean")
        if len(emp_500) > 0:
            ax.plot(emp_500["lag_ms"], emp_500["ccf_value"], "o-", color=COLORS[0],
                    markersize=3, linewidth=1.5, label="Empirical")
        ax.axhline(0, color="gray", linewidth=0.5)
        ax.axvline(0, color="gray", linewidth=0.5, alpha=0.3)
        ax.set_xlabel("Lag (ms)")
        ax.set_ylabel("Cross-Correlation")
        ax.set_title("Empirical vs Simulated CCF (Hawkes Model)")
        ax.legend(fontsize=9)
        save_fig(fig, "fig12_ccf_comparison")
    except Exception as e:
        print(f"  fig12 skipped: {e}")

    print(f"\nSaved hawkes_inference_results.csv and figures 9-12")


if __name__ == "__main__":
    main()
