"""
24_directional_predictability.py
Test directional predictability: does BTC trade direction predict ETH price
movement over the Hawkes excitation window (~400ms)?

Analyses 1-5 as specified, plus figures 13-15.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from numba import njit
from scipy import stats as sp_stats

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "processed")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
FIGURES_DIR = os.path.join(os.path.dirname(__file__), "..", "figures")

plt.rcParams.update({
    "font.family": "serif", "font.size": 11,
    "axes.grid": True, "grid.alpha": 0.3, "grid.color": "gray",
    "figure.dpi": 300, "savefig.bbox": "tight",
})
COLORS = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#F0E442", "#56B4E9"]

LOOKBACK_MS = 100
HORIZONS_MS = [50, 100, 200, 400, 800, 1000]
THRESHOLDS_BPS = [0, 1, 2, 5, 10]
SAMPLING_MS = 50  # grid spacing to avoid overlapping observations
TRAIN_DAYS = 5


def load_ticks(symbol):
    name = "btc_ticks_clean.parquet" if symbol == "BTC" else "eth_ticks_clean.parquet"
    df = pd.read_parquet(os.path.join(DATA_DIR, name), columns=["timestamp", "price"])
    return df["timestamp"].values, df["price"].values


def save_fig(fig, name):
    for ext in ["png", "pdf"]:
        fig.savefig(os.path.join(FIGURES_DIR, f"{name}.{ext}"))
    print(f"  Saved {name}")
    plt.close(fig)


@njit
def compute_signals(btc_t, btc_lp, eth_t, eth_lp,
                    grid, lookback_us, horizon_us):
    """Vectorized signal/target computation at grid points."""
    n = len(grid)
    X = np.empty(n)
    Y = np.empty(n)
    valid = np.empty(n, dtype=np.bool_)

    for i in range(n):
        t = grid[i]

        # BTC return: t - lookback to t
        j_now = np.searchsorted(btc_t, t) - 1
        j_past = np.searchsorted(btc_t, t - lookback_us) - 1
        if j_now < 0 or j_past < 0 or j_now >= len(btc_lp) or j_past >= len(btc_lp):
            valid[i] = False
            X[i] = 0.0
            Y[i] = 0.0
            continue
        X[i] = btc_lp[j_now] - btc_lp[j_past]

        # ETH return: t to t + horizon
        k_now = np.searchsorted(eth_t, t) - 1
        k_fwd = np.searchsorted(eth_t, t + horizon_us) - 1
        if k_now < 0 or k_fwd < 0 or k_now >= len(eth_lp) or k_fwd >= len(eth_lp):
            valid[i] = False
            Y[i] = 0.0
            continue
        Y[i] = eth_lp[k_fwd] - eth_lp[k_now]
        valid[i] = True

    return X, Y, valid


def ols(X, Y):
    """Simple OLS: Y = alpha + beta * X. Returns stats."""
    n = len(X)
    if n < 10:
        return None
    x_mean = np.mean(X)
    y_mean = np.mean(Y)
    ss_xx = np.sum((X - x_mean) ** 2)
    ss_xy = np.sum((X - x_mean) * (Y - y_mean))
    if ss_xx == 0:
        return None
    beta = ss_xy / ss_xx
    alpha = y_mean - beta * x_mean
    Y_hat = alpha + beta * X
    resid = Y - Y_hat
    ss_res = np.sum(resid ** 2)
    ss_tot = np.sum((Y - y_mean) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    sigma2 = ss_res / (n - 2)
    se_beta = np.sqrt(sigma2 / ss_xx)
    t_stat = beta / se_beta if se_beta > 0 else 0
    p_value = 2 * (1 - sp_stats.t.cdf(abs(t_stat), n - 2))
    return {
        "alpha": alpha, "beta": beta, "se_beta": se_beta,
        "t_statistic": t_stat, "p_value": p_value,
        "r_squared": r2, "n": n,
    }


@njit
def compute_hawkes_intensity_at_grid(params, btc_t, eth_t, grid):
    """Compute λ_ETH at each grid point using Hawkes recursive."""
    mu_B, mu_E = params[0], params[1]
    a_BB, a_BE, a_EB, a_EE = params[2], params[3], params[4], params[5]
    b_BB, b_BE, b_EB, b_EE = params[6], params[7], params[8], params[9]

    # Merge all event times
    n_b = len(btc_t)
    n_e = len(eth_t)
    n_total = n_b + n_e
    all_times = np.empty(n_total)
    all_assets = np.empty(n_total, dtype=np.int32)
    all_times[:n_b] = btc_t
    all_assets[:n_b] = 0
    all_times[n_b:] = eth_t
    all_assets[n_b:] = 1

    order = np.argsort(all_times)
    all_times = all_times[order]
    all_assets = all_assets[order]

    # Walk through events and grid points together
    n_grid = len(grid)
    intensities = np.empty(n_grid)

    R_EB = 0.0
    R_EE = 0.0
    prev_t = 0.0
    ev_idx = 0

    for g in range(n_grid):
        t_g = grid[g]

        # Process all events up to t_g
        while ev_idx < n_total and all_times[ev_idx] <= t_g:
            dt = all_times[ev_idx] - prev_t
            if dt > 0:
                R_EB *= np.exp(-b_EB * dt)
                R_EE *= np.exp(-b_EE * dt)
            if all_assets[ev_idx] == 0:  # BTC event
                R_EB += b_EB
            else:  # ETH event
                R_EE += b_EE
            prev_t = all_times[ev_idx]
            ev_idx += 1

        # Decay to grid point
        dt = t_g - prev_t
        if dt > 0:
            lam_E = mu_E + a_EB * R_EB * np.exp(-b_EB * dt) + a_EE * R_EE * np.exp(-b_EE * dt)
        else:
            lam_E = mu_E + a_EB * R_EB + a_EE * R_EE

        intensities[g] = lam_E

    return intensities


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    btc_t, btc_p = load_ticks("BTC")
    eth_t, eth_p = load_ticks("ETH")
    btc_lp = np.log(btc_p.astype(np.float64))
    eth_lp = np.log(eth_p.astype(np.float64))

    t_start = max(btc_t[0], eth_t[0])
    t_end = min(btc_t[-1], eth_t[-1])
    day_us = int(86400 * 1e6)

    # Train/test split
    train_end = t_start + TRAIN_DAYS * day_us
    print(f"Train: days 1-{TRAIN_DAYS}, Test: days {TRAIN_DAYS+1}-7")

    # Build grid (every SAMPLING_MS)
    sampling_us = SAMPLING_MS * 1000
    lookback_us = LOOKBACK_MS * 1000
    max_horizon_us = max(HORIZONS_MS) * 1000

    grid_full = np.arange(t_start + lookback_us, t_end - max_horizon_us, sampling_us)
    grid_train = grid_full[grid_full < train_end]
    grid_test = grid_full[grid_full >= train_end]
    print(f"Grid: {len(grid_full):,} total ({len(grid_train):,} train, {len(grid_test):,} test)")

    # ════════════════════════════════════════════════════
    # Analysis 1 & 2: Directional regression at multiple horizons
    # ════════════════════════════════════════════════════
    print("\n=== Analysis 1 & 2: Directional regression ===")
    horizon_rows = []

    for horizon_ms in HORIZONS_MS:
        horizon_us = horizon_ms * 1000
        X, Y, valid = compute_signals(btc_t, btc_lp, eth_t, eth_lp,
                                       grid_full, lookback_us, horizon_us)
        X, Y = X[valid], Y[valid]
        res = ols(X, Y)
        if res is None:
            continue
        res["horizon_ms"] = horizon_ms
        res["lookback_ms"] = LOOKBACK_MS
        horizon_rows.append(res)
        sig = "***" if res["p_value"] < 0.001 else ""
        print(f"  h={horizon_ms:>5}ms  β={res['beta']:.4f}  t={res['t_statistic']:.1f}  "
              f"R²={res['r_squared']:.5f}  n={res['n']:,} {sig}")

    horizon_df = pd.DataFrame(horizon_rows)
    horizon_df.to_csv(os.path.join(RESULTS_DIR, "horizon_analysis_hawkes.csv"), index=False)

    # Primary result at 400ms
    primary = horizon_df[horizon_df["horizon_ms"] == 400]
    if len(primary) > 0:
        p = primary.iloc[0]
        print(f"\n  Primary (400ms): β={p['beta']:.4f}, t={p['t_statistic']:.1f}, R²={p['r_squared']:.5f}")

    # Save directional_regression.csv (the 400ms result)
    if len(primary) > 0:
        primary.to_csv(os.path.join(RESULTS_DIR, "directional_regression.csv"), index=False)

    # ════════════════════════════════════════════════════
    # Analysis 3: Out-of-sample prediction
    # ════════════════════════════════════════════════════
    print("\n=== Analysis 3: Out-of-sample prediction ===")
    oos_rows = []

    for horizon_ms in HORIZONS_MS:
        horizon_us = horizon_ms * 1000

        # Train
        X_tr, Y_tr, v_tr = compute_signals(btc_t, btc_lp, eth_t, eth_lp,
                                             grid_train, lookback_us, horizon_us)
        X_tr, Y_tr = X_tr[v_tr], Y_tr[v_tr]
        res_tr = ols(X_tr, Y_tr)
        if res_tr is None:
            continue

        # Test
        X_te, Y_te, v_te = compute_signals(btc_t, btc_lp, eth_t, eth_lp,
                                             grid_test, lookback_us, horizon_us)
        X_te, Y_te = X_te[v_te], Y_te[v_te]

        # OOS prediction
        Y_pred = res_tr["alpha"] + res_tr["beta"] * X_te
        ss_res = np.sum((Y_te - Y_pred) ** 2)
        ss_tot = np.sum((Y_te - np.mean(Y_te)) ** 2)
        r2_oos = 1 - ss_res / ss_tot if ss_tot > 0 else 0

        oos_rows.append({
            "horizon_ms": horizon_ms,
            "beta_train": res_tr["beta"],
            "r2_insample": res_tr["r_squared"],
            "r2_oos": r2_oos,
            "n_train": len(X_tr),
            "n_test": len(X_te),
        })
        print(f"  h={horizon_ms:>5}ms  β_train={res_tr['beta']:.4f}  "
              f"R²_IS={res_tr['r_squared']:.5f}  R²_OOS={r2_oos:.5f}")

    oos_df = pd.DataFrame(oos_rows)
    oos_df.to_csv(os.path.join(RESULTS_DIR, "oos_prediction.csv"), index=False)

    # ════════════════════════════════════════════════════
    # Analysis 4: Profitability analysis (on TEST data only)
    # ════════════════════════════════════════════════════
    print("\n=== Analysis 4: Profitability analysis (test period) ===")
    horizon_us_400 = 400 * 1000
    X_te, Y_te, v_te = compute_signals(btc_t, btc_lp, eth_t, eth_lp,
                                         grid_test, lookback_us, horizon_us_400)
    X_te, Y_te = X_te[v_te], Y_te[v_te]

    duration_us = grid_test[-1] - grid_test[0]
    duration_years = duration_us / (1e6 * 86400 * 365)
    duration_days = duration_us / (1e6 * 86400)

    prof_rows = []
    for thr_bps in THRESHOLDS_BPS:
        thr = thr_bps * 1e-4
        if thr == 0:
            mask = np.ones(len(X_te), dtype=bool)
        else:
            mask = np.abs(X_te) > thr

        if np.sum(mask) < 100:
            continue

        signals = np.sign(X_te[mask])
        pnl = signals * Y_te[mask]

        n_trades = len(pnl)
        avg_ret = np.mean(pnl)
        std_ret = np.std(pnl)
        hit_rate = np.mean(pnl > 0)
        trades_per_year = n_trades / duration_years if duration_years > 0 else 0
        sharpe_gross = (avg_ret / std_ret) * np.sqrt(trades_per_year) if std_ret > 0 else 0

        # Break-even cost: avg gross return per trade
        breakeven_bps = avg_ret * 1e4

        prof_rows.append({
            "threshold_bps": thr_bps,
            "n_trades": n_trades,
            "trades_per_day": int(n_trades / duration_days) if duration_days > 0 else 0,
            "avg_return_bps": avg_ret * 1e4,
            "std_return_bps": std_ret * 1e4,
            "hit_rate": hit_rate,
            "sharpe_gross": sharpe_gross,
            "breakeven_cost_bps": breakeven_bps,
        })
        print(f"  thr={thr_bps:>2}bps  n={n_trades:>7,}  avg={avg_ret*1e4:.3f}bps  "
              f"hit={hit_rate:.1%}  SR={sharpe_gross:.1f}  BE={breakeven_bps:.3f}bps")

    prof_df = pd.DataFrame(prof_rows)
    prof_df.to_csv(os.path.join(RESULTS_DIR, "profitability_analysis.csv"), index=False)

    # ════════════════════════════════════════════════════
    # Analysis 5: Conditioning on Hawkes intensity
    # ════════════════════════════════════════════════════
    print("\n=== Analysis 5: Intensity-conditioned regression ===")

    # Load Hawkes params
    hawkes_df = pd.read_csv(os.path.join(RESULTS_DIR, "hawkes_parameter_estimates.csv"))
    pooled = hawkes_df[hawkes_df["day"] == "pooled"].iloc[0]
    hawkes_params = np.array([
        float(pooled["mu_BTC"]), float(pooled["mu_ETH"]),
        float(pooled["alpha_BTC_from_BTC"]), float(pooled["alpha_BTC_from_ETH"]),
        float(pooled["alpha_ETH_from_BTC"]), float(pooled["alpha_ETH_from_ETH"]),
        float(pooled["beta_BTC_from_BTC"]), float(pooled["beta_BTC_from_ETH"]),
        float(pooled["beta_ETH_from_BTC"]), float(pooled["beta_ETH_from_ETH"]),
    ])

    # Use a subset of grid for intensity computation (expensive)
    # Take every 10th point from full grid
    grid_sub = grid_full[::10]
    print(f"  Computing Hawkes intensity at {len(grid_sub):,} grid points...")

    # Convert grid to seconds for Hawkes computation
    btc_t_sec = (btc_t - t_start) / 1e6
    eth_t_sec = (eth_t - t_start) / 1e6
    grid_sub_sec = (grid_sub - t_start) / 1e6

    intensities = compute_hawkes_intensity_at_grid(
        hawkes_params, btc_t_sec, eth_t_sec, grid_sub_sec
    )

    # Get signals at these grid points
    X_sub, Y_sub, v_sub = compute_signals(btc_t, btc_lp, eth_t, eth_lp,
                                           grid_sub, lookback_us, horizon_us_400)

    mask_all = v_sub & np.isfinite(intensities) & (intensities > 0)
    X_s, Y_s, lam_s = X_sub[mask_all], Y_sub[mask_all], intensities[mask_all]

    median_lam = np.median(lam_s)
    hi_mask = lam_s >= median_lam
    lo_mask = lam_s < median_lam

    intensity_rows = []
    for label, m in [("all", np.ones(len(X_s), dtype=bool)),
                     ("high_intensity", hi_mask),
                     ("low_intensity", lo_mask)]:
        res = ols(X_s[m], Y_s[m])
        if res is None:
            continue
        res["group"] = label
        res["median_intensity"] = median_lam
        intensity_rows.append(res)
        print(f"  {label:20s}  β={res['beta']:.4f}  t={res['t_statistic']:.1f}  "
              f"R²={res['r_squared']:.5f}  n={res['n']:,}")

    int_df = pd.DataFrame(intensity_rows)
    int_df.to_csv(os.path.join(RESULTS_DIR, "intensity_conditioned.csv"), index=False)

    # ════════════════════════════════════════════════════
    # Figures
    # ════════════════════════════════════════════════════
    print("\n=== Generating figures ===")

    # Figure 13: β vs horizon
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    h = horizon_df["horizon_ms"].values
    b = horizon_df["beta"].values
    se = horizon_df["se_beta"].values
    r2 = horizon_df["r_squared"].values * 100

    ax1.errorbar(h, b, yerr=1.96 * se, fmt="o-", color=COLORS[0], markersize=5, capsize=4)
    ax1.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax1.axvline(400, color=COLORS[1], linestyle=":", linewidth=1, alpha=0.7, label="Hawkes 1/β = 408ms")
    ax1.set_xlabel("Forward Horizon (ms)")
    ax1.set_ylabel("β coefficient")
    ax1.set_title("BTC→ETH Directional Predictability")
    ax1.legend(fontsize=9)

    ax2.plot(h, r2, "s-", color=COLORS[2], markersize=5)
    ax2.axvline(400, color=COLORS[1], linestyle=":", linewidth=1, alpha=0.7)
    ax2.set_xlabel("Forward Horizon (ms)")
    ax2.set_ylabel("R² (%)")
    ax2.set_title("Variance Explained")

    fig.tight_layout()
    save_fig(fig, "fig13_beta_vs_horizon")

    # Figure 14: Cumulative PnL (test period, 400ms, threshold=0)
    if len(X_te) > 0:
        signals_all = np.sign(X_te)
        pnl_all = signals_all * Y_te
        cum_pnl = np.cumsum(pnl_all) * 1e4  # in bps

        fig, ax = plt.subplots(figsize=(8, 4))
        trade_idx = np.arange(len(cum_pnl))
        ax.plot(trade_idx, cum_pnl, "-", color=COLORS[0], linewidth=0.5)
        ax.axhline(0, color="gray", linewidth=0.5)
        ax.set_xlabel("Trade number (test period)")
        ax.set_ylabel("Cumulative PnL (bps)")
        ax.set_title("Cumulative PnL — BTC Direction → ETH 400ms (OOS)")
        # Add daily markers
        n_per_day = len(cum_pnl) // 2 if duration_days > 0 else len(cum_pnl)
        for d in range(1, 3):
            ax.axvline(d * n_per_day, color="gray", linestyle=":", linewidth=0.5, alpha=0.5)
        save_fig(fig, "fig14_cumulative_pnl")

    # Figure 15: β high vs low intensity
    if len(intensity_rows) >= 3:
        fig, ax = plt.subplots(figsize=(6, 4))
        groups = ["low_intensity", "all", "high_intensity"]
        group_labels = ["Low λ_ETH", "All", "High λ_ETH"]
        betas = []
        errs = []
        for g in groups:
            row = int_df[int_df["group"] == g]
            if len(row) > 0:
                betas.append(row.iloc[0]["beta"])
                errs.append(1.96 * row.iloc[0]["se_beta"])
            else:
                betas.append(0)
                errs.append(0)

        x_pos = np.arange(len(groups))
        bars = ax.bar(x_pos, betas, yerr=errs, capsize=6,
                      color=[COLORS[5], COLORS[0], COLORS[1]], alpha=0.8)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(group_labels)
        ax.set_ylabel("β coefficient")
        ax.set_title("Predictability Conditioned on Hawkes Intensity")
        ax.axhline(0, color="gray", linewidth=0.5)

        # Add significance annotations
        for i, g in enumerate(groups):
            row = int_df[int_df["group"] == g]
            if len(row) > 0:
                t = row.iloc[0]["t_statistic"]
                ax.text(i, betas[i] + errs[i] + 0.002, f"t={t:.0f}",
                        ha="center", va="bottom", fontsize=9)

        save_fig(fig, "fig15_intensity_conditioned")

    print("\nAll analyses complete.")


if __name__ == "__main__":
    main()
