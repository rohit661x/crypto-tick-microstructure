"""
19_generate_report.py
Generate final report and all figures for the rigorous analysis.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
FIGURES_DIR = os.path.join(os.path.dirname(__file__), "..", "figures")

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.color": "gray",
    "figure.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.1,
})

COLORS = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#F0E442", "#56B4E9"]


def save_fig(fig, name):
    for ext in ["png", "pdf"]:
        fig.savefig(os.path.join(FIGURES_DIR, f"{name}.{ext}"))
    print(f"  Saved {name}")
    plt.close(fig)


def fig5_contrast_function():
    df = pd.read_csv(os.path.join(RESULTS_DIR, "contrast_function.csv"))
    est = pd.read_csv(os.path.join(RESULTS_DIR, "lead_lag_estimates.csv"))

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(df["theta_ms"], df["contrast_normalized"], "-", color=COLORS[0], linewidth=1.5)

    theta_hat = est["theta_hat_ms"].values[0]
    ax.axvline(theta_hat, color=COLORS[1], linestyle="--", linewidth=1, alpha=0.8,
               label=f"$\\hat{{\\theta}}$ = {theta_hat:.0f}ms")
    ax.axvline(0, color="gray", linestyle=":", linewidth=0.8)

    if "ci_lower_ms" in est.columns:
        ci_lo = est["ci_lower_ms"].values[0]
        ci_hi = est["ci_upper_ms"].values[0]
        ax.axvspan(ci_lo, ci_hi, alpha=0.1, color=COLORS[1], label=f"95% CI [{ci_lo:.0f}, {ci_hi:.0f}]ms")

    ax.set_xlabel("Lead-Lag θ (ms)")
    ax.set_ylabel("Normalized Contrast U_n(θ)")
    ax.set_title("HRY Lead-Lag Contrast Function")
    ax.legend(fontsize=9)
    save_fig(fig, "fig5_contrast_function")


def fig6_granger_fstats():
    df = pd.read_csv(os.path.join(RESULTS_DIR, "granger_results.csv"))

    # Plot for window=50ms
    window = 50
    sub = df[df["window_ms"] == window]
    if len(sub) == 0:
        window = df["window_ms"].min()
        sub = df[df["window_ms"] == window]

    fig, ax = plt.subplots(figsize=(6, 4))

    be = sub[sub["direction"] == "BTC->ETH"]
    eb = sub[sub["direction"] == "ETH->BTC"]

    ax.plot(be["lag_ms"], be["f_statistic"], "o-", color=COLORS[0], label="BTC → ETH", markersize=4)
    ax.plot(eb["lag_ms"], eb["f_statistic"], "s-", color=COLORS[1], label="ETH → BTC", markersize=4)

    # 5% critical value for F(1, n) ≈ 3.84
    ax.axhline(3.84, color="gray", linestyle="--", linewidth=0.8, label="5% critical value")

    ax.set_xlabel(f"Lag (ms)")
    ax.set_ylabel("F-statistic")
    ax.set_title(f"Granger Causality F-statistics ({window}ms windows)")
    ax.legend(fontsize=9)
    save_fig(fig, "fig6_granger_fstats")


def fig7_predictability_decay():
    df = pd.read_csv(os.path.join(RESULTS_DIR, "predictive_regression.csv"))
    if len(df) == 0:
        print("  No predictive regression data, skipping fig7")
        return

    # Pick best lookback
    best_lb = df.groupby("lookback_ms")["r2_oos"].max().idxmax()
    sub = df[df["lookback_ms"] == best_lb].sort_values("horizon_ms")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    # R² OOS vs horizon
    ax1.plot(sub["horizon_ms"], sub["r2_oos"] * 100, "o-", color=COLORS[0], markersize=5)
    ax1.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax1.set_xlabel("Forward Horizon (ms)")
    ax1.set_ylabel("Out-of-Sample R² (%)")
    ax1.set_title(f"Predictability Decay (lookback={int(best_lb)}ms)")

    # t-stat vs horizon
    ax2.plot(sub["horizon_ms"], sub["t_statistic"], "s-", color=COLORS[1], markersize=5)
    ax2.axhline(1.96, color="gray", linestyle="--", linewidth=0.8, label="5% threshold")
    ax2.axhline(-1.96, color="gray", linestyle="--", linewidth=0.8)
    ax2.set_xlabel("Forward Horizon (ms)")
    ax2.set_ylabel("t-statistic")
    ax2.set_title(f"Statistical Significance Decay")
    ax2.legend(fontsize=9)

    fig.tight_layout()
    save_fig(fig, "fig7_predictability_decay")


def fig8_oos_r2_heatmap():
    df = pd.read_csv(os.path.join(RESULTS_DIR, "horizon_analysis.csv"))
    if len(df) == 0:
        print("  No horizon analysis data, skipping fig8")
        return

    pivot = df.pivot_table(index="lookback_ms", columns="horizon_ms", values="r2_oos")

    fig, ax = plt.subplots(figsize=(6, 4))
    im = ax.imshow(pivot.values * 100, cmap="RdYlGn", aspect="auto",
                   origin="lower")

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f"{int(c)}ms" for c in pivot.columns])
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([f"{int(i)}ms" for i in pivot.index])
    ax.set_xlabel("Forward Horizon")
    ax.set_ylabel("Lookback Window")
    ax.set_title("Out-of-Sample R² (%) — BTC→ETH Predictive Regression")

    # Annotate cells
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j] * 100
            color = "white" if abs(val) > 1 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=8, color=color)

    fig.colorbar(im, ax=ax, label="R² (%)")
    save_fig(fig, "fig8_oos_r2_heatmap")


def generate_text_report():
    """Generate final_report.txt."""
    lines = []
    lines.append("=" * 65)
    lines.append("ULTRA-LOW-LATENCY BTC/ETH DEPENDENCE ANALYSIS: FINAL RESULTS")
    lines.append("=" * 65)

    # 1. Covariance
    try:
        cov = pd.read_csv(os.path.join(RESULTS_DIR, "covariance_estimates.csv"))
        lines.append("\n1. COVARIANCE ESTIMATION")
        lines.append("-" * 40)
        for _, r in cov.iterrows():
            lines.append(f"  {r['method']:20s}  ρ = {r['correlation']:.4f}  "
                         f"[{r['corr_ci_lower']:.4f}, {r['corr_ci_upper']:.4f}]")
        sync_r = cov.loc[cov["method"] == "refresh_100ms", "correlation"].values
        hy_r = cov.loc[cov["method"] == "hayashi_yoshida", "correlation"].values
        if len(sync_r) > 0 and len(hy_r) > 0 and hy_r[0] != 0:
            lines.append(f"  Attenuation factor (sync/HY): {sync_r[0] / hy_r[0]:.2f}")
    except Exception:
        lines.append("\n1. COVARIANCE ESTIMATION: not available")

    # 2. Lead-lag
    try:
        ll = pd.read_csv(os.path.join(RESULTS_DIR, "lead_lag_estimates.csv"))
        r = ll.iloc[0]
        lines.append("\n2. LEAD-LAG STRUCTURE")
        lines.append("-" * 40)
        lines.append(f"  Estimated lead-lag: θ = {r['theta_hat_ms']:.0f}ms")
        lines.append(f"  95% CI: [{r['ci_lower_ms']:.0f}ms, {r['ci_upper_ms']:.0f}ms]")
        lines.append(f"  t-statistic: {r['t_statistic']:.2f}")
        lines.append(f"  p-value: {r['p_value']:.4f}")
        sig = "Reject H0" if r["significant_5pct"] else "Cannot reject H0"
        direction = "BTC leads ETH" if r["theta_hat_ms"] > 0 else "ETH leads BTC" if r["theta_hat_ms"] < 0 else "Contemporaneous"
        lines.append(f"  Conclusion: {sig} — {direction}")
    except Exception:
        lines.append("\n2. LEAD-LAG: not available")

    # 3. Granger
    try:
        gc = pd.read_csv(os.path.join(RESULTS_DIR, "granger_results.csv"))
        lines.append("\n3. GRANGER CAUSALITY")
        lines.append("-" * 40)
        for direction in ["BTC->ETH", "ETH->BTC"]:
            sub = gc[(gc["direction"] == direction) & (gc["window_ms"] == gc["window_ms"].min())]
            sig_lags = sub[sub["significant_5pct"]]["lag_ms"].values
            if len(sig_lags) > 0:
                lines.append(f"  {direction}: Significant at lags {', '.join(str(int(l)) + 'ms' for l in sig_lags[:5])}")
                max_f = sub.loc[sub["f_statistic"].idxmax()]
                lines.append(f"    Peak F={max_f['f_statistic']:.1f} at {int(max_f['lag_ms'])}ms (p={max_f['p_value']:.2e})")
            else:
                lines.append(f"  {direction}: Not significant at any lag")
    except Exception:
        lines.append("\n3. GRANGER CAUSALITY: not available")

    # 4. Information share
    try:
        info = pd.read_csv(os.path.join(RESULTS_DIR, "information_shares.csv"))
        lines.append("\n4. INFORMATION SHARE (Hasbrouck)")
        lines.append("-" * 40)
        for _, r in info.iterrows():
            lines.append(f"  Δ = {int(r['sampling_interval_ms'])}ms: "
                         f"IS(BTC) = {r['is_btc']:.1%} [{r['is_btc_lower']:.1%}, {r['is_btc_upper']:.1%}]  "
                         f"IS(ETH) = {r['is_eth']:.1%}  "
                         f"Cointegrated: {'Yes' if r['cointegrated'] else 'No'}")
    except Exception:
        lines.append("\n4. INFORMATION SHARE: not available")

    # 5. Predictability
    try:
        pr = pd.read_csv(os.path.join(RESULTS_DIR, "predictive_regression.csv"))
        lines.append("\n5. PREDICTABILITY (OOS)")
        lines.append("-" * 40)
        if len(pr) > 0:
            best = pr.loc[pr["r2_oos"].idxmax()]
            lines.append(f"  Best model: lookback={int(best['lookback_ms'])}ms, horizon={int(best['horizon_ms'])}ms")
            lines.append(f"    β = {best['beta']:.4f} (SE = {best['se_beta']:.4f})")
            lines.append(f"    t-stat = {best['t_statistic']:.1f} (p = {best['p_value']:.2e})")
            lines.append(f"    In-sample R² = {best['r2_insample']:.4f}")
            lines.append(f"    OOS R² = {best['r2_oos']:.4f}")

            # Decay
            best_lb = best["lookback_ms"]
            sub = pr[pr["lookback_ms"] == best_lb].sort_values("horizon_ms")
            sig_horizons = sub[sub["significant_5pct"]]["horizon_ms"].values
            if len(sig_horizons) > 0:
                lines.append(f"    Significant horizons: {', '.join(str(int(h)) + 'ms' for h in sig_horizons)}")
                lines.append(f"    Predictability half-life: ~{int(sig_horizons[-1])}ms")
    except Exception:
        lines.append("\n5. PREDICTABILITY: not available")

    # 6. Economic significance
    try:
        bt = pd.read_csv(os.path.join(RESULTS_DIR, "backtest_results.csv"))
        lines.append("\n6. ECONOMIC SIGNIFICANCE")
        lines.append("-" * 40)
        for _, r in bt.iterrows():
            lines.append(f"  Config: lb={int(r['lookback_ms'])}ms h={int(r['horizon_ms'])}ms thr={r['threshold_bps']}bps")
            lines.append(f"    Trades/day: {int(r['trades_per_day']):,}  Hit: {r['hit_rate']:.1%}  "
                         f"Gross SR: {r['gross_sharpe']:.2f}  Net SR: {r['net_sharpe']:.2f}")
    except Exception:
        lines.append("\n6. ECONOMIC SIGNIFICANCE: not available")

    lines.append("\n" + "=" * 65)

    report = "\n".join(lines)
    with open(os.path.join(RESULTS_DIR, "final_report.txt"), "w") as f:
        f.write(report + "\n")
    print(report)


def main():
    os.makedirs(FIGURES_DIR, exist_ok=True)

    print("Generating figures...")
    try:
        fig5_contrast_function()
    except Exception as e:
        print(f"  fig5 failed: {e}")

    try:
        fig6_granger_fstats()
    except Exception as e:
        print(f"  fig6 failed: {e}")

    try:
        fig7_predictability_decay()
    except Exception as e:
        print(f"  fig7 failed: {e}")

    try:
        fig8_oos_r2_heatmap()
    except Exception as e:
        print(f"  fig8 failed: {e}")

    print("\nGenerating report...")
    generate_text_report()
    print("\nSaved final_report.txt")


if __name__ == "__main__":
    main()
