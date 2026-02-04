"""
06_generate_figures.py
Generate publication-quality figures.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from scipy.optimize import curve_fit

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
FIGURES_DIR = os.path.join(os.path.dirname(__file__), "..", "figures")

# Style
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
    print(f"  Saved {name}.png and {name}.pdf")
    plt.close(fig)


def fig1_epps_effect():
    """Correlation vs sampling frequency with CI band."""
    df = pd.read_csv(os.path.join(RESULTS_DIR, "correlation_vs_frequency.csv"))

    fig, ax = plt.subplots(figsize=(6, 4))
    x = df["sampling_interval_ms"].values
    y = df["correlation"].values
    lo = df["ci_lower"].values
    hi = df["ci_upper"].values

    ax.fill_between(x, lo, hi, alpha=0.2, color=COLORS[0])
    ax.plot(x, y, "o-", color=COLORS[0], markersize=5, linewidth=1.5)

    ax.set_xscale("log")
    ax.set_xlabel("Sampling Interval Δ (ms)")
    ax.set_ylabel("Pearson Correlation (ρ)")
    ax.set_title("Correlation Attenuation in BTC/ETH at High Frequencies")

    # Custom tick labels
    labels = {100: "100ms", 250: "250ms", 500: "500ms", 1000: "1s",
              2000: "2s", 5000: "5s", 10000: "10s", 30000: "30s",
              60000: "1m", 300000: "5m"}
    ax.set_xticks(list(labels.keys()))
    ax.set_xticklabels(list(labels.values()), rotation=45, ha="right", fontsize=9)
    ax.xaxis.set_major_formatter(ScalarFormatter())

    save_fig(fig, "fig1_epps_effect")


def fig2_ccf_degradation():
    """Overlay CCF curves for different sampling frequencies."""
    df = pd.read_csv(os.path.join(RESULTS_DIR, "ccf_curves.csv"))
    intervals = sorted(df["sampling_interval_ms"].unique())

    fig, ax = plt.subplots(figsize=(10, 4))

    for i, interval_ms in enumerate(intervals):
        sub = df[df["sampling_interval_ms"] == interval_ms]
        label_map = {500: "500ms", 1000: "1s", 2000: "2s",
                     5000: "5s", 10000: "10s", 30000: "30s"}
        label = label_map.get(interval_ms, f"{interval_ms}ms")
        ax.plot(sub["lag_ms"], sub["ccf_value"], "-", color=COLORS[i % len(COLORS)],
                label=f"Δ = {label}", linewidth=1.2)

    # Significance thresholds (approximate — use largest n for most conservative)
    peaks_df = pd.read_csv(os.path.join(RESULTS_DIR, "ccf_peaks.csv"))
    max_n = peaks_df["n_observations"].max()
    sig = 1.96 / np.sqrt(max_n)
    ax.axhline(sig, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.axhline(-sig, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.axvline(0, color="black", linewidth=0.5, alpha=0.3)

    ax.set_xlabel("Lag (ms)")
    ax.set_ylabel("Cross-Correlation")
    ax.set_title("Cross-Correlation Function at Various Sampling Frequencies")
    ax.legend(fontsize=9, loc="upper right")

    save_fig(fig, "fig2_ccf_degradation")


def fig3_leadlag_recovery():
    """Lead-lag estimate vs sampling frequency with bootstrap CIs."""
    df = pd.read_csv(os.path.join(RESULTS_DIR, "ccf_peaks.csv"))

    fig, ax = plt.subplots(figsize=(6, 4))

    x = df["sampling_interval_ms"].values
    y = df["peak_lag_ms"].values
    lo = df["boot_peak_lag_ms_ci_lower"].values
    hi = df["boot_peak_lag_ms_ci_upper"].values
    yerr = np.array([y - lo, hi - y])

    ax.errorbar(x, y, yerr=yerr, fmt="o-", color=COLORS[1], markersize=5,
                linewidth=1.5, capsize=4)
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)

    ax.set_xscale("log")
    ax.set_xlabel("Sampling Interval Δ (ms)")
    ax.set_ylabel("Estimated Lead-Lag τ* (ms)")
    ax.set_title("Lead-Lag Estimate vs Sampling Frequency")

    labels = {500: "500ms", 1000: "1s", 2000: "2s",
              5000: "5s", 10000: "10s", 30000: "30s"}
    ax.set_xticks(list(labels.keys()))
    ax.set_xticklabels(list(labels.values()), fontsize=9)

    save_fig(fig, "fig3_leadlag_recovery")


def fig4_signature_plot():
    """Realized volatility signature plot for BTC and ETH."""
    df = pd.read_csv(os.path.join(RESULTS_DIR, "signature_plot_data.csv"))

    fig, ax = plt.subplots(figsize=(6, 4))

    x = df["sampling_interval_ms"].values

    # Annualized volatility = sqrt(daily_RV * 252)
    vol_btc = np.sqrt(df["rv_btc_daily"].values * 252) * 100
    vol_eth = np.sqrt(df["rv_eth_daily"].values * 252) * 100

    ax.plot(x, vol_btc, "o-", color=COLORS[0], label="BTC", markersize=5, linewidth=1.5)
    ax.plot(x, vol_eth, "s-", color=COLORS[1], label="ETH", markersize=5, linewidth=1.5)

    ax.set_xscale("log")
    ax.set_xlabel("Sampling Interval Δ (ms)")
    ax.set_ylabel("Annualized Realized Volatility (%)")
    ax.set_title("Realized Volatility Signature Plot")
    ax.legend(fontsize=10)

    labels = {100: "100ms", 250: "250ms", 500: "500ms", 1000: "1s",
              2000: "2s", 5000: "5s", 10000: "10s", 30000: "30s", 60000: "1m"}
    ax.set_xticks(list(labels.keys()))
    ax.set_xticklabels(list(labels.values()), rotation=45, ha="right", fontsize=9)

    save_fig(fig, "fig4_signature_plot")


def main():
    os.makedirs(FIGURES_DIR, exist_ok=True)
    print("Generating figures...")
    fig1_epps_effect()
    fig2_ccf_degradation()
    fig3_leadlag_recovery()
    fig4_signature_plot()
    print("All figures generated.")


if __name__ == "__main__":
    main()
