"""
04_noise_estimation.py
Realized variance signature plot, noise estimation, and cross-asset noise independence test.
"""

import os
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy import stats

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")

INTERVALS_MS = [100, 250, 500, 1_000, 2_000, 5_000, 10_000, 30_000, 60_000]
SECONDS_PER_DAY = 86400


def noise_model(delta_sec, sigma_sq_true, sigma_sq_noise):
    """RV(Δ) = σ²_true + 2σ²_noise / Δ"""
    return sigma_sq_true + 2 * sigma_sq_noise / delta_sec


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # ── Part A: Signature plot ──
    sig_rows = []

    for interval_ms in INTERVALS_MS:
        path = os.path.join(RESULTS_DIR, f"resampled_{interval_ms}ms.parquet")
        if not os.path.exists(path):
            continue

        df = pd.read_parquet(path)
        btc_r = df["btc_return"].dropna().values
        eth_r = df["eth_return"].dropna().values

        delta_sec = interval_ms / 1000.0

        # Realized variance (sum of squared returns)
        rv_btc = np.sum(btc_r ** 2)
        rv_eth = np.sum(eth_r ** 2)

        # Number of intervals per day
        n_per_day = SECONDS_PER_DAY / delta_sec

        # Duration in days
        n_obs = len(btc_r)
        duration_days = n_obs / n_per_day

        # Normalize to daily
        rv_btc_daily = rv_btc / duration_days
        rv_eth_daily = rv_eth / duration_days

        # Realized covariance (daily)
        mask = np.isfinite(btc_r) & np.isfinite(eth_r)
        rcov = np.sum(btc_r[mask] * eth_r[mask]) / duration_days

        sig_rows.append({
            "sampling_interval_ms": interval_ms,
            "delta_sec": delta_sec,
            "rv_btc_daily": rv_btc_daily,
            "rv_eth_daily": rv_eth_daily,
            "rcov_daily": rcov,
            "n_observations": n_obs,
            "duration_days": round(duration_days, 2),
        })

        print(f"Δ={interval_ms:>5}ms  RV_btc={rv_btc_daily:.6f}  RV_eth={rv_eth_daily:.6f}  RCov={rcov:.6f}")

    sig_df = pd.DataFrame(sig_rows)
    sig_df.to_csv(os.path.join(RESULTS_DIR, "signature_plot_data.csv"), index=False)

    # ── Fit noise model: RV(Δ) = σ²_true + 2σ²_noise/Δ ──
    noise_rows = []
    for asset, col in [("BTC", "rv_btc_daily"), ("ETH", "rv_eth_daily")]:
        x = sig_df["delta_sec"].values
        y = sig_df[col].values

        try:
            popt, pcov = curve_fit(noise_model, x, y, p0=[y[-1], 1e-6], maxfev=10000)
            sigma_sq_true, sigma_sq_noise = popt
            # Ensure non-negative
            sigma_sq_noise = max(sigma_sq_noise, 0)
            ratio = sigma_sq_noise / sigma_sq_true if sigma_sq_true > 0 else float("inf")
            print(f"\n{asset}: σ²_true={sigma_sq_true:.8f}  σ²_noise={sigma_sq_noise:.8f}  "
                  f"noise/signal={ratio:.4f}")
        except Exception as e:
            print(f"\n{asset}: Curve fit failed: {e}")
            sigma_sq_true = y[-1]
            sigma_sq_noise = 0
            ratio = 0

        noise_rows.append({
            "asset": asset,
            "sigma_sq_true": sigma_sq_true,
            "sigma_sq_noise": sigma_sq_noise,
            "noise_to_signal_ratio": ratio,
        })

    noise_df = pd.DataFrame(noise_rows)
    noise_df.to_csv(os.path.join(RESULTS_DIR, "noise_estimates.csv"), index=False)

    # ── Part B: Test cross-asset noise independence ──
    print("\n" + "=" * 60)
    print("Cross-Asset Noise Independence Test")
    print("=" * 60)

    # Compare covariance at fine vs coarse frequency
    cov_fine = sig_df.loc[sig_df["sampling_interval_ms"] == 100, "rcov_daily"].values
    cov_coarse = sig_df.loc[sig_df["sampling_interval_ms"] == 60_000, "rcov_daily"].values

    lines = []
    if len(cov_fine) > 0 and len(cov_coarse) > 0:
        cov_fine_val = cov_fine[0]
        cov_coarse_val = cov_coarse[0]
        ratio = cov_fine_val / cov_coarse_val if cov_coarse_val != 0 else float("inf")
        lines.append(f"Covariance at 100ms (daily): {cov_fine_val:.8f}")
        lines.append(f"Covariance at 1min (daily):  {cov_coarse_val:.8f}")
        lines.append(f"Ratio (fine/coarse):         {ratio:.4f}")
        if ratio < 0.8:
            lines.append("Interpretation: Significant attenuation — noise effects dominate at fine frequencies")
        elif ratio > 1.2:
            lines.append("Interpretation: Positive noise correlation (correlated microstructure noise)")
        else:
            lines.append("Interpretation: Covariance roughly stable — noise approximately orthogonal")
    else:
        lines.append("Could not compare (missing data at 100ms or 1min)")

    # Residual correlation test at 100ms
    path_fine = os.path.join(RESULTS_DIR, "resampled_100ms.parquet")
    if os.path.exists(path_fine):
        df_fine = pd.read_parquet(path_fine)
        btc_r = df_fine["btc_return"].dropna().values
        eth_r = df_fine["eth_return"].dropna().values
        mask = np.isfinite(btc_r) & np.isfinite(eth_r)
        btc_r, eth_r = btc_r[mask], eth_r[mask]

        # Demean
        btc_dm = btc_r - btc_r.mean()
        eth_dm = eth_r - eth_r.mean()

        # OLS: eth_dm = β * btc_dm + ε
        slope, intercept, r_value, p_value, std_err = stats.linregress(btc_dm, eth_dm)
        t_stat = slope / std_err if std_err > 0 else float("inf")

        lines.append("")
        lines.append("Residual correlation test (100ms demeaned returns):")
        lines.append(f"  Coefficient (β): {slope:.6f}")
        lines.append(f"  Standard error:  {std_err:.6f}")
        lines.append(f"  t-statistic:     {t_stat:.2f}")
        lines.append(f"  p-value:         {p_value:.2e}")
        if p_value < 0.05:
            lines.append("  Conclusion: Reject independence — microstructure noise is correlated across assets")
        else:
            lines.append("  Conclusion: Cannot reject independence")

    report = "\n".join(lines)
    print(report)

    with open(os.path.join(RESULTS_DIR, "orthogonality_test.txt"), "w") as f:
        f.write("Cross-Asset Noise Independence Test\n")
        f.write("=" * 50 + "\n\n")
        f.write(report + "\n")

    print(f"\nSaved signature_plot_data.csv, noise_estimates.csv, orthogonality_test.txt")


if __name__ == "__main__":
    main()
