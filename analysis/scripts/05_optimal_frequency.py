"""
05_optimal_frequency.py
Determine optimal sampling frequency balancing bias and resolution.
"""

import os
import pandas as pd

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")


def main():
    corr_df = pd.read_csv(os.path.join(RESULTS_DIR, "correlation_vs_frequency.csv"))
    peaks_df = pd.read_csv(os.path.join(RESULTS_DIR, "ccf_peaks.csv"))

    lines = []
    lines.append("Optimal Sampling Frequency Analysis")
    lines.append("=" * 50)

    # 1. Find Δ where correlation reaches 90% of asymptotic value
    max_corr = corr_df["correlation"].max()
    threshold_90 = 0.90 * max_corr
    stable = corr_df[corr_df["correlation"] >= threshold_90]
    if len(stable) > 0:
        delta_stable = int(stable["sampling_interval_ms"].min())
        lines.append(f"\nCorrelation stabilizes (90% of max={max_corr:.4f}) at: Δ = {delta_stable}ms")
    else:
        delta_stable = int(corr_df["sampling_interval_ms"].max())
        lines.append(f"\nCorrelation does not reach 90% of max; using largest Δ = {delta_stable}ms")

    # 2. Find smallest Δ where lead-lag peak is significant
    sig_peaks = peaks_df[peaks_df["peak_significant"] == True]
    if len(sig_peaks) > 0:
        delta_sig = int(sig_peaks["sampling_interval_ms"].min())
        lines.append(f"Lead-lag significant at: Δ ≥ {delta_sig}ms")
    else:
        delta_sig = None
        lines.append("Lead-lag not significant at any tested frequency")

    # 3. Optimal: smallest where both conditions met
    delta_opt = max(delta_stable, delta_sig) if delta_sig else delta_stable
    lines.append(f"\nRecommended optimal Δ*: {delta_opt}ms")

    # Report stats at optimal
    opt_corr = corr_df[corr_df["sampling_interval_ms"] == delta_opt]
    opt_peak = peaks_df[peaks_df["sampling_interval_ms"] == delta_opt]

    if len(opt_corr) > 0:
        row = opt_corr.iloc[0]
        lines.append(f"\nAt optimal Δ = {delta_opt}ms:")
        lines.append(f"  Correlation:  {row['correlation']:.4f} "
                      f"(95% CI: [{row['ci_lower']:.4f}, {row['ci_upper']:.4f}])")

    if len(opt_peak) > 0:
        row = opt_peak.iloc[0]
        lines.append(f"  Lead-lag:     {row['peak_lag']:+d} lags = {row['peak_lag_ms']:+d}ms")
        lines.append(f"  Peak CCF:     {row['peak_ccf']:.4f}")
        lines.append(f"  Significant:  {row['peak_significant']}")

    # Also show what happens at nearby frequencies
    lines.append("\n\nCorrelation recovery curve:")
    for _, row in corr_df.iterrows():
        pct = row["correlation"] / max_corr * 100 if max_corr > 0 else 0
        lines.append(f"  Δ={int(row['sampling_interval_ms']):>6}ms  "
                      f"ρ={row['correlation']:.4f}  ({pct:.0f}% of max)")

    report = "\n".join(lines)
    print(report)

    with open(os.path.join(RESULTS_DIR, "optimal_frequency.txt"), "w") as f:
        f.write(report + "\n")

    print(f"\nSaved optimal_frequency.txt")


if __name__ == "__main__":
    main()
