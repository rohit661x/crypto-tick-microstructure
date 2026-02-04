"""
07_summary_tables.py
Generate comprehensive summary table and LaTeX output.
"""

import os
import pandas as pd

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")

INTERVAL_LABELS = {
    100: "100ms", 250: "250ms", 500: "500ms", 1_000: "1s",
    2_000: "2s", 5_000: "5s", 10_000: "10s", 30_000: "30s",
    60_000: "1min", 300_000: "5min",
}


def main():
    corr_df = pd.read_csv(os.path.join(RESULTS_DIR, "correlation_vs_frequency.csv"))
    peaks_df = pd.read_csv(os.path.join(RESULTS_DIR, "ccf_peaks.csv"))

    rows = []
    for _, cr in corr_df.iterrows():
        ms = int(cr["sampling_interval_ms"])
        label = INTERVAL_LABELS.get(ms, f"{ms}ms")

        # Find matching peak
        pk = peaks_df[peaks_df["sampling_interval_ms"] == ms]

        row = {
            "Sampling Δ": label,
            "Correlation": f"{cr['correlation']:.4f}",
            "95% CI": f"[{cr['ci_lower']:.4f}, {cr['ci_upper']:.4f}]",
            "N": f"{int(cr['n_observations']):,}",
        }

        if len(pk) > 0:
            p = pk.iloc[0]
            row["Lead-Lag (ms)"] = f"{int(p['peak_lag_ms']):+d}"
            row["Peak CCF"] = f"{p['peak_ccf']:.4f}"
            row["Significant"] = "Yes" if p["peak_significant"] else "No"
        else:
            row["Lead-Lag (ms)"] = "—"
            row["Peak CCF"] = "—"
            row["Significant"] = "—"

        rows.append(row)

    summary = pd.DataFrame(rows)

    # Save CSV
    summary.to_csv(os.path.join(RESULTS_DIR, "summary_table.csv"), index=False)

    # Print
    print("\nSummary Table:")
    print(summary.to_string(index=False))

    # LaTeX
    latex_lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{BTC/ETH Lead-Lag Analysis: Summary Statistics}",
        r"\label{tab:summary}",
        r"\begin{tabular}{lcccccc}",
        r"\toprule",
        r"Sampling $\Delta$ & Correlation & 95\% CI & Lead-Lag (ms) & Peak CCF & Significant & $N$ \\",
        r"\midrule",
    ]

    for _, row in summary.iterrows():
        latex_lines.append(
            f"{row['Sampling Δ']} & {row['Correlation']} & {row['95% CI']} & "
            f"{row['Lead-Lag (ms)']} & {row['Peak CCF']} & {row['Significant']} & "
            f"{row['N']} \\\\"
        )

    latex_lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])

    latex = "\n".join(latex_lines)
    with open(os.path.join(RESULTS_DIR, "summary_table.tex"), "w") as f:
        f.write(latex)

    print(f"\nSaved summary_table.csv and summary_table.tex")


if __name__ == "__main__":
    main()
