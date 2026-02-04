"""
16_information_share.py
Hasbrouck information share via VECM on synchronized log prices.
Tests at multiple sampling frequencies.
"""

import os
import warnings
import numpy as np
import pandas as pd
from statsmodels.tsa.vector_ar.vecm import coint_johansen, VECM

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "processed")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")

INTERVALS_MS = [5_000, 10_000, 30_000]


def load_resampled(interval_ms):
    path = os.path.join(RESULTS_DIR, f"resampled_{interval_ms}ms.parquet")
    return pd.read_parquet(path)


def compute_information_share(btc_log_p, eth_log_p, k_ar_diff=5):
    """Compute Hasbrouck information shares from VECM."""
    data = pd.DataFrame({"btc": btc_log_p, "eth": eth_log_p})

    # Johansen cointegration test
    joh = coint_johansen(data, det_order=0, k_ar_diff=k_ar_diff)
    trace_stat = joh.lr1[0]
    crit_5 = joh.cvt[0, 1]
    cointegrated = trace_stat > crit_5

    # Fit VECM with rank 1
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        vecm = VECM(data, k_ar_diff=k_ar_diff, coint_rank=1, deterministic="co")
        vecm_fit = vecm.fit()

    alpha = vecm_fit.alpha  # (2, 1) adjustment coefficients
    beta = vecm_fit.beta    # (2, 1) cointegrating vector (+ det terms)
    Omega = vecm_fit.sigma_u  # (2, 2) residual covariance

    # Compute long-run impact matrix Psi(1) = C(1)
    # For VECM: C(1) = beta_perp @ (alpha_perp' @ Gamma @ beta_perp)^{-1} @ alpha_perp'
    # where Gamma = I - sum(Gamma_k)
    # Simplified: use the VMA(infinity) representation

    # Alpha_perp: orthogonal complement of alpha
    # For 2x1 alpha = [[a1],[a2]], alpha_perp = [[a2],[-a1]] (up to scale)
    a1, a2 = alpha[0, 0], alpha[1, 0]
    alpha_perp = np.array([[a2], [-a1]])

    # Beta_perp similarly
    b = beta[:2, 0]  # first 2 elements (exclude deterministic)
    beta_perp = np.array([[b[1]], [-b[0]]])

    # Gamma = I - sum of short-run matrices
    # vecm_fit.gamma is (2, 2*k_ar_diff) — each (2,2) block is one lag
    gamma_raw = vecm_fit.gamma  # shape (2, 2*k)
    n_lags = gamma_raw.shape[1] // 2
    gamma_sum = np.zeros((2, 2))
    for k in range(n_lags):
        gamma_sum += gamma_raw[:, 2 * k: 2 * (k + 1)]
    Gamma_mat = np.eye(2) - gamma_sum

    # Long-run impact row vector psi
    inner = alpha_perp.T @ Gamma_mat @ beta_perp
    if abs(inner[0, 0]) < 1e-12:
        return None  # degenerate
    psi = (1.0 / inner[0, 0]) * alpha_perp.T  # (1, 2)

    # Information shares via Cholesky
    # Order 1: BTC first
    try:
        F1 = np.linalg.cholesky(Omega)
    except np.linalg.LinAlgError:
        return None
    gamma1 = psi @ F1  # (1, 2)
    total1 = np.sum(gamma1 ** 2)
    is_btc_upper = gamma1[0, 0] ** 2 / total1

    # Order 2: ETH first (permute)
    P = np.array([[0, 1], [1, 0]], dtype=float)
    Omega_p = P @ Omega @ P.T
    try:
        F2 = np.linalg.cholesky(Omega_p)
    except np.linalg.LinAlgError:
        return None
    F2_orig = P.T @ F2 @ P
    gamma2 = psi @ F2_orig
    total2 = np.sum(gamma2 ** 2)
    is_btc_lower = gamma2[0, 0] ** 2 / total2

    is_btc = (is_btc_lower + is_btc_upper) / 2.0
    is_eth = 1.0 - is_btc

    return {
        "is_btc": is_btc,
        "is_eth": is_eth,
        "is_btc_lower": min(is_btc_lower, is_btc_upper),
        "is_btc_upper": max(is_btc_lower, is_btc_upper),
        "cointegrated": cointegrated,
        "johansen_trace": trace_stat,
        "johansen_crit_5pct": crit_5,
        "alpha_btc": a1,
        "alpha_eth": a2,
    }


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    rows = []

    for interval_ms in INTERVALS_MS:
        print(f"\nSampling Δ = {interval_ms}ms")
        try:
            df = load_resampled(interval_ms)
        except FileNotFoundError:
            print(f"  Resampled file not found, skipping")
            continue

        btc_lp = np.log(df["btc_price"].values)
        eth_lp = np.log(df["eth_price"].values)

        # Limit to 50K observations for VECM tractability
        max_obs = 10_000
        if len(btc_lp) > max_obs:
            btc_lp = btc_lp[:max_obs]
            eth_lp = eth_lp[:max_obs]
            print(f"  Truncated to {max_obs:,} observations")

        result = compute_information_share(btc_lp, eth_lp, k_ar_diff=5)
        if result is None:
            print("  Computation failed (degenerate)")
            continue

        result["sampling_interval_ms"] = interval_ms
        rows.append(result)

        coint_str = "Yes" if result["cointegrated"] else "No"
        print(f"  Cointegrated: {coint_str} (trace={result['johansen_trace']:.1f}, crit={result['johansen_crit_5pct']:.1f})")
        print(f"  IS(BTC): {result['is_btc']:.1%} [{result['is_btc_lower']:.1%}, {result['is_btc_upper']:.1%}]")
        print(f"  IS(ETH): {result['is_eth']:.1%}")

    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(RESULTS_DIR, "information_shares.csv"), index=False)
    print(f"\nSaved information_shares.csv")


if __name__ == "__main__":
    main()
