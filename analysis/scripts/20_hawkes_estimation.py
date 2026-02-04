"""
20_hawkes_estimation.py
Bivariate marked Hawkes process estimation for BTC-ETH tick data.
Exponential kernels, Laplace marks, O(N) recursive likelihood with numba.
MLE via L-BFGS-B with multiple random starts, day-by-day + pooled estimates.
"""

import os
import numpy as np
import pandas as pd
from numba import njit
from scipy.optimize import minimize

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "processed")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")

N_STARTS = 15  # random initializations
# Parameter order: [mu_BTC, mu_ETH, a_BB, a_BE, a_EB, a_EE, b_BB, b_BE, b_EB, b_EE]
# a_BE = alpha_{BTC←ETH}, a_EB = alpha_{ETH←BTC}
# b_BE = beta_{BTC←ETH},  b_EB = beta_{ETH←BTC}
N_PARAMS = 10
PARAM_NAMES = [
    "mu_BTC", "mu_ETH",
    "alpha_BTC_from_BTC", "alpha_BTC_from_ETH",
    "alpha_ETH_from_BTC", "alpha_ETH_from_ETH",
    "beta_BTC_from_BTC", "beta_BTC_from_ETH",
    "beta_ETH_from_BTC", "beta_ETH_from_ETH",
]

# Box constraints
BOUNDS = [
    (1e-3, None),   # mu_BTC
    (1e-3, None),   # mu_ETH
    (1e-6, 0.99),   # a_BB
    (1e-6, 0.99),   # a_BE
    (1e-6, 0.99),   # a_EB
    (1e-6, 0.99),   # a_EE
    (0.1, 1000.0),  # b_BB
    (0.1, 1000.0),  # b_BE
    (0.1, 1000.0),  # b_EB
    (0.1, 1000.0),  # b_EE
]


def load_ticks(symbol):
    name = "btc_ticks_clean.parquet" if symbol == "BTC" else "eth_ticks_clean.parquet"
    df = pd.read_parquet(os.path.join(DATA_DIR, name), columns=["timestamp", "price"])
    return df["timestamp"].values, df["price"].values


def prepare_day(btc_t, btc_p, eth_t, eth_p, day_start_us, day_end_us):
    """Extract one day, interleave events, compute marks (log price changes)."""
    bm = (btc_t >= day_start_us) & (btc_t < day_end_us)
    em = (eth_t >= day_start_us) & (eth_t < day_end_us)
    bt, bp = btc_t[bm], btc_p[bm]
    et, ep = eth_t[em], eth_p[em]

    if len(bt) < 100 or len(et) < 100:
        return None

    # Convert to seconds from day start
    bt_sec = (bt - day_start_us) / 1e6
    et_sec = (et - day_start_us) / 1e6

    # Log price changes (marks)
    bt_marks = np.diff(np.log(bp.astype(np.float64)))
    et_marks = np.diff(np.log(ep.astype(np.float64)))
    bt_sec_m = bt_sec[1:]  # align with marks
    et_sec_m = et_sec[1:]

    # Interleave: asset=0 for BTC, asset=1 for ETH
    n_b, n_e = len(bt_sec_m), len(et_sec_m)
    times = np.concatenate([bt_sec_m, et_sec_m])
    marks = np.concatenate([bt_marks, et_marks])
    assets = np.concatenate([np.zeros(n_b, dtype=np.int32), np.ones(n_e, dtype=np.int32)])

    order = np.argsort(times, kind="mergesort")
    times = times[order]
    marks = marks[order]
    assets = assets[order]

    T = (day_end_us - day_start_us) / 1e6  # total window in seconds

    return times, marks, assets, T


@njit
def neg_log_likelihood(params, times, marks, assets, T):
    """
    Negative log-likelihood for bivariate marked Hawkes process.
    O(N) recursive computation with exponential kernels.
    """
    mu_B, mu_E = params[0], params[1]
    a_BB, a_BE, a_EB, a_EE = params[2], params[3], params[4], params[5]
    b_BB, b_BE, b_EB, b_EE = params[6], params[7], params[8], params[9]

    n = len(times)

    # Recursive R values: R_{i←j} tracks decayed excitation from j to i
    R_BB = 0.0  # BTC from BTC
    R_BE = 0.0  # BTC from ETH
    R_EB = 0.0  # ETH from BTC
    R_EE = 0.0  # ETH from ETH

    log_lik = 0.0
    prev_t = 0.0

    # Mark parameters: σ estimated separately (Laplace scale = mean |mark|)
    # Here we only compute arrival likelihood; marks handled outside.

    for k in range(n):
        t = times[k]
        a = assets[k]
        dt = t - prev_t

        # Decay all R values
        if dt > 0:
            R_BB *= np.exp(-b_BB * dt)
            R_BE *= np.exp(-b_BE * dt)
            R_EB *= np.exp(-b_EB * dt)
            R_EE *= np.exp(-b_EE * dt)

        # Compute intensity for this event's asset
        if a == 0:  # BTC event
            lam = mu_B + a_BB * R_BB + a_BE * R_BE
        else:  # ETH event
            lam = mu_E + a_EB * R_EB + a_EE * R_EE

        if lam <= 0:
            return 1e15  # invalid

        log_lik += np.log(lam)

        # Update R: this event from asset a excites both assets
        if a == 0:  # BTC event
            R_BB += b_BB  # β * 1 (kernel is β·exp(-β·t), adding β so α·R gives α·β·exp sum)
            R_EB += b_EB
        else:  # ETH event
            R_BE += b_BE
            R_EE += b_EE

        prev_t = t

    # Compensator: ∫₀ᵀ λ_i(t) dt for each asset
    # For exponential kernel: ∫₀ᵀ α·β·exp(-β(t-s)) dt from s to T = α·(1 - exp(-β(T-s)))
    # Summing over all events: Σ_k α·(1 - exp(-β(T - t_k)))

    comp_B = mu_B * T
    comp_E = mu_E * T

    for k in range(n):
        t = times[k]
        a = assets[k]
        dt_to_end = T - t

        if a == 0:  # BTC event excites both
            comp_B += a_BB * (1.0 - np.exp(-b_BB * dt_to_end))
            comp_E += a_EB * (1.0 - np.exp(-b_EB * dt_to_end))
        else:  # ETH event excites both
            comp_B += a_BE * (1.0 - np.exp(-b_BE * dt_to_end))
            comp_E += a_EE * (1.0 - np.exp(-b_EE * dt_to_end))

    log_lik -= (comp_B + comp_E)

    return -log_lik


def random_init():
    """Generate random initial parameter vector."""
    mu_B = np.random.uniform(5, 50)
    mu_E = np.random.uniform(5, 50)
    a_BB = np.random.uniform(0.01, 0.5)
    a_BE = np.random.uniform(0.01, 0.3)
    a_EB = np.random.uniform(0.01, 0.3)
    a_EE = np.random.uniform(0.01, 0.5)
    b_BB = np.random.uniform(1, 100)
    b_BE = np.random.uniform(1, 100)
    b_EB = np.random.uniform(1, 100)
    b_EE = np.random.uniform(1, 100)
    return np.array([mu_B, mu_E, a_BB, a_BE, a_EB, a_EE, b_BB, b_BE, b_EB, b_EE])


def fit_one_day(times, marks, assets, T, n_starts=N_STARTS):
    """Fit Hawkes model to one day with multiple random starts."""
    best_nll = np.inf
    best_result = None

    for s in range(n_starts):
        x0 = random_init()
        try:
            res = minimize(
                neg_log_likelihood,
                x0,
                args=(times, marks, assets, T),
                method="L-BFGS-B",
                bounds=BOUNDS,
                options={"maxiter": 500, "ftol": 1e-10},
            )
            if res.fun < best_nll:
                best_nll = res.fun
                best_result = res
        except Exception:
            continue

    if best_result is None:
        return None

    params = best_result.x

    # Standard errors from Hessian
    se = np.full(N_PARAMS, np.nan)
    try:
        hess_inv = best_result.hess_inv
        if hasattr(hess_inv, "todense"):
            hess_inv = hess_inv.todense()
        hess_diag = np.diag(np.array(hess_inv))
        se = np.sqrt(np.maximum(hess_diag, 0))
    except Exception:
        pass

    # Mark scale parameters (Laplace): σ = mean|m|
    btc_mask = assets == 0
    eth_mask = assets == 1
    sigma_BTC = np.mean(np.abs(marks[btc_mask])) if np.sum(btc_mask) > 0 else np.nan
    sigma_ETH = np.mean(np.abs(marks[eth_mask])) if np.sum(eth_mask) > 0 else np.nan

    # Branching matrix spectral radius
    A = np.array([[params[2], params[3]], [params[4], params[5]]])
    spectral_radius = np.max(np.abs(np.linalg.eigvals(A)))

    # Check boundary hits
    boundary_hits = []
    for i, (lo, hi) in enumerate(BOUNDS):
        if hi is not None and abs(params[i] - hi) < 1e-4:
            boundary_hits.append(PARAM_NAMES[i])
        if lo is not None and abs(params[i] - lo) < 1e-4:
            boundary_hits.append(PARAM_NAMES[i])

    return {
        "params": params,
        "se": se,
        "nll": best_nll,
        "n_events": len(times),
        "T": T,
        "sigma_BTC": sigma_BTC,
        "sigma_ETH": sigma_ETH,
        "spectral_radius": spectral_radius,
        "boundary_hits": ",".join(boundary_hits) if boundary_hits else "",
        "converged": best_result.success,
    }


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    btc_t, btc_p = load_ticks("BTC")
    eth_t, eth_p = load_ticks("ETH")

    t_start = max(btc_t[0], eth_t[0])
    t_end = min(btc_t[-1], eth_t[-1])

    day_us = int(86400 * 1e6)
    day_starts = np.arange(t_start, t_end, day_us)

    rows = []

    for d_idx, d_start in enumerate(day_starts):
        d_end = d_start + day_us
        print(f"\nDay {d_idx + 1}/{len(day_starts)}")

        data = prepare_day(btc_t, btc_p, eth_t, eth_p, d_start, d_end)
        if data is None:
            print("  Insufficient data, skipping")
            continue

        times, marks, assets, T = data
        n_btc = np.sum(assets == 0)
        n_eth = np.sum(assets == 1)
        print(f"  Events: {len(times):,} (BTC={n_btc:,}, ETH={n_eth:,}), T={T:.0f}s")

        # Subsample for tractability if needed (keep ~200K events)
        max_events = 200_000
        if len(times) > max_events:
            step = len(times) // max_events
            times = times[::step]
            marks = marks[::step]
            assets = assets[::step]
            print(f"  Subsampled to {len(times):,} events (step={step})")

        result = fit_one_day(times, marks, assets, T, n_starts=N_STARTS)
        if result is None:
            print("  Optimization failed")
            continue

        row = {"day": d_idx + 1}
        for i, name in enumerate(PARAM_NAMES):
            row[name] = result["params"][i]
            row[f"{name}_se"] = result["se"][i]
        row["sigma_BTC"] = result["sigma_BTC"]
        row["sigma_ETH"] = result["sigma_ETH"]
        row["spectral_radius"] = result["spectral_radius"]
        row["neg_loglik"] = result["nll"]
        row["n_events"] = result["n_events"]
        row["boundary_hits"] = result["boundary_hits"]
        row["converged"] = result["converged"]

        rows.append(row)

        p = result["params"]
        print(f"  μ_BTC={p[0]:.2f}  μ_ETH={p[1]:.2f}")
        print(f"  α_BB={p[2]:.4f}  α_BE={p[3]:.4f}  α_EB={p[4]:.4f}  α_EE={p[5]:.4f}")
        print(f"  β_BB={p[6]:.2f}  β_BE={p[7]:.2f}  β_EB={p[8]:.2f}  β_EE={p[9]:.2f}")
        print(f"  1/β_EB={1000/p[8]:.1f}ms  1/β_BE={1000/p[7]:.1f}ms")
        print(f"  σ_BTC={result['sigma_BTC']:.6f}  σ_ETH={result['sigma_ETH']:.6f}")
        print(f"  ρ(A)={result['spectral_radius']:.4f}  NLL={result['nll']:.1f}")
        if result["boundary_hits"]:
            print(f"  ⚠ Boundary hits: {result['boundary_hits']}")

    if not rows:
        print("No days estimated successfully")
        return

    df = pd.DataFrame(rows)

    # Pooled estimates: average across days
    pooled = {"day": "pooled"}
    for name in PARAM_NAMES:
        vals = df[name].values
        pooled[name] = np.mean(vals)
        pooled[f"{name}_se"] = np.std(vals, ddof=1) / np.sqrt(len(vals))
    pooled["sigma_BTC"] = df["sigma_BTC"].mean()
    pooled["sigma_ETH"] = df["sigma_ETH"].mean()
    A_pooled = np.array([
        [pooled["alpha_BTC_from_BTC"], pooled["alpha_BTC_from_ETH"]],
        [pooled["alpha_ETH_from_BTC"], pooled["alpha_ETH_from_ETH"]],
    ])
    pooled["spectral_radius"] = np.max(np.abs(np.linalg.eigvals(A_pooled)))
    pooled["neg_loglik"] = df["neg_loglik"].sum()
    pooled["n_events"] = df["n_events"].sum()
    pooled["boundary_hits"] = ""
    pooled["converged"] = True

    df = pd.concat([df, pd.DataFrame([pooled])], ignore_index=True)
    df.to_csv(os.path.join(RESULTS_DIR, "hawkes_parameter_estimates.csv"), index=False)

    print(f"\n{'='*60}")
    print("POOLED ESTIMATES")
    print(f"{'='*60}")
    for name in PARAM_NAMES:
        print(f"  {name:30s} = {pooled[name]:.4f} (SE = {pooled[f'{name}_se']:.4f})")
    print(f"  {'sigma_BTC':30s} = {pooled['sigma_BTC']:.6f}")
    print(f"  {'sigma_ETH':30s} = {pooled['sigma_ETH']:.6f}")
    print(f"  {'spectral_radius':30s} = {pooled['spectral_radius']:.4f}")
    print(f"  α_EB - α_BE (asymmetry)      = {pooled['alpha_ETH_from_BTC'] - pooled['alpha_BTC_from_ETH']:.4f}")
    print(f"  1/β_EB (BTC→ETH timescale)   = {1000/pooled['beta_ETH_from_BTC']:.1f}ms")
    print(f"  1/β_BE (ETH→BTC timescale)   = {1000/pooled['beta_BTC_from_ETH']:.1f}ms")

    print(f"\nSaved hawkes_parameter_estimates.csv")


if __name__ == "__main__":
    main()
