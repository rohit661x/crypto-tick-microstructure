# Bivariate Marked Hawkes Process for BTC-ETH Co-movement

## Overview

A formal continuous-time stochastic model for joint BTC/ETH price dynamics at sub-500ms timescales, using mutually-exciting point processes. Goals: parameter estimation with standard errors from tick data, and a theoretical foundation for a paper connecting to existing empirical results (HY covariance, lead-lag contrast, Granger causality, information shares).

## Model Specification

Bivariate marked Hawkes process (N^BTC, N^ETH) on [0, T].

### Intensity functions

```
λ_BTC(t) = μ_BTC + α_{BTC←BTC} ∫ φ_{BTC←BTC}(t-s) dN_BTC(s) + α_{BTC←ETH} ∫ φ_{BTC←ETH}(t-s) dN_ETH(s)

λ_ETH(t) = μ_ETH + α_{ETH←BTC} ∫ φ_{ETH←BTC}(t-s) dN_BTC(s) + α_{ETH←ETH} ∫ φ_{ETH←ETH}(t-s) dN_ETH(s)
```

Exponential kernels: `φ_{i←j}(t) = β_{i←j} · exp(-β_{i←j} · t)`, L1-normalized.

### Subscript convention

`i←j` means excitation FROM asset j TO asset i. So:
- `α_{ETH←BTC}` = BTC trade triggers ETH trades
- `α_{BTC←ETH}` = ETH trade triggers BTC trades

Code uses explicit naming: `ETH_from_BTC`, `BTC_from_ETH`.

### Parameters

- `μ_BTC, μ_ETH` — baseline intensities (events/sec)
- `α_{i←j} ∈ [0, 1)` — branching ratios. Spectral radius of branching matrix A must be < 1 for stationarity.
- `β_{i←j} > 0` — decay rates. `1/β` is the excitation timescale (how long excitation lasts), NOT a delay. The exponential kernel peaks at t=0 and decays. The ~35ms lead-lag from contrast function analysis means excitation is strongest immediately after a BTC trade with half-life ~50ms, most effect within 0-100ms.
- `σ_BTC, σ_ETH` — Laplace mark scale parameters

### Marks

Each event k carries a price change mark `m_k ~ Laplace(0, σ_i)`:
```
log f(m | σ) = -log(2σ) - |m|/σ
```

Mark likelihood separates from arrival likelihood. σ estimated as mean absolute deviation of price changes.

## Estimation

### Log-likelihood

```
ℓ(θ) = Σ_i [Σ_k log λ_i(t_k^i) - ∫₀ᵀ λ_i(t) dt] + Σ_i Σ_k log f(m_k^i | σ_i)
```

### Recursive computation (O(N))

Process events from BOTH assets in chronological order (interleaved). At each event from asset j at time t:
1. Decay all four R values: `R_{i←j} *= exp(-β_{i←j} · Δt)` for all i,j
2. Compute `λ_j(t) = μ_j + Σ_i α_{j←i} · R_{j←i}` for likelihood contribution
3. Increment: `R_{i←j} += 1` for all i

Compensator (integral term) has closed form with exponential kernels — no numerical integration needed.

### Numerical stability

- Use `expm1`/`log1p` for microsecond Δt values
- Clamp R values after large gaps to avoid underflow
- Numba JIT for the recursive likelihood (matching existing HY approach)

### Optimization

- L-BFGS-B with box constraints: `μ > 0`, `α ∈ [0, 0.99]`, `β > 0`
- 10-20 random initializations, report best likelihood
- Flag estimates hitting boundary constraints
- Standard errors from inverse observed Fisher information (Hessian of -ℓ)

### Pooling strategy

- Primary: average day-by-day estimates (robust to non-stationarity across days)
- Sensitivity check: joint likelihood across all days

## Connection to Existing Results

### Lead-lag contrast → Branching asymmetry

`α_{ETH←BTC} - α_{BTC←ETH}` captures the directional lead-lag. Decay rate `β_{ETH←BTC}` governs how quickly excitation fades — the ~35ms contrast peak corresponds to `1/β_{ETH←BTC}` on the order of 50ms.

### Granger causality → LR test on off-diagonal α

Null: `α_{ETH←BTC} = 0` (BTC does not excite ETH). LR test statistic: `2·[ℓ(θ̂) - ℓ(θ̂ | α_{ETH←BTC}=0)]`. Distribution: technically ½χ²(0) + ½χ²(1) (boundary test, β unidentified under null), but χ²(1) is conservative and standard. Single clean test replaces testing at multiple arbitrary lag windows.

### HY covariance → Simulation-based goodness-of-fit

Simulate many replications from fitted model, compute HY covariance on each, compare empirical HY to simulated distribution. Avoids analytic complexity of closed-form theoretical covariance for marked processes.

### Information shares — distinct concepts

- VECM information shares: contribution to long-run price discovery
- Hawkes branching fractions: fraction of trading activity originating from the other asset

Present both and note they capture different aspects of lead-lag.

## Implementation

### 20_hawkes_estimation.py

- Load cleaned tick data (parquet) for both assets
- Extract event times (seconds from midnight) and price-change marks per day
- Interleave events chronologically with asset labels
- Numba-JIT recursive log-likelihood (arrival + mark components)
- L-BFGS-B with box constraints, numerical stability safeguards
- 10-20 random initializations, best likelihood selected
- Day-by-day estimation + pooled (averaged) estimates
- Flag boundary hits, report branching matrix eigenvalues
- Output: `hawkes_parameter_estimates.csv` (μ, α, β, σ per day + pooled, with SEs)

### 21_hawkes_diagnostics.py

- Time-rescaling theorem: transformed inter-arrival times → unit-rate Poisson
- QQ plots and KS test on rescaled inter-arrivals
- Simulation-based HY goodness-of-fit: simulate from fitted params, compare to empirical HY
- Simulation-based CCF comparison: 100 paths, compute CCF on each, compare to empirical CCF
- Output: `hawkes_diagnostics.csv`, diagnostic figures

### 22_hawkes_inference.py

- LR tests: `α_{ETH←BTC} = 0` and `α_{BTC←ETH} = 0` (χ²(1))
- Asymmetry test: bootstrap CI on `α_{ETH←BTC} - α_{BTC←ETH}`
- Comparison table: Hawkes branching fractions alongside VECM information shares
- Decay timescale interpretation: `1/β` values with context
- Summary table: all parameter estimates, SEs, CIs, interpretation
- Output: `hawkes_inference_results.csv`
- Figure 9: Impulse response functions φ_{ETH←BTC}(t) and φ_{BTC←ETH}(t), 0-500ms
- Figure 10: Estimated intensity over 1-minute sample window with event ticks overlaid
- Figure 11: Residual QQ plot and autocorrelation of rescaled inter-arrivals
- Figure 12: Empirical CCF vs simulated 95% confidence band (100 paths)

### 23_hawkes_simulation.py

- Ogata's thinning algorithm for bivariate marked Hawkes simulation
- Generate N replications, compute summary statistics (HY cov, CCF, correlations)
- Output: `hawkes_simulations.parquet`, `simulation_summary_stats.csv`

## Output Summary Table

| Parameter | Description | Interpretation |
|-----------|-------------|----------------|
| μ_BTC, μ_ETH | Baseline intensity | events/sec without excitation |
| α_{ETH←BTC} | BTC excites ETH | fraction of ETH events triggered by BTC |
| α_{BTC←ETH} | ETH excites BTC | fraction of BTC events triggered by ETH |
| α_{BTC←BTC} | BTC self-excitation | clustering within BTC |
| α_{ETH←ETH} | ETH self-excitation | clustering within ETH |
| β_{ETH←BTC} | BTC→ETH decay rate | 1/β = excitation half-life (ms) |
| β_{BTC←ETH} | ETH→BTC decay rate | 1/β = excitation half-life (ms) |
| σ_BTC, σ_ETH | Mark scale | mean absolute price change |
| ρ(A) | Spectral radius | <1 confirms stationarity |

## References

- Bacry, Mastromatteo, Muzy (2015) — Hawkes processes in finance
- Ait-Sahalia, Cacho-Diaz, Laeven (2015) — Modeling financial contagion using mutually exciting jump processes
- Bacry & Muzy (2016) — Nonparametric kernel estimation
- Ogata (1981) — Thinning algorithm for point process simulation
