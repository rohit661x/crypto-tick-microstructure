# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

CSSC is a Python research project analyzing ultra-high-frequency dependence between Bitcoin and Ethereum using tick-level Binance aggTrades data. It implements microstructure-aware statistical methods: Hayashi-Yoshida covariance estimation, lead-lag detection via HRY contrast functions, Granger causality, information share decomposition, and predictive regression with strategy backtesting.

## Commands

```bash
# Data pipeline (run sequentially)
python scripts/collect_data.py        # Download 7 days of Binance aggTrades
python scripts/validate_data.py       # Clean and convert to parquet
python scripts/summary_stats.py       # Generate summary statistics

# Analysis pipeline (run from repo root)
python analysis/scripts/01_resample_data.py
python analysis/scripts/13_hayashi_yoshida.py
python analysis/scripts/14_lead_lag_contrast.py
python analysis/scripts/15_granger_causality.py
python analysis/scripts/16_information_share.py
python analysis/scripts/17_predictive_regression.py
python analysis/scripts/18_strategy_backtest.py
python analysis/scripts/19_generate_report.py

# Hawkes process pipeline (run after existing pipeline)
python analysis/scripts/20_hawkes_estimation.py
python analysis/scripts/21_hawkes_diagnostics.py
python analysis/scripts/22_hawkes_inference.py
python analysis/scripts/23_hawkes_simulation.py
```

No test framework, linting, or build system is configured. Dependencies: `pip install pandas pyarrow requests` (scripts also use numpy, scipy, statsmodels, numba, matplotlib).

## Architecture

**Data flow:** Raw CSV (Binance) → cleaned parquet → resampled parquet (10 frequencies: 100ms–5min) → analysis CSVs + figures.

**`scripts/`** — Data collection and preparation. `collect_data.py` downloads from Binance data archive, `validate_data.py` cleans and outputs parquet.

**`analysis/scripts/`** — Numbered analysis pipeline. Scripts 01–07 handle resampling, correlation, CCF, noise estimation, and optimal frequency selection. Scripts 13–19 implement the core econometric methods (HY covariance, lead-lag contrast, Granger causality, VECM information shares, predictive regression, backtesting, and report generation). Scripts 20–23 implement bivariate marked Hawkes process estimation, diagnostics, inference, and simulation (see `docs/plans/2026-02-02-hawkes-process-design.md`).

**`data/raw/`** — Raw CSV tick data (~450MB each). **`data/processed/`** — Cleaned parquet (~110MB each).

**`analysis/results/`** — Output CSVs (covariance estimates, lead-lag, Granger, information shares, backtest results, Hawkes parameter estimates). **`analysis/figures/`** — PNG/PDF figures (fig1–fig12).

**`docs/plans/`** — Design documents for new analysis modules.

## Key Implementation Details

- Numba JIT compilation is used in `13_hayashi_yoshida.py` for O(N) HY estimator performance
- `np.searchsorted()` for efficient timestamp alignment across tick streams
- Day-by-day processing to manage memory with multi-million tick datasets
- Last-tick forward-fill interpolation for resampling; log returns for stationarity
- Bootstrap inference uses day-level block resampling (200 iterations)
- VECM information shares use Johansen cointegration test + Cholesky decomposition

### Hawkes process module (scripts 20–23)

- Bivariate marked Hawkes process with exponential kernels and Laplace price-change marks
- Subscript convention: `i←j` means excitation FROM asset j TO asset i. Code uses `ETH_from_BTC`, `BTC_from_ETH` naming.
- Decay rate interpretation: `1/β` is the excitation timescale (how long excitation lasts), NOT a delay. The kernel peaks at t=0.
- O(N) recursive likelihood via interleaved chronological processing of both assets' events
- Numerical stability: expm1/log1p for microsecond Δt, R-value clamping for large gaps
- MLE with L-BFGS-B, 10-20 random starts, day-by-day estimation averaged for pooled estimates
- Ogata's thinning algorithm for simulation; simulation-based goodness-of-fit against empirical HY and CCF
