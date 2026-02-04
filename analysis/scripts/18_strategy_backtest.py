"""
18_strategy_backtest.py
Backtest a simple lead-lag strategy: trade ETH based on lagged BTC signal.
Compute gross and net Sharpe ratios.
"""

import os
import numpy as np
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "processed")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")

CONFIGS = [
    {"lookback_ms": 25, "horizon_ms": 25, "threshold_bps": 1.0},
    {"lookback_ms": 50, "horizon_ms": 50, "threshold_bps": 1.0},
    {"lookback_ms": 100, "horizon_ms": 100, "threshold_bps": 1.0},
    {"lookback_ms": 50, "horizon_ms": 50, "threshold_bps": 0.5},
    {"lookback_ms": 50, "horizon_ms": 50, "threshold_bps": 2.0},
]

COST_BPS = 2.0  # round-trip transaction cost


def load_ticks(symbol):
    name = "btc_ticks_clean.parquet" if symbol == "BTC" else "eth_ticks_clean.parquet"
    df = pd.read_parquet(os.path.join(DATA_DIR, name), columns=["timestamp", "price"])
    return df["timestamp"].values, df["price"].values


def backtest(btc_t, btc_lp, eth_t, eth_lp, lookback_ms, horizon_ms,
             threshold_bps, cost_bps):
    """Run backtest on out-of-sample (last 30%) data."""
    sampling_us = max(lookback_ms, horizon_ms) * 1000  # non-overlapping
    lookback_us = lookback_ms * 1000
    horizon_us = horizon_ms * 1000

    t_start = max(btc_t[0], eth_t[0]) + lookback_us
    t_end = min(btc_t[-1], eth_t[-1]) - horizon_us

    grid = np.arange(t_start, t_end, sampling_us)

    # Use last 30% as OOS
    n_total = len(grid)
    oos_start = int(n_total * 0.7)
    grid = grid[oos_start:]

    btc_idx_now = np.searchsorted(btc_t, grid, side="right") - 1
    btc_idx_past = np.searchsorted(btc_t, grid - lookback_us, side="right") - 1
    eth_idx_now = np.searchsorted(eth_t, grid, side="right") - 1
    eth_idx_fwd = np.searchsorted(eth_t, grid + horizon_us, side="right") - 1

    btc_idx_now = np.clip(btc_idx_now, 0, len(btc_lp) - 1)
    btc_idx_past = np.clip(btc_idx_past, 0, len(btc_lp) - 1)
    eth_idx_now = np.clip(eth_idx_now, 0, len(eth_lp) - 1)
    eth_idx_fwd = np.clip(eth_idx_fwd, 0, len(eth_lp) - 1)

    btc_signal = btc_lp[btc_idx_now] - btc_lp[btc_idx_past]
    eth_fwd_ret = eth_lp[eth_idx_fwd] - eth_lp[eth_idx_now]

    threshold = threshold_bps * 1e-4

    # Only trade when signal exceeds threshold
    mask = np.abs(btc_signal) > threshold
    signals = np.sign(btc_signal[mask])
    returns = signals * eth_fwd_ret[mask]

    if len(returns) < 100:
        return None

    avg_ret = np.mean(returns)
    std_ret = np.std(returns)

    # Trades per year (crypto = 365 days)
    duration_us = grid[-1] - grid[0]
    duration_years = duration_us / (1e6 * 86400 * 365)
    trades_per_year = len(returns) / duration_years if duration_years > 0 else 0

    gross_sharpe = (avg_ret / std_ret) * np.sqrt(trades_per_year) if std_ret > 0 else 0

    # Net of costs
    net_avg_ret = avg_ret - cost_bps * 1e-4
    net_sharpe = (net_avg_ret / std_ret) * np.sqrt(trades_per_year) if std_ret > 0 else 0

    hit_rate = np.mean(returns > 0)
    duration_days = duration_us / (1e6 * 86400)
    trades_per_day = len(returns) / duration_days if duration_days > 0 else 0

    return {
        "lookback_ms": lookback_ms,
        "horizon_ms": horizon_ms,
        "threshold_bps": threshold_bps,
        "cost_bps": cost_bps,
        "n_trades": len(returns),
        "trades_per_day": int(trades_per_day),
        "avg_return_bps": avg_ret * 1e4,
        "std_return_bps": std_ret * 1e4,
        "hit_rate": hit_rate,
        "gross_sharpe": gross_sharpe,
        "net_avg_return_bps": net_avg_ret * 1e4,
        "net_sharpe": net_sharpe,
        "oos_days": round(duration_days, 1),
    }


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    btc_t, btc_p = load_ticks("BTC")
    eth_t, eth_p = load_ticks("ETH")
    btc_lp = np.log(btc_p.astype(np.float64))
    eth_lp = np.log(eth_p.astype(np.float64))

    rows = []
    for cfg in CONFIGS:
        print(f"Backtest: lookback={cfg['lookback_ms']}ms, horizon={cfg['horizon_ms']}ms, "
              f"threshold={cfg['threshold_bps']}bps")

        result = backtest(btc_t, btc_lp, eth_t, eth_lp,
                          cfg["lookback_ms"], cfg["horizon_ms"],
                          cfg["threshold_bps"], COST_BPS)

        if result is None:
            print("  Insufficient trades")
            continue

        print(f"  Trades/day: {result['trades_per_day']:,}  "
              f"Hit: {result['hit_rate']:.1%}  "
              f"Gross SR: {result['gross_sharpe']:.2f}  "
              f"Net SR: {result['net_sharpe']:.2f}")
        rows.append(result)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS_DIR, "backtest_results.csv"), index=False)
    print(f"\nSaved backtest_results.csv")


if __name__ == "__main__":
    main()
