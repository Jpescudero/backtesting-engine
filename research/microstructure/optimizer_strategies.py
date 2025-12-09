"""
Unified Strategy Optimizer
-------------------------
Runs parameter gridsearch for:
- V1 (forward returns)
- V2 (microstructure + forward returns)
- V3 (microstructure + SL/TP simulation)

Supports multiprocessing and heatmap generation.
"""

from __future__ import annotations
import sys
import time
import itertools
import argparse
from pathlib import Path
import multiprocessing as mp

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import research.microstructure.study_opening_sweeps as V1
import research.microstructure.study_opening_sweeps_v2 as V2
import research.microstructure.study_opening_sweeps_v3 as V3

from src.config.paths import REPORTS_DIR


# =============================================================
# EVALUATION
# =============================================================

def evaluate_trade_results(trades_df: pd.DataFrame):
    """Compute performance metrics from a trade dataframe."""

    if len(trades_df) < 20:
        return {
            "sharpe": -999,
            "mean": 0,
            "winrate": 0,
            "n_trades": len(trades_df)
        }

    if "r_multiple" in trades_df:
        r = trades_df["r_multiple"]
    else:
        r = trades_df["fwd_return"]

    sharpe = r.mean() / (r.std() + 1e-9)
    return {
        "sharpe": sharpe,
        "mean": r.mean(),
        "winrate": (r > 0).mean(),
        "n_trades": len(trades_df)
    }


# =============================================================
# PARAMETER GRIDS
# =============================================================

def generate_param_grid(strategy):
    if strategy in ("v1", "v2"):
        grid = {
            "wick_factor": [1.2, 1.5, 1.8, 2.0],
            "atr_percentile": [0.2, 0.4, 0.6],
            "volume_percentile": [0.4, 0.6, 0.8],
        }

    elif strategy == "v3":
        grid = {
            "wick_factor": [1.5, 1.8, 2.0],
            "atr_percentile": [0.4, 0.5, 0.6],
            "volume_percentile": [0.4, 0.6, 0.8],
            "sl_buffer_atr": [0.2, 0.3, 0.4],
            "sl_buffer_relative": [0.0, 0.1, 0.2],
            "tp_multiplier": [1.0, 1.2, 1.5],
        }

    else:
        raise ValueError("strategy must be v1, v2 or v3")

    keys = list(grid.keys())
    vals = list(grid.values())
    combos = [dict(zip(keys, v)) for v in itertools.product(*vals)]
    return combos


# =============================================================
# WORKER
# =============================================================

def worker_eval(args):
    strategy, df, df_ohlcv, params = args

    if strategy == "v1":
        trades = V1.run_with_params(df, params)
    elif strategy == "v2":
        trades = V2.run_with_params(df, params)
    elif strategy == "v3":
        trades = V3.run_with_params(df, df_ohlcv, params)
    else:
        raise ValueError("Invalid strategy")

    metrics = evaluate_trade_results(trades)
    metrics.update(params)
    return metrics


# =============================================================
# HEATMAPS
# =============================================================

def generate_heatmap(dfres, x_param, y_param, value_param, outpath):
    pivot = dfres.pivot_table(index=y_param, columns=x_param, values=value_param)
    plt.figure(figsize=(8,6))
    plt.title(f"{value_param} by {x_param} / {y_param}")
    plt.imshow(pivot, cmap="viridis", aspect="auto")
    plt.colorbar(label=value_param)
    plt.xticks(range(len(pivot.columns)), pivot.columns)
    plt.yticks(range(len(pivot.index)), pivot.index)
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()


# =============================================================
# MAIN OPTIMIZER
# =============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", required=True, choices=["v1","v2","v3"])
    parser.add_argument("--n-jobs", type=int, default=mp.cpu_count()-1)
    args = parser.parse_args()

    strategy = args.strategy
    n_jobs = args.n_jobs

    print(f"\n🚀 OPTIMIZER — Strategy {strategy}")
    print(f"Using {n_jobs} workers\n")

    # preload data ONCE
    if strategy == "v1":
        df = V1.preload_data()
        df_ohlcv = None

    elif strategy == "v2":
        df = V2.preload_data()
        df_ohlcv = None

    elif strategy == "v3":
        df, df_ohlcv = V3.preload_data()

    grid = generate_param_grid(strategy)
    print(f"Total parameter combos: {len(grid)}")

    tasks = []
    for p in grid:
        tasks.append((strategy, df, df_ohlcv, p))

    t0 = time.time()
    with mp.Pool(n_jobs) as pool:
        results = pool.map(worker_eval, tasks)
    dt = time.time() - t0

    print(f"\n⏱ Completed in {dt:.2f} seconds\n")

    outdir = REPORTS_DIR / "research" / "microstructure" / "optimizer"
    outdir.mkdir(parents=True, exist_ok=True)

    dfres = pd.DataFrame(results)
    dfres.to_csv(outdir / f"optimizer_results_{strategy}.csv", index=False)

    top = dfres.sort_values("sharpe", ascending=False).head(20)
    top.to_csv(outdir / f"optimizer_top_{strategy}.csv", index=False)

    print("TOP RESULTS:")
    print(top.head())

    # Example heatmap (customize as needed)
    if strategy == "v3":
        generate_heatmap(
            dfres,
            x_param="wick_factor",
            y_param="sl_buffer_atr",
            value_param="sharpe",
            outpath=outdir / "heatmap_sharpe_v3.png"
        )

    print(f"\nResults saved to: {outdir}\n")


if __name__ == "__main__":
    main()
