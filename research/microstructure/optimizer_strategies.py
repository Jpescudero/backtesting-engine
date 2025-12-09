"""
Unified Strategy Optimizer (Final Clean Version)
------------------------------------------------
- Compatible with Windows multiprocessing
- Supports V1, V2, V3
- Provides advanced outputs for analysis
"""

from __future__ import annotations
import sys
import time
import json
import itertools
from multiprocessing import Pool, cpu_count
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Tuple, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# PATH FIX FOR WINDOWS + PROJECT IMPORTING
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import research.microstructure.study_opening_sweeps_v1 as V1
import research.microstructure.study_opening_sweeps_v2 as V2
import research.microstructure.study_opening_sweeps_v3 as V3

from src.config.paths import REPORTS_DIR


# ============================================================
# DATA ADAPTER CLASS
# ============================================================

@dataclass
class HeatmapSpec:
    x: str
    y: str
    value: str
    filename: str


@dataclass
class StrategyAdapter:
    name: str
    param_grid: Dict[str, List[Any]]
    preload: Callable[[], Tuple[pd.DataFrame,Any]]
    run: Callable[[pd.DataFrame,Any,Dict[str,Any]], pd.DataFrame]
    heatmaps: Tuple[HeatmapSpec,...]


# ============================================================
# GLOBAL WRAPPERS FOR WINDOWS MULTIPROCESSING
# ============================================================

def v1_preload():
    return V1.preload_data()

def v1_run(df, context, params):
    return V1.run_with_params(df, context, params)

def v2_preload():
    return V2.preload_data()

def v2_run(df, context, params):
    return V2.run_with_params(df, context, params)

def v3_preload():
    return V3.preload_data()

def v3_run(df, context, params):
    return V3.run_with_params(df, context, params)


# ============================================================
# STRATEGY REGISTRATION
# ============================================================

def adapter_v1():
    grid = {
        "wick_factor": [1.2, 1.5, 1.8, 2.0],
        "atr_percentile": [0.2, 0.4, 0.6],
        "volume_percentile": [0.4, 0.6, 0.8],
    }

    heatmaps = (
        HeatmapSpec("wick_factor","atr_percentile","sharpe","v1_heatmap_wick_atr.png"),
    )

    return StrategyAdapter(
        name="v1",
        param_grid=grid,
        preload=v1_preload,
        run=v1_run,
        heatmaps=heatmaps,
    )


def adapter_v2():
    grid = {
        "wick_factor": [1.2,1.5,1.8,2.0],
        "atr_percentile": [0.2,0.4,0.6],
        "volume_percentile": [0.4,0.6,0.8],
    }

    heatmaps = (
        HeatmapSpec("wick_factor","atr_percentile","sharpe","v2_heatmap_wick_atr.png"),
    )

    return StrategyAdapter(
        name="v2",
        param_grid=grid,
        preload=v2_preload,
        run=v2_run,
        heatmaps=heatmaps,
    )


def adapter_v3():
    grid = {
        "wick_factor": [1.5, 1.8, 2.0],
        "atr_percentile": [0.4,0.5,0.6],
        "volume_percentile": [0.4,0.6,0.8],
        "sl_buffer_atr": [0.2,0.3,0.4],
        "sl_buffer_relative": [0.0,0.1,0.2],
        "tp_multiplier": [1.0,1.2,1.5],
    }

    heatmaps = (
        HeatmapSpec("wick_factor","sl_buffer_atr","sharpe","v3_heatmap_wick_sl.png"),
        HeatmapSpec("tp_multiplier","sl_buffer_relative","sharpe","v3_heatmap_tp_slrel.png"),
    )

    return StrategyAdapter(
        name="v3",
        param_grid=grid,
        preload=v3_preload,
        run=v3_run,
        heatmaps=heatmaps,
    )


ADAPTERS = {
    "v1": adapter_v1(),
    "v2": adapter_v2(),
    "v3": adapter_v3(),
}


# ============================================================
# EVALUATION
# ============================================================

def evaluate(trades: pd.DataFrame):
    if trades is None or len(trades)==0:
        return {"sharpe": -999, "mean": 0, "winrate": 0, "n_trades": 0}

    col = "r_multiple" if "r_multiple" in trades else "fwd_return"
    r = trades[col]

    sharpe = r.mean() / (r.std() + 1e-9)

    return {
        "sharpe": sharpe,
        "mean": r.mean(),
        "winrate": (r > 0).mean(),
        "n_trades": len(trades),
    }


# ============================================================
# WORKER FOR MULTIPROCESSING
# ============================================================

def worker(task):
    adapter, df, context, params = task
    trades = adapter.run(df, context, params)
    metrics = evaluate(trades)
    metrics.update(params)
    return metrics


# ============================================================
# HEATMAP UTILITY
# ============================================================

def generate_heatmap(dfres, spec: HeatmapSpec, outdir: Path):
    pivot = dfres.pivot_table(
        index=spec.y,
        columns=spec.x,
        values=spec.value,
        aggfunc="mean"
    )

    plt.figure(figsize=(8,5))
    plt.imshow(pivot, cmap="viridis", aspect="auto")
    plt.title(f"{spec.value} — {spec.y} vs {spec.x}")
    plt.colorbar()
    plt.xticks(range(len(pivot.columns)), pivot.columns)
    plt.yticks(range(len(pivot.index)), pivot.index)
    plt.tight_layout()
    plt.savefig(outdir / spec.filename, dpi=150)
    plt.close()


# ============================================================
# MAIN OPTIMIZER FUNCTION
# ============================================================

def run_optimizer(strategy: str, n_jobs: int):

    adapter = ADAPTERS[strategy]

    # PRELOAD DATA
    df, context = adapter.preload()

    grid = adapter.param_grid
    keys = list(grid.keys())
    values = list(grid.values())
    combos = [dict(zip(keys, v)) for v in itertools.product(*values)]

    tasks = [(adapter, df, context, p) for p in combos]

    # OUTPUT DIR
    outdir = REPORTS_DIR / "research" / "microstructure" / f"optimizer_{strategy}"
    outdir.mkdir(parents=True, exist_ok=True)

    # RUN
    t0 = time.time()
    with Pool(processes=n_jobs) as pool:
        results = pool.map(worker, tasks)
    dt = time.time() - t0

    dfres = pd.DataFrame(results)
    dfres.to_csv(outdir / f"optimizer_results_{strategy}.csv", index=False)

    top = dfres.sort_values("sharpe", ascending=False).head(20)
    top.to_csv(outdir / f"optimizer_top_{strategy}.csv", index=False)

    # SUMMARY FILE
    summary = outdir / f"optimizer_summary_{strategy}.txt"
    with open(summary,"w") as f:
        f.write(f"Optimizer Summary — {strategy.upper()}\n")
        f.write("="*60 + "\n\n")
        f.write(f"Completed in: {dt:.2f} seconds\n")
        f.write(f"Total combos: {len(combos)}\n\n")

        best = top.iloc[0]
        f.write("BEST PARAMETERS:\n")
        for k in keys:
            f.write(f"  {k}: {best[k]}\n")
        f.write(f"\nBest Sharpe: {best['sharpe']:.6f}\n")
        f.write(f"Mean: {best['mean']:.6f}\n")
        f.write(f"Winrate: {best['winrate']:.4f}\n")

    # HEATMAPS
    for spec in adapter.heatmaps:
        generate_heatmap(dfres, spec, outdir)

    print(f"Results saved to: {outdir}")


# ============================================================
# ENTRY POINT
# ============================================================

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", required=True, choices=["v1","v2","v3"])
    parser.add_argument("--n-jobs", type=int, default=max(1, cpu_count()-1))
    args = parser.parse_args()

    run_optimizer(args.strategy, args.n_jobs)


if __name__ == "__main__":
    main()
