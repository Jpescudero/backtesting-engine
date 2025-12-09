"""
Ultra-Fast Optimizer for V3 Only
--------------------------------
- ThreadPoolExecutor (faster than multiprocessing on Windows)
- Grid reduced for speed
- Full structural optimization
- Generates: results, summary, heatmaps
"""

from __future__ import annotations
import sys
import json
from pathlib import Path
from time import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import research.microstructure.study_opening_sweeps_v3 as V3
from src.config.paths import REPORTS_DIR


# ======================================================================
# GRID REDUCIDO
# ======================================================================

GRID_V3 = {
    "wick_factor": [1.5, 1.8, 2.0],
    "atr_percentile": [0.4, 0.5],     # reducido
    "volume_percentile": [0.4, 0.6],  # reducido
    "sl_buffer_atr": [0.3],           # fijo
    "sl_buffer_relative": [0.1],      # fijo
    "tp_multiplier": [1.0, 1.2],      # reducido
}


# ======================================================================
# UTILIDADES
# ======================================================================

def evaluate(trades):
    if len(trades) == 0:
        return -999, 0, 0, 0
    r = trades["r_multiple"]
    sharpe = r.mean() / (r.std()+1e-9)
    return sharpe, r.mean(), (r>0).mean(), len(trades)


def generate_heatmap(df, x, y, val, out):
    pivot = df.pivot_table(index=y, columns=x, values=val, aggfunc="mean")
    plt.figure(figsize=(6,4))
    plt.imshow(pivot, cmap="viridis", aspect="auto")
    plt.colorbar()
    plt.title(f"{val} — {y} vs {x}")
    plt.xticks(range(len(pivot.columns)), pivot.columns)
    plt.yticks(range(len(pivot.index)), pivot.index)
    plt.tight_layout()
    plt.savefig(out, dpi=140)
    plt.close()


# ======================================================================
# OPTIMIZADOR SOLO V3
# ======================================================================

def run_optimizer_v3():
    print("🔧 Preloading data...")
    df, context = V3.preload_data()

    param_keys = list(GRID_V3.keys())
    combos = []

    # Build combinations
    import itertools
    for values in itertools.product(*GRID_V3.values()):
        combos.append(dict(zip(param_keys, values)))

    print(f"Total combinations: {len(combos)}")

    results = []
    t0 = time()

    print("🚀 Running optimization (ThreadPool)...")

    with ThreadPoolExecutor(max_workers=8) as exe:
        futures = {
            exe.submit(V3.run_with_params, df, context, params): params
            for params in combos
        }

        for fut in as_completed(futures):
            params = futures[fut]
            trades = fut.result()
            sharpe, meanr, winr, count = evaluate(trades)
            row = {"sharpe": sharpe, "mean": meanr, "winrate": winr, "n_trades": count}
            row.update(params)
            results.append(row)

    dt = time() - t0
    print(f"⏱ Completed in {dt:.2f} seconds")

    dfres = pd.DataFrame(results)

    outdir = REPORTS_DIR / "research" / "microstructure" / "optimizer_v3_fast"
    outdir.mkdir(parents=True, exist_ok=True)

    dfres.to_csv(outdir / "optimizer_results_v3.csv", index=False)

    top = dfres.sort_values("sharpe", ascending=False).head(10)
    top.to_csv(outdir / "optimizer_top_v3.csv", index=False)

    # SUMMARY
    with open(outdir / "optimizer_summary_v3.txt","w") as f:
        f.write("V3 Optimizer Summary (FAST VERSION)\n")
        f.write("=====================================\n\n")
        best = top.iloc[0]
        f.write(f"Best Sharpe: {best['sharpe']:.6f}\n\nParameters:\n")
        for k in param_keys:
            f.write(f"  {k}: {best[k]}\n")

    # Heatmaps
    generate_heatmap(
        dfres,
        x="wick_factor",
        y="atr_percentile",
        val="sharpe",
        out=outdir / "heatmap_sharpe.png",
    )

    print(f"Results saved to: {outdir}")


# ======================================================================
# ENTRY POINT
# ======================================================================

def main():
    run_optimizer_v3()


if __name__ == "__main__":
    main()
