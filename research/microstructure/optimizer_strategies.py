"""Unified and extensible optimizer for opening sweep research strategies."""

# ruff: noqa: E402, I001

from __future__ import annotations

import argparse
import itertools
import json
import multiprocessing as mp
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Sequence

import matplotlib.pyplot as plt
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import research.microstructure.study_opening_sweeps_v1 as V1  # noqa: E402
import research.microstructure.study_opening_sweeps_v2 as V2  # noqa: E402
import research.microstructure.study_opening_sweeps_v3 as V3  # noqa: E402

from src.config.paths import REPORTS_DIR  # noqa: E402

StrategyData = tuple[pd.DataFrame, Any]

_GLOBAL_ADAPTER: StrategyAdapter | None = None
_GLOBAL_DATA: pd.DataFrame | None = None
_GLOBAL_CONTEXT: Any = None


@dataclass
class HeatmapSpec:
    x_param: str
    y_param: str
    value_param: str
    filename: str


@dataclass
class StrategyAdapter:
    """Adapter that exposes a unified API for the optimizer."""

    name: str
    param_grid: dict[str, Sequence[float]]
    preload: Callable[[], StrategyData]
    run: Callable[[pd.DataFrame, Any, dict[str, float]], pd.DataFrame]
    heatmaps: Sequence[HeatmapSpec] = ()

    def parameter_sets(self) -> list[dict[str, float]]:
        keys = list(self.param_grid.keys())
        values = list(self.param_grid.values())
        combos = [dict(zip(keys, vals)) for vals in itertools.product(*values)]
        return combos


def _init_worker(adapter: StrategyAdapter, data: pd.DataFrame, context: Any) -> None:
    global _GLOBAL_ADAPTER, _GLOBAL_DATA, _GLOBAL_CONTEXT
    _GLOBAL_ADAPTER = adapter
    _GLOBAL_DATA = data
    _GLOBAL_CONTEXT = context


def _worker_eval(params: dict[str, float]) -> dict[str, Any]:
    if _GLOBAL_ADAPTER is None or _GLOBAL_DATA is None:
        raise RuntimeError("Worker not initialized")

    trades = _GLOBAL_ADAPTER.run(_GLOBAL_DATA, _GLOBAL_CONTEXT, params)
    metrics = evaluate_trade_results(trades)
    metrics.update(params)
    return metrics

StrategyData = tuple[pd.DataFrame, Any]

_GLOBAL_ADAPTER: StrategyAdapter | None = None
_GLOBAL_DATA: pd.DataFrame | None = None
_GLOBAL_CONTEXT: Any = None


@dataclass
class HeatmapSpec:
    x_param: str
    y_param: str
    value_param: str
    filename: str


@dataclass
class StrategyAdapter:
    """Adapter that exposes a unified API for the optimizer."""

    name: str
    param_grid: dict[str, Sequence[float]]
    preload: Callable[[], StrategyData]
    run: Callable[[pd.DataFrame, Any, dict[str, float]], pd.DataFrame]
    heatmaps: Sequence[HeatmapSpec] = ()

    def parameter_sets(self) -> list[dict[str, float]]:
        keys = list(self.param_grid.keys())
        values = list(self.param_grid.values())
        combos = [dict(zip(keys, vals)) for vals in itertools.product(*values)]
        return combos


def _init_worker(adapter: StrategyAdapter, data: pd.DataFrame, context: Any) -> None:
    global _GLOBAL_ADAPTER, _GLOBAL_DATA, _GLOBAL_CONTEXT
    _GLOBAL_ADAPTER = adapter
    _GLOBAL_DATA = data
    _GLOBAL_CONTEXT = context


def _worker_eval(params: dict[str, float]) -> dict[str, Any]:
    if _GLOBAL_ADAPTER is None or _GLOBAL_DATA is None:
        raise RuntimeError("Worker not initialized")

    trades = _GLOBAL_ADAPTER.run(_GLOBAL_DATA, _GLOBAL_CONTEXT, params)
    metrics = evaluate_trade_results(trades)
    metrics.update(params)
    return metrics


# =============================================================
# EVALUATION
# =============================================================


def evaluate_trade_results(trades_df: pd.DataFrame) -> dict[str, float]:
    """Compute simple performance metrics from a trade dataframe."""

    if len(trades_df) < 20:
        return {
            "sharpe": -999.0,
            "mean": 0.0,
            "winrate": 0.0,
            "n_trades": float(len(trades_df)),
        }

    if "r_multiple" in trades_df:
        returns = trades_df["r_multiple"]
    else:
        returns = trades_df["fwd_return"]

    sharpe = returns.mean() / (returns.std() + 1e-9)
    return {
        "sharpe": float(sharpe),
        "mean": float(returns.mean()),
        "winrate": float((returns > 0).mean()),
        "n_trades": float(len(trades_df)),
    }


# =============================================================
# REPORTING
# =============================================================


def _optimizer_folder(strategy: str) -> Path:
    stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    out = REPORTS_DIR / "research" / "microstructure" / "reports" / "optimizer"
    out = out / f"{strategy}_{stamp}"
    out.mkdir(parents=True, exist_ok=True)
    return out


def _generate_heatmap(dfres: pd.DataFrame, spec: HeatmapSpec, outdir: Path) -> None:
    pivot = dfres.pivot_table(index=spec.y_param, columns=spec.x_param, values=spec.value_param)
    plt.figure(figsize=(8, 6))
    plt.title(f"{spec.value_param} by {spec.x_param} / {spec.y_param}")
    plt.imshow(pivot, cmap="viridis", aspect="auto")
    plt.colorbar(label=spec.value_param)
    plt.xticks(range(len(pivot.columns)), pivot.columns)
    plt.yticks(range(len(pivot.index)), pivot.index)
    plt.tight_layout()
    plt.savefig(outdir / spec.filename, dpi=150)
    plt.close()


def _persist_reports(
    strategy: str,
    dfres: pd.DataFrame,
    runtime: float,
    outdir: Path,
    heatmaps: Sequence[HeatmapSpec],
) -> None:
    dfres.to_csv(outdir / "optimizer_results.csv", index=False)

    top = dfres.sort_values("sharpe", ascending=False).head(20)
    top.to_csv(outdir / "optimizer_top.csv", index=False)

    summary = {
        "strategy": strategy,
        "runtime_seconds": runtime,
        "evaluations": len(dfres),
        "best": top.iloc[0].to_dict() if not top.empty else {},
        "timestamp": datetime.utcnow().isoformat(),
    }
    with open(outdir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    for spec in heatmaps:
        _generate_heatmap(dfres, spec, outdir)


# =============================================================
# STRATEGY REGISTRY
# =============================================================


def _adapter_v1() -> StrategyAdapter:
    grid = {
        "wick_factor": [1.2, 1.5, 1.8, 2.0],
        "atr_percentile": [0.2, 0.4, 0.6],
        "volume_percentile": [0.4, 0.6, 0.8],
    }

    def _preload() -> StrategyData:
        return V1.preload_data(), None

    def _run(df: pd.DataFrame, _: Any, params: dict[str, float]) -> pd.DataFrame:
        return V1.run_with_params(df, params)

    return StrategyAdapter(
        name="v1",
        param_grid=grid,
        preload=_preload,
        run=_run,
    )


def _adapter_v2() -> StrategyAdapter:
    grid = {
        "wick_factor": [1.2, 1.5, 1.8, 2.0],
        "atr_percentile": [0.2, 0.4, 0.6],
        "volume_percentile": [0.4, 0.6, 0.8],
    }

    def _preload() -> StrategyData:
        return V2.preload_data(), None

    def _run(df: pd.DataFrame, _: Any, params: dict[str, float]) -> pd.DataFrame:
        return V2.run_with_params(df, params)

    return StrategyAdapter(
        name="v2",
        param_grid=grid,
        preload=_preload,
        run=_run,
    )


def _adapter_v3() -> StrategyAdapter:
    grid = {
        "wick_factor": [1.5, 1.8, 2.0],
        "atr_percentile": [0.4, 0.5, 0.6],
        "volume_percentile": [0.4, 0.6, 0.8],
        "sl_buffer_atr": [0.2, 0.3, 0.4],
        "sl_buffer_relative": [0.0, 0.1, 0.2],
        "tp_multiplier": [1.0, 1.2, 1.5],
    }

    heatmaps = (
        HeatmapSpec(
            x_param="wick_factor",
            y_param="sl_buffer_atr",
            value_param="sharpe",
            filename="heatmap_wick_slbuffer.png",
        ),
        HeatmapSpec(
            x_param="tp_multiplier",
            y_param="sl_buffer_relative",
            value_param="sharpe",
            filename="heatmap_tp_slrel.png",
        ),
    )

    def _preload() -> StrategyData:
        return V3.preload_data()

    def _run(df: pd.DataFrame, context: Any, params: dict[str, float]) -> pd.DataFrame:
        return V3.run_with_params(df, context, params)

    return StrategyAdapter(
        name="v3",
        param_grid=grid,
        preload=_preload,
        run=_run,
        heatmaps=heatmaps,
    )


ADAPTERS: dict[str, Callable[[], StrategyAdapter]] = {
    "v1": _adapter_v1,
    "v2": _adapter_v2,
    "v3": _adapter_v3,
}


# =============================================================
# MAIN OPTIMIZER
# =============================================================


def run_optimizer(strategy_key: str, n_jobs: int) -> Path:
    if strategy_key not in ADAPTERS:
        raise ValueError("strategy must be v1, v2 or v3")

    adapter = ADAPTERS[strategy_key]()
    data, context = adapter.preload()

    params_grid = adapter.parameter_sets()
    print(f"Total parameter combos: {len(params_grid)}")

    outdir = _optimizer_folder(strategy_key)

    start = time.time()
    with mp.Pool(
        processes=n_jobs,
        initializer=_init_worker,
        initargs=(adapter, data, context),
    ) as pool:
        results = pool.map(_worker_eval, params_grid)
    runtime = time.time() - start

    dfres = pd.DataFrame(results)
    _persist_reports(adapter.name, dfres, runtime, outdir, adapter.heatmaps)

    top = dfres.sort_values("sharpe", ascending=False).head(5)
    print("TOP RESULTS:\n", top)
    print(f"\nResults saved to: {outdir}\n")
    return outdir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", required=True, choices=list(ADAPTERS.keys()))
    parser.add_argument("--n-jobs", type=int, default=mp.cpu_count() - 1)
    args = parser.parse_args()

    run_optimizer(args.strategy, args.n_jobs)


if __name__ == "__main__":
    main()
