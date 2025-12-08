"""Simple benchmark runner for common backtesting scenarios.

This script generates synthetic OHLCV data, exercises a couple of
representative workloads (signal generation + backtest engine) and
collects timing and peak-memory stats using ``time.perf_counter`` and
``tracemalloc``. Results are persisted under ``bench/results`` so they
can be uploaded as CI artifacts.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import tracemalloc
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Dict, List

import numpy as np

# Allow running from repo root without installing as a package
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.append(str(REPO_ROOT / "src"))

from src.data.feeds import OHLCVArrays  # noqa: E402
from src.engine.core import BacktestConfig, run_backtest_basic  # noqa: E402
from src.strategies.microstructure_reversal import (  # noqa: E402
    StrategyMicrostructureReversal,
)


@dataclass
class BenchmarkResult:
    name: str
    seconds: float
    peak_mb: float
    notes: str

    def to_dict(self) -> Dict[str, float | str]:
        data = asdict(self)
        data["seconds"] = round(self.seconds, 4)
        data["peak_mb"] = round(self.peak_mb, 2)
        return data


def synthetic_ohlcv(n: int, seed: int = 42) -> OHLCVArrays:
    rng = np.random.default_rng(seed)
    timestamps = np.arange(n, dtype="datetime64[s]")

    prices = 100 + rng.standard_normal(n).cumsum()
    high = prices + rng.random(n)
    low = prices - rng.random(n)
    open_ = prices + rng.normal(0, 0.2, n)
    close = prices + rng.normal(0, 0.2, n)
    volume = rng.integers(50, 2000, size=n).astype(float)

    return OHLCVArrays(ts=timestamps, o=open_, h=high, low=low, c=close, v=volume)


def _benchmark(name: str, func: Callable[[], object], warmup: int = 1, notes: str = "") -> BenchmarkResult:
    # Warm up JIT paths (Numba) or caches so the measured run is stable
    for _ in range(max(0, warmup)):
        func()

    tracemalloc.start()
    start = time.perf_counter()
    func()
    elapsed = time.perf_counter() - start
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    peak_mb = peak / (1024 * 1024)
    return BenchmarkResult(name=name, seconds=elapsed, peak_mb=peak_mb, notes=notes)


def run_benchmarks(bars: int) -> List[BenchmarkResult]:
    data = synthetic_ohlcv(bars)
    strat = StrategyMicrostructureReversal()

    def _signals():
        return strat.generate_signals(data)

    def _backtest():
        return run_backtest_basic(data, config=BacktestConfig(max_bars_in_trade=60))

    results = [
        _benchmark("signals_microstructure_reversal", _signals, warmup=1, notes="Signal generation on synthetic OHLCV"),
        _benchmark("engine_backtest_basic", _backtest, warmup=1, notes="Numba-accelerated example strategy"),
    ]

    return results


def _write_outputs(results: List[BenchmarkResult], bars: int, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "bars": bars,
        "python": sys.version.split()[0],
        "results": [r.to_dict() for r in results],
    }

    json_path = output_dir / "benchmark_results.json"
    md_path = output_dir / "benchmark_results.md"

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    lines = [
        f"# Benchmark results (bars={bars})",
        "",
        "| Scenario | Seconds | Peak MB | Notes |",
        "| --- | --- | --- | --- |",
    ]
    for r in results:
        lines.append(
            f"| {r.name} | {r.seconds:.4f} | {r.peak_mb:.2f} | {r.notes} |"
        )

    with md_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"[bench] Wrote {json_path.relative_to(REPO_ROOT)}")
    print(f"[bench] Wrote {md_path.relative_to(REPO_ROOT)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark common backtesting workloads")
    parser.add_argument("--bars", type=int, default=150_000, help="Number of synthetic OHLCV bars to generate")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("bench/results"),
        help="Directory to write benchmark artifacts",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    bench_results = run_benchmarks(bars=args.bars)
    _write_outputs(bench_results, bars=args.bars, output_dir=args.output_dir)
    for result in bench_results:
        print(
            f"[bench] {result.name}: {result.seconds:.4f}s, "
            f"peak {result.peak_mb:.2f} MB ({result.notes})"
        )
