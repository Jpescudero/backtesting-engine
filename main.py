# main.py
"""
Punto de entrada para lanzar un backtest sencillo y generar
outputs fáciles de analizar/compartir.

Ahora delega la preparación de datos, el backtest y el reporting en
módulos reutilizables para mantener el archivo pequeño y testable.
"""

from __future__ import annotations

import argparse
import logging
from typing import Iterable, Sequence

from src.engine.core import BacktestConfig
from src.pipeline.backtest_runner import (
    BacktestRunConfig,
    StrategyParams,
    run_single_backtest,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ejecuta un backtest Barrida Apertura")
    parser.add_argument("--symbol", default="NDXm", help="Símbolo a backtestear")
    parser.add_argument("--timeframe", default="1m", help="Timeframe del feed NPZ")
    parser.add_argument("--volume-percentile", type=float, default=80.0,
                        help="Percentil de volumen para la estrategia")
    parser.add_argument("--disable-two-bearish-bars", action="store_true",
                        help="Desactivar condición de dos velas bajistas")
    parser.add_argument("--initial-cash", type=float, default=100_000.0)
    parser.add_argument("--commission", type=float, default=1.0)
    parser.add_argument("--trade-size", type=float, default=1.0)
    parser.add_argument("--slippage", type=float, default=0.0)
    parser.add_argument("--sl-pct", type=float, default=0.01)
    parser.add_argument("--tp-pct", type=float, default=0.02)
    parser.add_argument("--max-bars", type=int, default=60, help="Máximo de velas en una operación")
    parser.add_argument("--entry-threshold", type=float, default=0.0)
    parser.add_argument("--no-report-files", action="store_true", help="No generar Excel/JSON")
    parser.add_argument("--no-main-plots", action="store_true", help="No generar plots principales")
    parser.add_argument("--no-trade-plots", action="store_true", help="No generar plots de trades")
    parser.add_argument("--headless", action="store_true", help="Usar backend Agg y no mostrar ventanas")
    return parser.parse_args(argv)


def print_metrics(title: str, stats: dict) -> None:
    print(f"\n=== {title} ===")
    for k, v in stats.items():
        print(f"{k:25s}: {v}")


def print_timings(timings: dict) -> None:
    print("\n================ Tiempos de ejecución (segundos) ================")
    for key in sorted(timings.keys()):
        print(f"{key:35s}: {timings[key]:8.3f}")
    print("=================================================================")


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    strategy_params = StrategyParams(
        volume_percentile=args.volume_percentile,
        use_two_bearish_bars=not args.disable_two_bearish_bars,
    )

    backtest_config = BacktestConfig(
        initial_cash=args.initial_cash,
        commission_per_trade=args.commission,
        trade_size=args.trade_size,
        slippage=args.slippage,
        sl_pct=args.sl_pct,
        tp_pct=args.tp_pct,
        max_bars_in_trade=args.max_bars,
        entry_threshold=args.entry_threshold,
    )

    run_config = BacktestRunConfig(
        symbol=args.symbol,
        timeframe=args.timeframe,
        generate_report_files=not args.no_report_files,
        generate_main_plots=not args.no_main_plots,
        generate_trade_plots=not args.no_trade_plots,
        headless=args.headless,
        strategy_params=strategy_params,
        backtest_config=backtest_config,
    )

    artifacts = run_single_backtest(run_config)

    print_metrics("Métricas de equity (tipo Darwinex)", artifacts.equity_stats)
    print_metrics("Métricas de trades", artifacts.trade_stats)
    print("\n=== Tail de la curva de equity ===")
    print(artifacts.equity_series.tail())
    print("\n=== Primeros trades ===")
    print(artifacts.trades_df.head())

    if run_config.generate_report_files:
        print("\n=== Ficheros de resumen generados ===")
        print(f"Excel: {artifacts.reports.excel_path}")
        print(f"JSON:  {artifacts.reports.json_path}")
    if run_config.generate_main_plots:
        print(f"Plot equity/trades: {artifacts.reports.equity_path}")
    if run_config.generate_trade_plots:
        print(f"Plot mejores/peores trades: {artifacts.reports.trade_plot_path}")

    print_timings(artifacts.timings)


if __name__ == "__main__":
    main()
