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
    parser.add_argument("--config-file", type=str, default=None,
                        help="Ruta a archivo de configuración simple key=value")
    parser.add_argument("--initial-cash", type=float, default=None,
                        help="Capital inicial; si no se indica se toma del config o 100k")
    parser.add_argument("--commission", type=float, default=1.0)
    parser.add_argument("--trade-size", type=float, default=1.0)
    parser.add_argument("--slippage", type=float, default=0.0)
    parser.add_argument("--sl-pct", type=float, default=0.01)
    parser.add_argument("--tp-pct", type=float, default=0.02)
    parser.add_argument("--max-bars", type=int, default=60, help="Máximo de velas en una operación")
    parser.add_argument("--entry-threshold", type=float, default=0.0)
    parser.add_argument("--train-years", type=str, default=None,
                        help="Años de entrenamiento separados por comas, p.ej. 2019,2020")
    parser.add_argument("--test-years", type=str, default=None,
                        help="Años de test separados por comas, p.ej. 2021,2022")
    parser.add_argument(
        "--use-test-years",
        action="store_true",
        default=None,
        help="Si se marca, carga los años de test en lugar de los de train",
    )
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


def _parse_years(raw: str | None) -> list[int] | None:
    if not raw:
        return None
    try:
        return [int(y.strip()) for y in raw.split(",") if y.strip()]
    except ValueError as exc:
        raise ValueError(f"Formato de años inválido: '{raw}'") from exc


def _load_config_file(path: str | None) -> dict:
    if not path:
        return {}

    config: dict[str, str] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            clean = line.strip()
            if not clean or clean.startswith("#"):
                continue
            if "=" not in clean:
                raise ValueError(f"Línea inválida en config: '{clean}' (esperado key=value)")
            key, value = clean.split("=", maxsplit=1)
            config[key.strip()] = value.strip()
    return config


def _get_setting(
    cli_value,
    config: dict,
    key: str,
    default,
    transform=lambda x: x,
):
    if cli_value is not None:
        return cli_value
    if key in config:
        return transform(config[key])
    return default


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    config_file_values = _load_config_file(args.config_file)

    initial_cash = _get_setting(args.initial_cash, config_file_values, "initial_cash", 100_000.0, float)
    train_years = _get_setting(_parse_years(args.train_years), config_file_values, "train_years", None, _parse_years)
    test_years = _get_setting(_parse_years(args.test_years), config_file_values, "test_years", None, _parse_years)
    use_test_years = _get_setting(args.use_test_years, config_file_values, "use_test_years", False, lambda v: str(v).lower() == "true")

    strategy_params = StrategyParams(
        volume_percentile=args.volume_percentile,
        use_two_bearish_bars=not args.disable_two_bearish_bars,
    )

    backtest_config = BacktestConfig(
        initial_cash=initial_cash,
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
        train_years=train_years,
        test_years=test_years,
        use_test_years=use_test_years,
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
