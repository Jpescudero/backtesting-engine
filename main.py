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
import math
from pathlib import Path
from typing import Iterable, Sequence

from src.engine.core import BacktestConfig
from src.pipeline.backtest_runner import (
    BacktestRunConfig,
    StrategyParams,
    run_single_backtest,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ejecuta un backtest Microstructure Reversal")
    parser.add_argument("--symbol", default="NDXm", help="Símbolo o índice a backtestear")
    parser.add_argument("--timeframe", default="1m", help="Timeframe del feed NPZ")
    parser.add_argument(
        "--strategy",
        default="microstructure_reversal",
        choices=["microstructure_reversal"],
        help="Estrategia a ejecutar",
    )
    parser.add_argument("--ema-short", type=int, default=20, help="EMA corta para filtro de tendencia")
    parser.add_argument("--ema-long", type=int, default=50, help="EMA larga para filtro de tendencia")
    parser.add_argument("--atr-period", type=int, default=20, help="Periodo ATR para normalizar rangos")
    parser.add_argument(
        "--atr-timeframe",
        type=str,
        default="1m",
        help="Timeframe dedicado para el cálculo del ATR (p.ej. 1m)",
    )
    parser.add_argument(
        "--atr-timeframe-period",
        type=int,
        default=10,
        help="Periodo ATR a usar en el timeframe dedicado",
    )
    parser.add_argument("--min-pullback-atr", type=float, default=0.3, help="Retroceso mínimo en ATR")
    parser.add_argument("--max-pullback-atr", type=float, default=1.3, help="Retroceso máximo en ATR")
    parser.add_argument("--max-pullback-bars", type=int, default=12, help="Velas máximas del pullback")
    parser.add_argument("--exhaustion-close-min", type=float, default=0.35, help="Posición mínima del cierre de la vela de agotamiento")
    parser.add_argument("--exhaustion-close-max", type=float, default=0.65, help="Posición máxima del cierre de la vela de agotamiento")
    parser.add_argument("--exhaustion-body-max-ratio", type=float, default=0.5, help="Relación máxima cuerpo/rango para la vela de agotamiento")
    parser.add_argument("--shift-body-atr", type=float, default=0.45, help="Mínimo cuerpo de la vela shift en ATR")
    parser.add_argument("--structure-break-lookback", type=int, default=3, help="Ventana de ruptura de microestructura")
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


def _format_number(value, decimals: int = 2, pct: bool = False) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float) and math.isnan(value):
        return "n/a"
    if pct:
        return f"{value * 100:.{decimals}f}%"
    if isinstance(value, float):
        return f"{value:.{decimals}f}"
    return str(value)


def _format_money(value) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float) and math.isnan(value):
        return "n/a"
    return f"{value:,.2f}"


def print_metrics(title: str, stats: dict, ordered_keys: Sequence[str]) -> None:
    print(f"\n=== {title} ===")
    pct_keys = {"total_return", "annualized_return", "max_drawdown", "winrate", "var_95_monthly"}
    for key in ordered_keys:
        if key not in stats:
            continue
        val = stats[key]
        is_pct = key in pct_keys
        print(f"{key:25s}: {_format_number(val, decimals=4 if is_pct else 2, pct=is_pct)}")


def print_timings(timings: dict) -> None:
    print("\n================ Tiempos de ejecución (segundos) ================")
    total_time = sum(timings.values())
    for key in sorted(timings.keys()):
        step_time = timings[key]
        pct = (step_time / total_time * 100.0) if total_time > 0 else 0.0
        print(f"{key:35s}: {step_time:8.3f} s ({pct:5.1f}%)")
    print(f"{'TOTAL':35s}: {total_time:8.3f} s (100.0%)")
    print("=================================================================")


def _print_run_context(run_config: BacktestRunConfig) -> None:
    cfg = run_config.backtest_config
    header = [
        ("Símbolo/TF", f"{run_config.symbol} / {run_config.timeframe}"),
        ("Capital inicial", _format_money(cfg.initial_cash)),
        ("Comisión por trade", _format_number(cfg.commission_per_trade)),
        ("Slippage", _format_number(cfg.slippage)),
        ("Tamaño trade", _format_number(cfg.trade_size)),
        ("SL %", _format_number(cfg.sl_pct, pct=True)),
        ("TP %", _format_number(cfg.tp_pct, pct=True)),
        ("Max barras trade", _format_number(cfg.max_bars_in_trade)),
        ("Umbral entrada", _format_number(cfg.entry_threshold)),
        (
            "Estrategia",
            run_config.strategy_name,
        ),
        (
            "Params estrategia",
            " | ".join(
                [
                    f"EMA {run_config.strategy_params.ema_short}/{run_config.strategy_params.ema_long}",
                    "ATR tf "
                    f"{run_config.strategy_params.atr_timeframe}"
                    f" p{run_config.strategy_params.atr_timeframe_period}",
                    f"ATR {run_config.strategy_params.atr_period}",
                    f"pullback {run_config.strategy_params.min_pullback_atr}-{run_config.strategy_params.max_pullback_atr} ATR",
                    f"shift ≥{run_config.strategy_params.shift_body_atr} ATR",
                ]
            ),
        ),
        ("Años train", ",".join(map(str, run_config.train_years)) if run_config.train_years else "(todos)"),
        (
            "Años test",
            ",".join(map(str, run_config.test_years)) if run_config.test_years else "(no definidos)",
        ),
        ("Usar años test", "sí" if run_config.use_test_years else "no"),
        ("Generar reportes", "sí" if run_config.generate_report_files else "no"),
        ("Plots principales", "sí" if run_config.generate_main_plots else "no"),
        ("Plots trades", "sí" if run_config.generate_trade_plots else "no"),
        ("Headless", "sí" if run_config.headless else "no"),
    ]

    print("=== Configuración de ejecución ===")
    for label, value in header:
        print(f"{label:25s}: {value}")


def print_headline_kpis(equity_stats: dict, trade_stats: dict) -> None:
    kpis = [
        ("Retorno total", equity_stats.get("total_return"), True),
        ("Max drawdown", equity_stats.get("max_drawdown"), True),
        ("Sharpe", equity_stats.get("sharpe_ratio"), False),
        ("# trades", trade_stats.get("n_trades"), False),
        ("Winrate", trade_stats.get("winrate"), True),
    ]

    print("\n=== KPIs rápidos ===")
    for label, value, is_pct in kpis:
        print(f"{label:25s}: {_format_number(value, decimals=4 if is_pct else 2, pct=is_pct)}")


def _describe_artifact(path: str | Path | None) -> str:
    if not path:
        return "(no generado)"
    p = Path(path)
    if not p.exists():
        return f"{p} (no encontrado)"
    size_kb = p.stat().st_size / 1024.0
    return f"{p} ({size_kb:.1f} KB)"


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


def _resolve_config_file(cli_path: str | None) -> str | None:
    """Devuelve la ruta del archivo de configuración a usar.

    Prioriza la ruta explícita de CLI; si no existe, intenta usar un archivo
    `run_settings.txt` en el directorio del proyecto y, en su defecto,
    `run_settings.example.txt`. Si ninguno existe, devuelve ``None`` para
    seguir con los valores por defecto en código.
    """

    if cli_path:
        return cli_path

    cwd = Path(__file__).resolve().parent
    project_root = cwd
    # El script vive en la raíz del repo; por claridad dejamos el cálculo
    # explícito en caso de moverlo en el futuro.
    candidates = [project_root / "run_settings.txt", project_root / "run_settings.example.txt"]

    for candidate in candidates:
        if candidate.exists():
            return str(candidate)

    return None


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

    config_file_values = _load_config_file(_resolve_config_file(args.config_file))

    symbol = _get_setting(args.symbol, config_file_values, "symbol", "NDXm")
    timeframe = _get_setting(args.timeframe, config_file_values, "timeframe", "1m")
    strategy_name = _get_setting(args.strategy, config_file_values, "strategy", "microstructure_reversal")

    initial_cash = _get_setting(args.initial_cash, config_file_values, "initial_cash", 100_000.0, float)
    train_years = _get_setting(_parse_years(args.train_years), config_file_values, "train_years", None, _parse_years)
    test_years = _get_setting(_parse_years(args.test_years), config_file_values, "test_years", None, _parse_years)
    use_test_years = _get_setting(args.use_test_years, config_file_values, "use_test_years", False, lambda v: str(v).lower() == "true")

    strategy_params = StrategyParams(
        ema_short=args.ema_short,
        ema_long=args.ema_long,
        atr_period=args.atr_period,
        atr_timeframe=args.atr_timeframe,
        atr_timeframe_period=args.atr_timeframe_period,
        min_pullback_atr=args.min_pullback_atr,
        max_pullback_atr=args.max_pullback_atr,
        max_pullback_bars=args.max_pullback_bars,
        exhaustion_close_min=args.exhaustion_close_min,
        exhaustion_close_max=args.exhaustion_close_max,
        exhaustion_body_max_ratio=args.exhaustion_body_max_ratio,
        shift_body_atr=args.shift_body_atr,
        structure_break_lookback=args.structure_break_lookback,
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
        symbol=symbol,
        timeframe=timeframe,
        strategy_name=strategy_name,
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

    _print_run_context(run_config)

    artifacts = run_single_backtest(run_config)

    print_headline_kpis(artifacts.equity_stats, artifacts.trade_stats)

    equity_keys = [
        "start_equity",
        "end_equity",
        "total_return",
        "annualized_return",
        "max_drawdown",
        "return_drawdown_ratio",
        "sharpe_ratio",
        "sortino_ratio",
        "volatility_annual",
        "var_95_monthly",
        "n_days",
        "n_months",
    ]
    trade_keys = [
        "n_trades",
        "winrate",
        "avg_pnl",
        "avg_win",
        "avg_loss",
        "payoff_ratio",
        "expectancy_per_trade",
        "avg_holding_bars",
        "exit_reason_counts",
    ]

    print_metrics("Métricas de equity (tipo Darwinex)", artifacts.equity_stats, equity_keys)
    print_metrics("Métricas de trades", artifacts.trade_stats, trade_keys)
    print("\n=== Tail de la curva de equity ===")
    print(artifacts.equity_series.tail().to_frame("equity"))
    print("\n=== Primeros trades ===")
    print(artifacts.trades_df.head())
    if not artifacts.trades_df.empty:
        print("\n=== Últimos trades ===")
        print(artifacts.trades_df.tail())

    if run_config.generate_report_files:
        print("\n=== Ficheros de resumen generados ===")
        print(f"Excel: {_describe_artifact(artifacts.reports.excel_path)}")
        print(f"JSON:  {_describe_artifact(artifacts.reports.json_path)}")
    if run_config.generate_main_plots:
        print(f"Plot equity/trades: {_describe_artifact(artifacts.reports.equity_path)}")
    if run_config.generate_trade_plots:
        print(
            "Plot mejores trades: "
            f"{_describe_artifact(artifacts.reports.best_trade_plot_path)}"
        )
        print(
            "Plot peores trades: "
            f"{_describe_artifact(artifacts.reports.worst_trade_plot_path)}"
        )

    print_timings(artifacts.timings)


if __name__ == "__main__":
    main()
