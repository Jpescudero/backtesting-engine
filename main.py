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

from src.config.paths import REPORTS_DIR, bootstrap_data_roots
from src.engine.core import BacktestConfig
from src.pipeline.backtest_runner import (
    BacktestRunConfig,
    StrategyParams,
    load_run_config_from_metadata,
    run_single_backtest,
)
from src.pipeline.reporting import _strategy_suffix
from src.strategies.microstructure_sweep import SweepParams
from src.strategies.opening_sweep_v4 import OpeningSweepV4Params
from src.visualization.trades_dashboard import build_trades_dashboard


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ejecuta un backtest")
    parser.add_argument("--symbol", default=None, help="Símbolo o índice a backtestear")
    parser.add_argument("--timeframe", default=None, help="Timeframe del feed NPZ")
    parser.add_argument(
        "--strategy",
        default=None,
        choices=["microstructure_reversal", "microstructure_sweep", "opening_sweep_v4"],
        help="Estrategia a ejecutar",
    )
    parser.add_argument(
        "--ema-short", type=int, default=None, help="EMA corta para filtro de tendencia"
    )
    parser.add_argument(
        "--ema-long", type=int, default=None, help="EMA larga para filtro de tendencia"
    )
    parser.add_argument(
        "--atr-period", type=int, default=None, help="Periodo ATR para normalizar rangos"
    )
    parser.add_argument(
        "--atr-timeframe",
        type=str,
        default=None,
        help="Timeframe dedicado para el cálculo del ATR (p.ej. 1m)",
    )
    parser.add_argument(
        "--atr-timeframe-period",
        type=int,
        default=None,
        help="Periodo ATR a usar en el timeframe dedicado",
    )
    parser.add_argument(
        "--min-pullback-atr", type=float, default=None, help="Retroceso mínimo en ATR"
    )
    parser.add_argument(
        "--max-pullback-atr", type=float, default=None, help="Retroceso máximo en ATR"
    )
    parser.add_argument(
        "--max-pullback-bars", type=int, default=None, help="Velas máximas del pullback"
    )
    parser.add_argument(
        "--exhaustion-close-min",
        type=float,
        default=None,
        help="Posición mínima del cierre de la vela de agotamiento",
    )
    parser.add_argument(
        "--exhaustion-close-max",
        type=float,
        default=None,
        help="Posición máxima del cierre de la vela de agotamiento",
    )
    parser.add_argument(
        "--exhaustion-body-max-ratio",
        type=float,
        default=None,
        help="Relación máxima cuerpo/rango para la vela de agotamiento",
    )
    parser.add_argument(
        "--shift-body-atr", type=float, default=None, help="Mínimo cuerpo de la vela shift en ATR"
    )
    parser.add_argument(
        "--structure-break-lookback",
        type=int,
        default=None,
        help="Ventana de ruptura de microestructura",
    )

    opening_defaults = OpeningSweepV4Params()

    # Parámetros Microstructure Sweep
    parser.add_argument(
        "--sweep-lookback",
        type=int,
        default=None,
        help="Ventana de lookback para mínimos previos",
    )
    parser.add_argument(
        "--min-sweep-break-atr",
        type=float,
        default=None,
        help="Mínima ruptura del mínimo previo en ATR",
    )
    parser.add_argument(
        "--min-lower-wick-body-ratio",
        type=float,
        default=None,
        help="Relación mínima mecha/cuerpo",
    )
    parser.add_argument(
        "--min-sweep-range-atr",
        type=float,
        default=None,
        help="Rango mínimo de la vela de barrida en ATR",
    )
    parser.add_argument(
        "--confirm-body-atr",
        type=float,
        default=None,
        help="Cuerpo mínimo de la vela de confirmación en ATR",
    )
    parser.add_argument(
        "--no-confirm-close-above-mid",
        action="store_false",
        dest="confirm_close_above_mid",
        help="Permitir cierres por debajo de la mitad de la vela de barrida",
    )
    parser.set_defaults(confirm_close_above_mid=None)
    parser.add_argument(
        "--volume-period",
        type=int,
        default=None,
        help="Periodo para volumen medio",
    )
    parser.add_argument(
        "--min-rvol", type=float, default=None, help="Volumen relativo mínimo"
    )
    parser.add_argument(
        "--vol-percentile-min",
        type=float,
        default=None,
        help="Percentil inferior de volumen intradía",
    )
    parser.add_argument(
        "--vol-percentile-max",
        type=float,
        default=None,
        help="Percentil superior de volumen intradía",
    )
    parser.add_argument(
        "--no-trend-filter",
        action="store_false",
        dest="use_trend_filter",
        help="Desactivar filtro de tendencia EMA",
    )
    parser.set_defaults(use_trend_filter=None)
    parser.add_argument(
        "--max-atr-mult-intraday",
        type=float,
        default=None,
        help="Umbral máximo de ATR intradía",
    )
    parser.add_argument(
        "--max-trades-per-day",
        type=int,
        default=None,
        help="Máximo de operaciones diarias",
    )
    parser.add_argument(
        "--sweep-max-holding-bars",
        type=int,
        default=None,
        help="Máximo de velas en posición para Sweep",
    )
    parser.add_argument(
        "--atr-stop-mult",
        type=float,
        default=None,
        help="Buffer ATR para stop loss",
    )
    parser.add_argument(
        "--rr-multiple",
        type=float,
        default=None,
        help="Multiplicador RR para TP",
    )
    # Parámetros Opening Sweep V4
    parser.add_argument(
        "--wick-factor",
        type=float,
        default=None,
        help="Mínima relación mecha/cuerpo para validar la barrida",
    )
    parser.add_argument(
        "--atr-percentile",
        type=float,
        default=None,
        help="Percentil de ATR normalizado requerido para operar",
    )
    parser.add_argument(
        "--volume-percentile",
        type=float,
        default=None,
        help="Percentil de volumen intradía requerido para operar",
    )
    parser.add_argument(
        "--sl-buffer-atr",
        type=float,
        default=None,
        help="Buffer ATR absoluto usado en el stop loss",
    )
    parser.add_argument(
        "--sl-buffer-relative",
        type=float,
        default=None,
        help="Buffer relativo (ATR normalizado) usado en el stop loss",
    )
    parser.add_argument(
        "--tp-multiplier",
        type=float,
        default=None,
        help="Multiplicador RR para calcular el take profit",
    )
    parser.add_argument(
        "--max-horizon",
        type=int,
        default=None,
        help="Máximo de barras a mantener la posición en Opening Sweep",
    )
    parser.add_argument(
        "--config-file",
        type=str,
        default=None,
        help="Ruta a archivo de configuración simple key=value",
    )
    parser.add_argument(
        "--sync-cloud",
        dest="sync_cloud",
        action="store_true",
        default=None,
        help=("Sincroniza el hub de datos local hacia los mirrors en la nube configurados"),
    )
    parser.add_argument(
        "--no-sync-cloud",
        dest="sync_cloud",
        action="store_false",
        help=("Desactiva la sincronización cloud aunque esté activada en run_settings"),
    )
    parser.set_defaults(sync_cloud=None)
    parser.add_argument("--seed", type=int, default=None, help="Seed global para reproducibilidad")
    parser.add_argument(
        "--snapshot-interval",
        type=int,
        default=None,
        help="Guardar snapshots de estado cada N eventos",
    )
    parser.add_argument(
        "--snapshot-path", type=str, default=None, help="Ruta personalizada para snapshots"
    )
    parser.add_argument(
        "--resume-snapshot",
        type=str,
        default=None,
        help="Ruta a snapshot desde el que reanudar el backtest",
    )
    parser.add_argument(
        "--run-metadata",
        type=str,
        default=None,
        help="Ruta donde escribir los metadatos completos del run",
    )
    parser.add_argument(
        "--replay-metadata",
        type=str,
        default=None,
        help="Ruta a metadatos previos para reproducir la ejecución",
    )
    parser.add_argument(
        "--initial-cash",
        type=float,
        default=None,
        help="Capital inicial; si no se indica se toma del config o 100k",
    )
    parser.add_argument("--trade-size", type=float, default=1.0)
    parser.add_argument("--sl-pct", type=float, default=0.01)
    parser.add_argument("--tp-pct", type=float, default=0.02)
    parser.add_argument(
        "--costs-path",
        type=str,
        default="config/costs/costs.yaml",
        help="Ruta al YAML de costes centralizado",
    )
    parser.add_argument(
        "--cost-instrument",
        type=str,
        default=None,
        help="Instrumento a cargar desde el YAML de costes (por defecto, el símbolo)",
    )
    parser.add_argument("--max-bars", type=int, default=60, help="Máximo de velas en una operación")
    parser.add_argument("--entry-threshold", type=float, default=0.0)
    parser.add_argument(
        "--train-years",
        type=str,
        default=None,
        help="Años de entrenamiento separados por comas, p.ej. 2019,2020",
    )
    parser.add_argument(
        "--test-years",
        type=str,
        default=None,
        help="Años de test separados por comas, p.ej. 2021,2022",
    )
    parser.add_argument(
        "--use-test-years",
        action="store_true",
        default=None,
        help="Si se marca, carga los años de test en lugar de los de train",
    )
    parser.add_argument("--no-report-files", action="store_true", help="No generar Excel/JSON")
    parser.add_argument("--no-main-plots", action="store_true", help="No generar plots principales")
    parser.add_argument("--no-trade-plots", action="store_true", help="No generar plots de trades")
    parser.add_argument(
        "--headless", action="store_true", help="Usar backend Agg y no mostrar ventanas"
    )
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


def print_level_metrics(stats: dict) -> None:
    print("\n=== Estadísticas de niveles SL/TP y break-even ===")
    distance_sections = {
        "SL distancia %": stats.get("sl_distance_pct", {}),
        "TP distancia %": stats.get("tp_distance_pct", {}),
        "SL distancia abs": stats.get("sl_distance_abs", {}),
        "TP distancia abs": stats.get("tp_distance_abs", {}),
    }
    for title, values in distance_sections.items():
        if not values:
            continue
        print(f"- {title}")
        for key in ["mean", "median", "min", "max", "p25", "p75", "std", "count"]:
            if key not in values:
                continue
            val = values[key]
            print(f"    {key:8s}: {_format_number(val, decimals=4)}")

    breakeven_count = stats.get("breakeven_count", 0)
    breakeven_rate = stats.get("breakeven_rate", 0.0)
    print("- Break-even")
    print(f"    count: {breakeven_count}")
    print(f"    rate:  {breakeven_rate * 100:.2f}%")


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
    atr_timeframe = getattr(run_config.strategy_params, "atr_timeframe", "n/a")
    atr_period = getattr(run_config.strategy_params, "atr_timeframe_period", "n/a")
    if run_config.strategy_name == "microstructure_sweep":
        params_summary = " | ".join(
            [
                f"EMA {run_config.strategy_params.ema_short}/{run_config.strategy_params.ema_long}",
                f"ATR tf {atr_timeframe} p{atr_period}",
                f"ATR {run_config.strategy_params.atr_period}",
                f"sweep lb {run_config.strategy_params.sweep_lookback}",
                f"wick≥{run_config.strategy_params.min_lower_wick_body_ratio}×cuerpo",
                f"body conf ≥{run_config.strategy_params.confirm_body_atr} ATR",
                f"RR {run_config.strategy_params.rr_multiple}x",
            ]
        )
    elif run_config.strategy_name == "opening_sweep_v4":
        params_summary = " | ".join(
            [
                f"wick≥{run_config.strategy_params.wick_factor}×cuerpo",
                f"ATR pct≥{run_config.strategy_params.atr_percentile}",
                f"Vol pct≥{run_config.strategy_params.volume_percentile}",
                f"SL buff {run_config.strategy_params.sl_buffer_atr}+"
                f"{run_config.strategy_params.sl_buffer_relative}×ATRnorm",
                f"TP×{run_config.strategy_params.tp_multiplier}",
                f"max {run_config.strategy_params.max_horizon} barras",
            ]
        )
    else:
        pullback_range = (
            f"pullback {run_config.strategy_params.min_pullback_atr}-"
            f"{run_config.strategy_params.max_pullback_atr} ATR"
        )
        params_summary = " | ".join(
            [
                f"EMA {run_config.strategy_params.ema_short}/{run_config.strategy_params.ema_long}",
                f"ATR tf {atr_timeframe} p{atr_period}",
                f"ATR {run_config.strategy_params.atr_period}",
                pullback_range,
                f"shift ≥{run_config.strategy_params.shift_body_atr} ATR",
            ]
        )
    header = [
        ("Símbolo/TF", f"{run_config.symbol} / {run_config.timeframe}"),
        ("Capital inicial", _format_money(cfg.initial_cash)),
        ("Instrumento costes", run_config.cost_instrument or run_config.symbol),
        ("Archivo costes", str(run_config.cost_config_path)),
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
            params_summary,
        ),
        (
            "Años train",
            ",".join(map(str, run_config.train_years)) if run_config.train_years else "(todos)",
        ),
        (
            "Años test",
            (
                ",".join(map(str, run_config.test_years))
                if run_config.test_years
                else "(no definidos)"
            ),
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
    total_return = equity_stats.get("net_total_return", equity_stats.get("gross_total_return"))
    max_drawdown = equity_stats.get("net_max_drawdown", equity_stats.get("gross_max_drawdown"))
    sharpe = equity_stats.get("net_sharpe_ratio", equity_stats.get("gross_sharpe_ratio"))
    n_trades = trade_stats.get("net_n_trades", trade_stats.get("n_trades"))
    winrate = trade_stats.get("net_winrate", trade_stats.get("winrate"))
    kpis = [
        ("Retorno total", total_return, True),
        ("Max drawdown", max_drawdown, True),
        ("Sharpe", sharpe, False),
        ("# trades", n_trades, False),
        ("Winrate", winrate, True),
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

    sync_cloud = _get_setting(
        args.sync_cloud,
        config_file_values,
        "sync_cloud",
        False,
        lambda v: str(v).lower() == "true",
    )

    bootstrap_data_roots(sync_to_cloud=sync_cloud)

    meta_config: BacktestRunConfig | None = None
    if args.replay_metadata:
        meta_config = load_run_config_from_metadata(args.replay_metadata)

    symbol = _get_setting(args.symbol, config_file_values, "symbol", "NDXm")
    timeframe = _get_setting(args.timeframe, config_file_values, "timeframe", "1m")
    strategy_name = _get_setting(
        args.strategy, config_file_values, "strategy", "microstructure_reversal"
    )

    if meta_config:
        symbol = meta_config.symbol
        timeframe = meta_config.timeframe
        strategy_name = meta_config.strategy_name

    initial_cash = _get_setting(
        args.initial_cash, config_file_values, "initial_cash", 100_000.0, float
    )
    train_years = _get_setting(
        _parse_years(args.train_years), config_file_values, "train_years", None, _parse_years
    )
    test_years = _get_setting(
        _parse_years(args.test_years), config_file_values, "test_years", None, _parse_years
    )
    use_test_years = _get_setting(
        args.use_test_years,
        config_file_values,
        "use_test_years",
        False,
        lambda v: str(v).lower() == "true",
    )

    if meta_config:
        train_years = meta_config.train_years
        test_years = meta_config.test_years
        use_test_years = meta_config.use_test_years

    if meta_config:
        strategy_params = meta_config.strategy_params
    elif strategy_name == "microstructure_sweep":
        sweep_defaults = SweepParams()
        strategy_params = SweepParams(
            ema_short=_get_setting(
                args.ema_short, config_file_values, "ema_short", sweep_defaults.ema_short, int
            ),
            ema_long=_get_setting(
                args.ema_long, config_file_values, "ema_long", sweep_defaults.ema_long, int
            ),
            atr_period=_get_setting(
                args.atr_period, config_file_values, "atr_period", sweep_defaults.atr_period, int
            ),
            atr_timeframe=_get_setting(
                args.atr_timeframe,
                config_file_values,
                "atr_timeframe",
                sweep_defaults.atr_timeframe,
            ),
            atr_timeframe_period=_get_setting(
                args.atr_timeframe_period,
                config_file_values,
                "atr_timeframe_period",
                sweep_defaults.atr_timeframe_period,
                int,
            ),
            sweep_lookback=_get_setting(
                args.sweep_lookback,
                config_file_values,
                "sweep_lookback",
                sweep_defaults.sweep_lookback,
                int,
            ),
            min_sweep_break_atr=_get_setting(
                args.min_sweep_break_atr,
                config_file_values,
                "min_sweep_break_atr",
                sweep_defaults.min_sweep_break_atr,
                float,
            ),
            min_lower_wick_body_ratio=_get_setting(
                args.min_lower_wick_body_ratio,
                config_file_values,
                "min_lower_wick_body_ratio",
                sweep_defaults.min_lower_wick_body_ratio,
                float,
            ),
            min_sweep_range_atr=_get_setting(
                args.min_sweep_range_atr,
                config_file_values,
                "min_sweep_range_atr",
                sweep_defaults.min_sweep_range_atr,
                float,
            ),
            confirm_body_atr=_get_setting(
                args.confirm_body_atr,
                config_file_values,
                "confirm_body_atr",
                sweep_defaults.confirm_body_atr,
                float,
            ),
            confirm_close_above_mid=_get_setting(
                args.confirm_close_above_mid,
                config_file_values,
                "confirm_close_above_mid",
                sweep_defaults.confirm_close_above_mid,
                lambda v: str(v).lower() == "true",
            ),
            volume_period=_get_setting(
                args.volume_period,
                config_file_values,
                "volume_period",
                sweep_defaults.volume_period,
                int,
            ),
            min_rvol=_get_setting(
                args.min_rvol, config_file_values, "min_rvol", sweep_defaults.min_rvol, float
            ),
            vol_percentile_min=_get_setting(
                args.vol_percentile_min,
                config_file_values,
                "vol_percentile_min",
                sweep_defaults.vol_percentile_min,
                float,
            ),
            vol_percentile_max=_get_setting(
                args.vol_percentile_max,
                config_file_values,
                "vol_percentile_max",
                sweep_defaults.vol_percentile_max,
                float,
            ),
            use_trend_filter=_get_setting(
                args.use_trend_filter,
                config_file_values,
                "use_trend_filter",
                sweep_defaults.use_trend_filter,
                lambda v: str(v).lower() == "true",
            ),
            max_atr_mult_intraday=_get_setting(
                args.max_atr_mult_intraday,
                config_file_values,
                "max_atr_mult_intraday",
                sweep_defaults.max_atr_mult_intraday,
                float,
            ),
            max_trades_per_day=_get_setting(
                args.max_trades_per_day,
                config_file_values,
                "max_trades_per_day",
                sweep_defaults.max_trades_per_day,
                int,
            ),
            max_holding_bars=_get_setting(
                args.sweep_max_holding_bars,
                config_file_values,
                "max_holding_bars",
                sweep_defaults.max_holding_bars,
                int,
            ),
            atr_stop_mult=_get_setting(
                args.atr_stop_mult,
                config_file_values,
                "atr_stop_mult",
                sweep_defaults.atr_stop_mult,
                float,
            ),
            rr_multiple=_get_setting(
                args.rr_multiple,
                config_file_values,
                "rr_multiple",
                sweep_defaults.rr_multiple,
                float,
            ),
        )
    elif strategy_name == "opening_sweep_v4":
        opening_defaults = OpeningSweepV4Params()
        strategy_params = OpeningSweepV4Params(
            wick_factor=_get_setting(
                args.wick_factor,
                config_file_values,
                "wick_factor",
                opening_defaults.wick_factor,
                float,
            ),
            atr_percentile=_get_setting(
                args.atr_percentile,
                config_file_values,
                "atr_percentile",
                opening_defaults.atr_percentile,
                float,
            ),
            volume_percentile=_get_setting(
                args.volume_percentile,
                config_file_values,
                "volume_percentile",
                opening_defaults.volume_percentile,
                float,
            ),
            sl_buffer_atr=_get_setting(
                args.sl_buffer_atr,
                config_file_values,
                "sl_buffer_atr",
                opening_defaults.sl_buffer_atr,
                float,
            ),
            sl_buffer_relative=_get_setting(
                args.sl_buffer_relative,
                config_file_values,
                "sl_buffer_relative",
                opening_defaults.sl_buffer_relative,
                float,
            ),
            tp_multiplier=_get_setting(
                args.tp_multiplier,
                config_file_values,
                "tp_multiplier",
                opening_defaults.tp_multiplier,
                float,
            ),
            max_horizon=_get_setting(
                args.max_horizon,
                config_file_values,
                "max_horizon",
                opening_defaults.max_horizon,
                int,
            ),
        )
    else:
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

    if meta_config:
        backtest_config = meta_config.backtest_config
    else:
        backtest_config = BacktestConfig(
            initial_cash=initial_cash,
            trade_size=args.trade_size,
            sl_pct=args.sl_pct,
            tp_pct=args.tp_pct,
            max_bars_in_trade=args.max_bars,
            entry_threshold=args.entry_threshold,
            cost_config_path=args.costs_path,
            cost_instrument=args.cost_instrument or symbol,
        )

    if strategy_name == "opening_sweep_v4":
        backtest_config.max_bars_in_trade = getattr(
            strategy_params, "max_horizon", backtest_config.max_bars_in_trade
        )

    backtest_config.cost_config_path = args.costs_path or backtest_config.cost_config_path
    backtest_config.cost_instrument = (
        args.cost_instrument or backtest_config.cost_instrument or symbol
    )

    seed_value = args.seed if args.seed is not None else (meta_config.seed if meta_config else None)
    snapshot_interval = (
        args.snapshot_interval
        if args.snapshot_interval is not None
        else (meta_config.snapshot_interval if meta_config else None)
    )
    snapshot_path = (
        Path(args.snapshot_path)
        if args.snapshot_path
        else (meta_config.snapshot_path if meta_config else None)
    )
    resume_snapshot = (
        Path(args.resume_snapshot)
        if args.resume_snapshot
        else (meta_config.resume_snapshot if meta_config else None)
    )
    run_metadata_path = Path(args.run_metadata) if args.run_metadata else None
    replay_metadata_path = Path(args.replay_metadata) if args.replay_metadata else None

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
        seed=seed_value,
        snapshot_interval=snapshot_interval,
        snapshot_path=snapshot_path,
        resume_snapshot=resume_snapshot,
        run_metadata_path=run_metadata_path,
        replay_metadata=replay_metadata_path,
        cost_config_path=Path(backtest_config.cost_config_path),
        cost_instrument=backtest_config.cost_instrument,
        strategy_params=strategy_params,
        backtest_config=backtest_config,
    )

    _print_run_context(run_config)

    artifacts = run_single_backtest(run_config)

    print_headline_kpis(artifacts.equity_stats, artifacts.trade_stats)

    equity_keys = [
        "net_start_equity",
        "net_end_equity",
        "net_total_return",
        "net_annualized_return",
        "net_max_drawdown",
        "net_return_drawdown_ratio",
        "net_sharpe_ratio",
        "net_sortino_ratio",
        "net_volatility_annual",
        "net_var_95_monthly",
        "net_n_days",
        "net_n_months",
        "gross_total_return",
        "gross_max_drawdown",
    ]
    trade_keys = [
        "net_n_trades",
        "net_winrate",
        "net_avg_pnl",
        "net_avg_win",
        "net_avg_loss",
        "net_payoff_ratio",
        "net_expectancy_per_trade",
        "net_avg_holding_bars",
        "net_exit_reason_counts",
        "gross_n_trades",
        "gross_winrate",
        "gross_avg_pnl",
    ]

    print_metrics("Métricas de equity (tipo Darwinex)", artifacts.equity_stats, equity_keys)
    print_metrics("Métricas de trades", artifacts.trade_stats, trade_keys)
    print_level_metrics(artifacts.trade_level_stats)
    print("\n=== Tail de la curva de equity ===")
    print(artifacts.equity_series.tail().to_frame("equity"))
    print("\n=== Primeros trades ===")
    print(artifacts.trades_df.head())
    if not artifacts.trades_df.empty:
        print("\n=== Últimos trades ===")
        print(artifacts.trades_df.tail())

    dashboard_path = None
    if not artifacts.trades_df.empty:
        reports_dir = (
            artifacts.reports.excel_path.parent
            if artifacts.reports.excel_path is not None
            else (run_config.reports_dir or (REPORTS_DIR / run_config.symbol))
        )
        reports_dir.mkdir(parents=True, exist_ok=True)
        strategy_suffix = _strategy_suffix(run_config.strategy_name)
        dashboard_path = reports_dir / f"trades_dashboard_{strategy_suffix}.html"
        volatility_col = "volatility" if "volatility" in artifacts.trades_df.columns else None
        build_trades_dashboard(
            artifacts.trades_df,
            dashboard_path,
            volatility_col=volatility_col,
        )

    if run_config.generate_report_files:
        print("\n=== Ficheros de resumen generados ===")
        print(f"Excel: {_describe_artifact(artifacts.reports.excel_path)}")
        print(f"JSON:  {_describe_artifact(artifacts.reports.json_path)}")
        print(f"Levels stats: {_describe_artifact(artifacts.reports.levels_report_path)}")
    if run_config.generate_main_plots:
        print(f"Plot equity/trades: {_describe_artifact(artifacts.reports.equity_path)}")
    if run_config.generate_trade_plots:
        print(f"Plot mejores trades: {_describe_artifact(artifacts.reports.best_trade_plot_path)}")
        print(f"Plot peores trades: {_describe_artifact(artifacts.reports.worst_trade_plot_path)}")
    if dashboard_path:
        print(f"Dashboard trades: {_describe_artifact(dashboard_path)}")

    print_timings(artifacts.timings)


if __name__ == "__main__":
    main()
