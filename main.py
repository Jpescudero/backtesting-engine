# main.py
"""
Punto de entrada para lanzar un backtest sencillo y generar
outputs fáciles de analizar/compartir.

Incluye:
- Estadísticas de equity y de trades.
- (Opcional) Ficheros Excel + JSON (modo ultraligero).
- (Opcional) Figura con las mejores y peores operaciones (3+3).
- (Opcional) Gráficas equity + nº trades/mes.

Además, mide y muestra tiempos de ejecución por fase para
identificar cuellos de botella.
"""

from __future__ import annotations

from pathlib import Path
import time

import matplotlib.pyplot as plt

from src.config.paths import (
    ensure_directories_exist,
    PARQUET_TICKS_DIR,
    OTHER_DATA_DIR,
    NPZ_DIR,
    REPORTS_DIR,
)
from src.data.data_utils import list_tick_files
from src.data.data_to_parquet import data_to_parquet, DEFAULT_SYMBOL
from src.data.bars1m_to_excel import (
    generate_1m_bars_csv,
    get_default_output_csv,
)
from src.data.csv_1m_to_npz import csv_1m_to_npz
from src.data.feeds import NPZOHLCVFeed

from src.engine.core import BacktestConfig, run_backtest_with_signals

from src.strategies.barrida_apertura import StrategyBarridaApertura

from src.analytics.reporting import equity_to_series, trades_to_dataframe
from src.analytics.plots import plot_equity_curve, plot_trades_per_month
from src.analytics.metrics import equity_curve_metrics, trades_metrics
from src.analytics.backtest_output import save_backtest_summary_to_excel
from src.analytics.trade_plots import plot_best_and_worst_trades


# ============================================================
# Flags de comportamiento
# ============================================================

# Si estás iterando en la lógica de la estrategia o del motor,
# puedes poner esto a False para NO generar Excel/JSON
GENERATE_REPORT_FILES = True

# Si sólo quieres ver métricas por consola y nada de gráficos:
GENERATE_MAIN_PLOTS = True      # equity + trades/mes
GENERATE_TRADE_PLOTS = True     # 3 mejores + 3 peores trades


# ============================================================
# Helpers de preparación de datos
# ============================================================


def ensure_ticks_and_csv(symbol: str = DEFAULT_SYMBOL) -> Path:
    """
    Se asegura de que existan:
      - Parquets de ticks para el símbolo.
      - CSV de barras de 1 minuto para el símbolo.
    """
    ensure_directories_exist()

    try:
        tick_files = list_tick_files(PARQUET_TICKS_DIR, symbol=symbol)
    except FileNotFoundError:
        tick_files = []

    if not tick_files:
        print(f"[ensure_ticks_and_csv] No hay parquet de ticks para {symbol}. Generando desde Darwinex...")
        data_to_parquet(symbol=symbol)

        tick_files = list_tick_files(PARQUET_TICKS_DIR, symbol=symbol)
        if not tick_files:
            raise RuntimeError(f"No se han podido generar parquets de ticks para {symbol}")

    bars_csv = get_default_output_csv(symbol=symbol)
    if not bars_csv.exists():
        print(f"[ensure_ticks_and_csv] No existe CSV de barras 1m para {symbol}. Generando...")
        bars_csv = generate_1m_bars_csv(symbol=symbol)

    if not bars_csv.exists():
        raise RuntimeError(f"No se ha podido generar el CSV de barras 1m para {symbol}")

    print(f"[ensure_ticks_and_csv] CSV de barras 1m listo: {bars_csv}")
    return bars_csv


def ensure_npz_from_csv(symbol: str = DEFAULT_SYMBOL, timeframe: str = "1m") -> Path:
    """
    Se asegura de que exista al menos un fichero .npz para el símbolo/timeframe.
    """
    ensure_directories_exist()

    symbol_npz_dir = (NPZ_DIR / symbol).resolve()
    symbol_npz_dir.mkdir(parents=True, exist_ok=True)

    pattern = f"*_{timeframe}.npz" if not timeframe.startswith("_") else f"*{timeframe}.npz"
    existing_npz = list(symbol_npz_dir.glob(pattern))

    if existing_npz:
        print(f"[ensure_npz_from_csv] Encontrados {len(existing_npz)} NPZ para {symbol} ({timeframe}).")
        return existing_npz[0]

    print(f"[ensure_npz_from_csv] No hay NPZ para {symbol} ({timeframe}). Creando desde CSV 1m...")
    bars_csv = ensure_ticks_and_csv(symbol=symbol)

    npz_path = csv_1m_to_npz(symbol=symbol, csv_path=bars_csv)

    return npz_path


# ============================================================
# Backtest + profiling de tiempos
# ============================================================


def run_single_backtest(symbol: str = "NDXm") -> None:
    """
    Ejecuta un único backtest sobre el símbolo indicado usando:
      - Datos 1m (NPZ)
      - Estrategia de barrida en aperturas
      - Motor run_backtest_with_signals (Numba)

    Además imprime tiempos por cada fase importante.
    """
    timings = {}

    t0 = time.perf_counter()

    # 1) Preparación de datos (ticks -> csv -> npz)
    t_data_start = time.perf_counter()
    bars_csv_path = ensure_ticks_and_csv(symbol=symbol)
    npz_path = ensure_npz_from_csv(symbol=symbol, timeframe="1m")
    t_data_end = time.perf_counter()
    timings["01_datos_preparacion"] = t_data_end - t_data_start

    print(f"[run_single_backtest] CSV de barras 1m listo: {bars_csv_path}")
    print(f"[run_single_backtest] NPZ listo: {npz_path}")

    # 2) Carga de feed NPZ
    t_feed_start = time.perf_counter()
    feed = NPZOHLCVFeed(symbol=symbol, timeframe="1m")
    data = feed.load_all()
    t_feed_end = time.perf_counter()
    timings["02_carga_feed_npz"] = t_feed_end - t_feed_start

    # 3) Generación de señales de la estrategia
    t_strat_start = time.perf_counter()
    strategy = StrategyBarridaApertura(
        volume_percentile=80.0,
        use_two_bearish_bars=True,
    )
    strat_res = strategy.generate_signals(data)
    t_strat_end = time.perf_counter()
    timings["03_generar_senales_estrategia"] = t_strat_end - t_strat_start

    n_signals = int((strat_res.signals != 0).sum())
    print(f"[run_single_backtest] Estrategia Barrida: {n_signals} señales generadas")
    print(f"[run_single_backtest] Meta estrategia: {strat_res.meta}")

    # 4) Backtest (motor numba)
    t_bt_start = time.perf_counter()
    config = BacktestConfig(
        initial_cash=100_000.0,
        commission_per_trade=1.0,
        trade_size=1.0,
        slippage=0.0,
        sl_pct=0.01,
        tp_pct=0.02,
        max_bars_in_trade=60,
        entry_threshold=0.0,
    )

    result = run_backtest_with_signals(data, strat_res.signals, config=config)
    t_bt_end = time.perf_counter()
    timings["04_backtest_motor"] = t_bt_end - t_bt_start

    print("\n=== Resumen del backtest ===")
    print("Cash final:       ", result.cash)
    print("Posición final:   ", result.position)
    print("Número de trades: ", result.extra.get("n_trades", 0))

    # 5) Conversión a pandas (equity/trades)
    t_pandas_start = time.perf_counter()
    eq_series = equity_to_series(result, data)
    trades_df = trades_to_dataframe(result, data)
    t_pandas_end = time.perf_counter()
    timings["05_conversion_pandas"] = t_pandas_end - t_pandas_start

    # 6) Cálculo de métricas
    t_metrics_start = time.perf_counter()
    eq_stats = equity_curve_metrics(eq_series)
    tr_stats = trades_metrics(trades_df)
    t_metrics_end = time.perf_counter()
    timings["06_metricas"] = t_metrics_end - t_metrics_start

    print("\n=== Métricas de equity (tipo Darwinex) ===")
    for k, v in eq_stats.items():
        print(f"{k:25s}: {v}")

    print("\n=== Métricas de trades ===")
    for k, v in tr_stats.items():
        print(f"{k:25s}: {v}")

    print("\n=== Tail de la curva de equity ===")
    print(eq_series.tail())

    print("\n=== Primeros trades ===")
    print(trades_df.head())

    reports_dir = (REPORTS_DIR / symbol).resolve()

    # 7) Generación de Excel + JSON (opcional, ultraligero)
    if GENERATE_REPORT_FILES:
        t_reports_start = time.perf_counter()
        excel_path, json_path = save_backtest_summary_to_excel(
            base_dir=reports_dir,
            filename=f"backtest_{symbol}_barrida_apertura.xlsx",
            symbol=symbol,
            strategy_name="barrida_apertura",
            equity_series=eq_series,
            trades_df=trades_df,
            equity_stats=eq_stats,
            trade_stats=tr_stats,
            meta=getattr(strat_res, "meta", {}),
        )
        t_reports_end = time.perf_counter()
        timings["07_reportes_excel_json"] = t_reports_end - t_reports_start

        print("\n=== Ficheros de resumen generados ===")
        print(f"Excel: {excel_path}")
        print(f"JSON:  {json_path}")
    else:
        print("\n[run_single_backtest] GENERATE_REPORT_FILES=False, se omite Excel/JSON.")
        timings["07_reportes_excel_json"] = 0.0

    # 8) Gráficas básicas (equity + nº de trades por mes)
    if GENERATE_MAIN_PLOTS:
        t_plots_start = time.perf_counter()
        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=False)

        plot_equity_curve(result, data, ax=ax1)
        plot_trades_per_month(result, data, ax=ax2)

        plt.tight_layout()
        plt.show()
        t_plots_end = time.perf_counter()
        timings["08_plots_equity_trades_mes"] = t_plots_end - t_plots_start
    else:
        print("\n[run_single_backtest] GENERATE_MAIN_PLOTS=False, se omiten plots equity/trades_mes.")
        timings["08_plots_equity_trades_mes"] = 0.0

    # 9) Figura de mejores/peores trades (opcional)
    if GENERATE_TRADE_PLOTS:
        t_tradeplots_start = time.perf_counter()
        print("\n[run_single_backtest] Generando figura de mejores/peores trades...")
        fig_trades = plot_best_and_worst_trades(
            trades_df=trades_df,
            data=data,
            n_best=3,
            n_worst=3,
            pnl_col="pnl",
            entry_col="entry_idx",
            exit_col="exit_idx",
            direction_col="direction",
            window=30,
            figsize=(14, 10),
            save_path=reports_dir / "best_worst_trades.png",
        )
        plt.show()
        t_tradeplots_end = time.perf_counter()
        timings["09_plots_mejores_peores_trades"] = t_tradeplots_end - t_tradeplots_start

        print(
            "[run_single_backtest] Figura de mejores/peores trades guardada en:",
            reports_dir / "best_worst_trades.png",
        )
    else:
        print("\n[run_single_backtest] GENERATE_TRADE_PLOTS=False, se omite figura de mejores/peores trades.")
        timings["09_plots_mejores_peores_trades"] = 0.0

    t1 = time.perf_counter()
    timings["00_total"] = t1 - t0

    # 10) Resumen final de tiempos
    print("\n================ Tiempos de ejecución (segundos) ================")
    for key in sorted(timings.keys()):
        print(f"{key:35s}: {timings[key]:8.3f}")
    print("=================================================================")


def main(symbol: str = "NDXm") -> None:
    run_single_backtest(symbol=symbol)


if __name__ == "__main__":
    main(symbol="NDXm")
