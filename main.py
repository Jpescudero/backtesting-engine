# main.py

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from src.config.paths import (
    ensure_directories_exist,
    PARQUET_TICKS_DIR,
    OTHER_DATA_DIR,  # ahora mismo no lo usamos directamente, pero lo mantenemos por claridad
    NPZ_DIR,
)
from src.data.data_utils import list_tick_files
from src.data.data_to_parquet import data_to_parquet, DEFAULT_SYMBOL
from src.data.bars1m_to_excel import (
    generate_1m_bars_csv,
    get_default_output_csv,
)
from src.data.csv_1m_to_npz import csv_1m_to_npz
from src.data.feeds import NPZOHLCVFeed

# IMPORTANTE: ahora usamos el motor basado en señales externas
from src.engine.core import BacktestConfig, run_backtest_with_signals

# Estrategia de barrida en aperturas
from src.strategies.barrida_apertura import StrategyBarridaApertura

from src.analytics.reporting import equity_to_series, trades_to_dataframe
from src.analytics.plots import plot_equity_curve, plot_trades_per_month
from src.analytics.metrics import equity_curve_metrics, trades_metrics


# ============================================================
# Helpers de preparación de datos
# ============================================================

def ensure_ticks_and_csv(symbol: str = DEFAULT_SYMBOL) -> Path:
    """
    Se asegura de que existan:
      - Parquets de ticks para el símbolo.
      - CSV de barras de 1 minuto para el símbolo.

    Si no existen, los genera a partir de los datos de Darwinex.

    Devuelve:
        Path al CSV de barras de 1 minuto.
    """
    ensure_directories_exist()

    # 1) Parquets de ticks
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

    # 2) CSV de barras de 1 minuto
    bars_csv = get_default_output_csv(symbol=symbol)
    if not bars_csv.exists():
        print(f"[ensure_ticks_and_csv] No existe CSV de barras 1m para {symbol}. Generando...")
        bars_csv = generate_1m_bars_csv(symbol=symbol)

    return bars_csv


def ensure_npz_from_csv(symbol: str = DEFAULT_SYMBOL, timeframe: str = "1m") -> Path:
    """
    Se asegura de que exista al menos un fichero .npz para el símbolo/timeframe.

    Si no existe, lo crea a partir del CSV de barras de 1 minuto.

    Devuelve:
        Path al fichero .npz principal.
    """
    ensure_directories_exist()

    symbol_npz_dir = (NPZ_DIR / symbol).resolve()
    symbol_npz_dir.mkdir(parents=True, exist_ok=True)

    pattern = f"*_{timeframe}.npz"
    existing_npz = list(symbol_npz_dir.glob(pattern))

    if existing_npz:
        print(f"[ensure_npz_from_csv] Encontrados {len(existing_npz)} NPZ para {symbol} ({timeframe}).")
        # Devolvemos el primero (puedes cambiar la lógica si quieres algo más sofisticado)
        return existing_npz[0]

    # Si no hay NPZ, generamos CSV y luego NPZ
    print(f"[ensure_npz_from_csv] No hay NPZ para {symbol} ({timeframe}). Creando desde CSV 1m...")
    bars_csv = ensure_ticks_and_csv(symbol=symbol)

    npz_path = csv_1m_to_npz(
        csv_path=bars_csv,
        symbol=symbol,
        timeframe=timeframe,
        out_dir=symbol_npz_dir,
    )

    return npz_path


# ============================================================
# Backtest "single run" usando estrategia de barrida en apertura
# ============================================================

def run_single_backtest(symbol: str = "NDXm") -> None:
    """
    Ejecuta un único backtest sobre el símbolo indicado usando:

      - Datos 1m (NPZ)
      - Estrategia de barrida en aperturas (StrategyBarridaApertura)
      - Motor run_backtest_with_signals (Numba)
    """
    # 1) Asegurar que CSV y NPZ existen
    bars_csv_path = ensure_ticks_and_csv(symbol=symbol)
    npz_path = ensure_npz_from_csv(symbol=symbol, timeframe="1m")

    print(f"[run_single_backtest] CSV de barras 1m listo: {bars_csv_path}")
    print(f"[run_single_backtest] NPZ listo: {npz_path}")

    # 2) Cargar datos desde NPZ
    feed = NPZOHLCVFeed(symbol=symbol, timeframe="1m")
    data = feed.load_all()

    # 3) Definir estrategia de barrida en apertura
    #    Puedes jugar con estos parámetros:
    #       - volume_percentile: umbral de volumen alto
    #       - use_two_bearish_bars: exigir 2 velas bajistas consecutivas
    strategy = StrategyBarridaApertura(
        volume_percentile=80.0,
        use_two_bearish_bars=True,
    )

    strat_res = strategy.generate_signals(data)

    n_signals = int((strat_res.signals != 0).sum())
    print(f"[run_single_backtest] Estrategia Barrida: {n_signals} señales generadas")
    print(f"[run_single_backtest] Meta estrategia: {strat_res.meta}")

    # 4) Configurar y ejecutar backtest
    config = BacktestConfig(
        initial_cash=100_000.0,
        commission_per_trade=1.0,
        trade_size=1.0,
        slippage=0.0,
        sl_pct=0.01,             # 1% SL
        tp_pct=0.02,             # 2% TP
        max_bars_in_trade=60,    # máx 60 minutos en la operación (1 barra = 1 min)
        entry_threshold=0.0,     # ya no se usa en esta estrategia, pero lo dejamos por compatibilidad
    )

    # Usamos el motor basado en señales externas
    result = run_backtest_with_signals(data, strat_res.signals, config=config)

    print("\n=== Resumen del backtest ===")
    print("Cash final:        ", result.cash)
    print("Posición final:    ", result.position)
    print("Número de trades:  ", result.extra.get("n_trades", 0))

    # 5) Conversión a pandas
    eq_series = equity_to_series(result, data)
    trades_df = trades_to_dataframe(result, data)

    # 6) Métricas de equity (tipo Darwinex)
    eq_stats = equity_curve_metrics(eq_series)
    print("\n=== Métricas de equity (tipo Darwinex) ===")
    for k, v in eq_stats.items():
        print(f"{k:25s}: {v}")

    # 7) Métricas de trades
    tr_stats = trades_metrics(trades_df)
    print("\n=== Métricas de trades ===")
    for k, v in tr_stats.items():
        print(f"{k:25s}: {v}")

    # 8) Mostrar un pequeño resumen de series/tablas (últimos valores)
    print("\n=== Tail de la curva de equity ===")
    print(eq_series.tail())

    print("\n=== Primeros trades ===")
    print(trades_df.head())

    # 9) Gráficas: equity + nº de trades por mes (igual que antes)
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=False)

    plot_equity_curve(result, data, ax=ax1)
    plot_trades_per_month(result, data, ax=ax2)

    plt.tight_layout()
    plt.show()


def main(symbol: str = "NDXm") -> None:
    """
    Punto de entrada principal del script.
    Por ahora ejecuta un único backtest sobre el símbolo indicado,
    usando la estrategia de barrida en aperturas.
    """
    run_single_backtest(symbol=symbol)


if __name__ == "__main__":
    main(symbol="NDXm")
