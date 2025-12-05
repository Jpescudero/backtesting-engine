# main.py
from __future__ import annotations

from pathlib import Path

from src.config.paths import (
    ensure_directories_exist,
    PARQUET_TICKS_DIR,
    OTHER_DATA_DIR,
)
from src.data.data_utils import list_tick_files
from src.data.data_to_parquet import data_to_parquet, DEFAULT_SYMBOL
from src.data.bars1m_to_excel import (
    generate_1m_bars_csv,
    get_default_output_csv,
)

from src.data.feeds import NPZOHLCVFeed
from src.engine.core import BacktestConfig, run_backtest_basic
from src.analytics.reporting import equity_to_series, trades_to_dataframe
from src.analytics.plots import plot_equity_curve, plot_trades_per_month
from src.analytics.metrics import equity_curve_metrics, trades_metrics


from src.data.csv_1m_to_npz import csv_1m_to_npz
from src.config.paths import NPZ_DIR



import matplotlib.pyplot as plt


def ensure_data_ready(symbol: str = DEFAULT_SYMBOL) -> Path:
    """
    Se asegura de que existan:
      - Parquets de ticks para el símbolo.
      - CSV de barras de 1m para el símbolo.

    Si no existen, los genera.

    Devuelve la ruta del CSV de barras 1m.
    """
    ensure_directories_exist()

    # 1) Comprobar parquets de ticks
    try:
        tick_files = list_tick_files(PARQUET_TICKS_DIR, symbol=symbol)
    except FileNotFoundError:
        tick_files = []

    if not tick_files:
        print(f"No hay parquet de ticks para {symbol}. Generando desde Darwinex...")
        data_to_parquet(symbol=symbol)
        # Volvemos a mirar
        tick_files = list_tick_files(PARQUET_TICKS_DIR, symbol=symbol)
        if not tick_files:
            raise RuntimeError(f"No se han podido generar parquets de ticks para {symbol}")

    # 2) Comprobar CSV de barras 1m
    bars_csv = get_default_output_csv(symbol=symbol)
    if not bars_csv.exists():
        print(f"No existe CSV de barras 1m para {symbol}. Generando...")
        bars_csv = generate_1m_bars_csv(symbol=symbol)

    return bars_csv


def ensure_ticks_and_csv(symbol: str = DEFAULT_SYMBOL) -> Path:
    """
    Se asegura de que existan:
      - Parquets de ticks
      - CSV de barras 1m
    Devuelve la ruta del CSV.
    """
    ensure_directories_exist()

    # 1) Parquets de ticks
    try:
        tick_files = list_tick_files(PARQUET_TICKS_DIR, symbol=symbol)
    except FileNotFoundError:
        tick_files = []

    if not tick_files:
        print(f"No hay parquet de ticks para {symbol}. Generando desde Darwinex...")
        data_to_parquet(symbol=symbol)
        tick_files = list_tick_files(PARQUET_TICKS_DIR, symbol=symbol)
        if not tick_files:
            raise RuntimeError(f"No se han podido generar parquets de ticks para {symbol}")

    # 2) CSV de barras 1m
    bars_csv = get_default_output_csv(symbol=symbol)
    if not bars_csv.exists():
        print(f"No existe CSV de barras 1m para {symbol}. Generando...")
        bars_csv = generate_1m_bars_csv(symbol=symbol)

    return bars_csv

def ensure_npz_from_csv(symbol: str = DEFAULT_SYMBOL, timeframe: str = "1m") -> Path:
    """
    Se asegura de que exista al menos un .npz para el símbolo/timeframe.
    Si no existe, lo crea a partir del CSV de barras 1m.
    Devuelve la ruta del npz principal.
    """
    ensure_directories_exist()

    symbol_npz_dir = (NPZ_DIR / symbol).resolve()
    symbol_npz_dir.mkdir(parents=True, exist_ok=True)

    pattern = f"*_{timeframe}.npz"
    existing_npz = list(symbol_npz_dir.glob(pattern))

    if existing_npz:
        # Ya hay alguno, usamos el primero
        print(f"Encontrados {len(existing_npz)} NPZ para {symbol} ({timeframe}).")
        return existing_npz[0]

    print(f"No hay NPZ para {symbol} ({timeframe}). Creando desde CSV 1m...")

    bars_csv = get_default_output_csv(symbol=symbol)
    if not bars_csv.exists():
        raise FileNotFoundError(
            f"CSV de barras 1m no encontrado en {bars_csv}. "
            f"Asegúrate de haber llamado antes a ensure_ticks_and_csv()."
        )

    npz_path = csv_1m_to_npz(symbol=symbol, csv_path=bars_csv)
    print(f"NPZ creado en: {npz_path}")
    return npz_path


def main(symbol: str = "NDXm") -> None:
    # 1) Asegurar datos (ticks, CSV, NPZ) – esto ya lo tienes
    bars_csv_path = ensure_ticks_and_csv(symbol=symbol)
    npz_path = ensure_npz_from_csv(symbol=symbol, timeframe="1m")

    # 2) Cargar datos
    feed = NPZOHLCVFeed(symbol=symbol, timeframe="1m")
    data = feed.load_all()

    # 3) Configurar y ejecutar backtest mejorado
    config = BacktestConfig(
        initial_cash=100_000.0,
        commission_per_trade=1.0,
        trade_size=1.0,
        slippage=0.0,
        sl_pct=0.01,
        tp_pct=0.02,
        max_bars_in_trade=60,
        entry_threshold=0.001,
    )

    result = run_backtest_basic(data, config=config)

    print("Backtest terminado.")
    print("Cash final:", result.cash)
    print("Posición final:", result.position)
    print("Número de trades:", result.extra["n_trades"])

    # --- Conversión a pandas ---
    eq_series = equity_to_series(result, data)
    trades_df = trades_to_dataframe(result, data)

    # --- Métricas de equity (tipo Darwinex) ---
    eq_stats = equity_curve_metrics(eq_series)
    print("\n=== Métricas de equity (tipo Darwinex) ===")
    for k, v in eq_stats.items():
        print(f"{k:25s}: {v}")

    # --- Métricas de trades ---
    tr_stats = trades_metrics(trades_df)
    print("\n=== Métricas de trades ===")
    for k, v in tr_stats.items():
        print(f"{k:25s}: {v}")

    # 4) Ver equity y trades en tablas (opcional)
    eq_series = equity_to_series(result, data)
    trades_df = trades_to_dataframe(result, data)
    print(eq_series.tail())
    print(trades_df.head())

    # 5) Gráficas: equity + nº de trades por mes
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=False)

    plot_equity_curve(result, data, ax=ax1)
    plot_trades_per_month(result, data, ax=ax2)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main(symbol="NDXm")
