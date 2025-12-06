# main.py

from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from src.config.paths import (
    ensure_directories_exist,
    PARQUET_TICKS_DIR,
    OTHER_DATA_DIR,  # no lo usamos ahora mismo, pero lo mantenemos
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

# Motor de backtesting (Numba, con señales externas)
from src.engine.core import BacktestConfig, run_backtest_with_signals

# Estrategia de barrida en aperturas
from src.strategies.barrida_apertura import StrategyBarridaApertura

# Analytics / reporting
from src.analytics.reporting import equity_to_series, trades_to_dataframe
from src.analytics.plots import plot_equity_curve, plot_trades_per_month
from src.analytics.metrics import equity_curve_metrics, trades_metrics
from src.analytics.backtest_output import save_backtest_summary_to_excel


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
# Helpers de métricas simples (ATR y tamaño de posición)
# ============================================================

def estimate_atr_pct_from_data(data, period: int = 14) -> float:
    """
    Estima el ATR medio (en % del precio) a partir de un objeto OHLCVArrays.

    Devolvemos la mediana de (ATR / close) para que sea robusto a outliers.
    """
    c = data.c.astype(float)
    h = data.h.astype(float)
    l = data.l.astype(float)

    n = c.shape[0]
    if n < 2:
        return 0.005  # fallback 0.5%

    # True Range clásico
    tr1 = h[1:] - l[1:]
    prev_close = c[:-1]
    tr2 = np.abs(h[1:] - prev_close)
    tr3 = np.abs(l[1:] - prev_close)
    tr = np.maximum(tr1, np.maximum(tr2, tr3))

    if tr.shape[0] < period:
        atr = tr
    else:
        # media móvil simple del TR
        kernel = np.ones(period, dtype=float) / float(period)
        atr = np.convolve(tr, kernel, mode="valid")

    c_tail = c[-len(atr):]
    atr_pct = atr / c_tail
    atr_pct = atr_pct[np.isfinite(atr_pct) & (atr_pct > 0)]

    if atr_pct.size == 0:
        return 0.005

    return float(np.median(atr_pct))


def estimate_trade_size_from_data(
    data,
    initial_cash: float,
    notional_fraction: float = 0.5,
    min_size: float = 0.01,
) -> float:
    """
    Estima un tamaño de posición fraccional para que con `initial_cash`
    puedas abrir operaciones aunque el subyacente tenga un precio alto.

    - notional_fraction: fracción del capital que quieres usar por operación.
    - min_size: tamaño mínimo permitido (para evitar size=0).
    """
    c = data.c.astype(float)
    valid = c[np.isfinite(c) & (c > 0)]
    if valid.size == 0:
        return min_size

    median_price = float(np.median(valid))
    # tamaño = (cash * fracción) / precio_medio
    size = (initial_cash * notional_fraction) / median_price
    if size < min_size:
        size = min_size
    return float(size)


# ============================================================
# Plot local de 3 mejores y 3 peores trades
# ============================================================

def plot_best_and_worst_trades_local(
    result,
    data,
    n_best: int = 3,
    n_worst: int = 3,
    window_bars: int = 60,
    save_path: Path | None = None,
) -> None:
    """
    Plotea las n_best mejores y n_worst peores operaciones usando result.trade_log
    y los datos OHLCV de 'data'. Cada subplot muestra el close con la entrada/salida.
    """
    trade_log = result.trade_log
    if not trade_log or "pnl" not in trade_log:
        print("[plot_best_and_worst_trades_local] No hay trade_log o falta pnl, no se plotea.")
        return

    pnl = np.asarray(trade_log["pnl"], dtype=float)
    entry_idx = np.asarray(trade_log["entry_idx"], dtype=int)
    exit_idx = np.asarray(trade_log["exit_idx"], dtype=int)

    n_trades = pnl.shape[0]
    if n_trades == 0:
        print("[plot_best_and_worst_trades_local] No hay trades.")
        return

    # Índices ordenados por pnl
    best_order = np.argsort(-pnl)  # descendente
    worst_order = np.argsort(pnl)  # ascendente

    best_ids = best_order[: min(n_best, n_trades)]
    worst_ids = worst_order[: min(n_worst, n_trades)]

    # Datos de mercado
    ts = data.ts
    c = data.c.astype(float)

    total_plots = len(best_ids) + len(worst_ids)
    if total_plots == 0:
        print("[plot_best_and_worst_trades_local] No hay suficientes trades para ploteo.")
        return

    # Configuramos figura 3x2 (hasta 6 trades)
    fig, axes = plt.subplots(3, 2, figsize=(14, 8), sharex=False)
    axes = axes.flatten()

    plot_idx = 0

    # Helper interno
    def _plot_single(ax, trade_id: int, kind: str) -> None:
        ei = int(entry_idx[trade_id])
        xi = int(exit_idx[trade_id])
        pnl_val = float(pnl[trade_id])

        start = max(0, ei - window_bars)
        end = min(len(c) - 1, xi + window_bars)

        ts_slice = ts[start : end + 1]
        c_slice = c[start : end + 1]

        ax.plot(ts_slice, c_slice, linewidth=1.0)

        ax.scatter(ts[ei], c[ei], marker="^", s=40, label="Entry")
        ax.scatter(ts[xi], c[xi], marker="v", s=40, label="Exit")

        ax.set_title(
            f"{kind.upper()} #{trade_id} | {ts[ei]} | PnL={pnl_val:.2f}",
            fontsize=9,
        )
        ax.set_ylabel("Precio")
        ax.legend(loc="best", fontsize=7)

    # Best trades
    for tid in best_ids:
        if plot_idx >= len(axes):
            break
        _plot_single(axes[plot_idx], int(tid), kind="BEST")
        plot_idx += 1

    # Worst trades
    for tid in worst_ids:
        if plot_idx >= len(axes):
            break
        _plot_single(axes[plot_idx], int(tid), kind="WORST")
        plot_idx += 1

    # Si sobran subplots, los ocultamos
    for j in range(plot_idx, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)
        print(f"[plot_best_and_worst_trades_local] Imagen guardada en: {save_path}")


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
    # 0) Directorio de reports
    reports_dir = (REPORTS_DIR / symbol).resolve()
    reports_dir.mkdir(parents=True, exist_ok=True)

    # 1) Asegurar que CSV y NPZ existen
    bars_csv_path = ensure_ticks_and_csv(symbol=symbol)
    npz_path = ensure_npz_from_csv(symbol=symbol, timeframe="1m")

    print(f"[run_single_backtest] CSV de barras 1m listo: {bars_csv_path}")
    print(f"[run_single_backtest] NPZ listo: {npz_path}")

    # 2) Cargar datos desde NPZ
    feed = NPZOHLCVFeed(symbol=symbol, timeframe="1m")
    data = feed.load_all()

    # 3) Definir estrategia de barrida en apertura (más filtrada)
    strategy = StrategyBarridaApertura(
        volume_percentile=99.0,          # sólo el 1% de mayor volumen
        use_two_bearish_bars=True,       # dos velas bajistas seguidas
        pre_open_minutes=5,              # 5 minutos antes de la apertura
        post_open_minutes=20,            # 20 minutos después
        confirm_reversal=True,
        min_reversal_strength_atr=1.0,   # reversal mínimo 1 ATR
    )

    strat_res = strategy.generate_signals(data)

    n_signals = int((strat_res.signals != 0).sum())
    print(f"[run_single_backtest] Estrategia Barrida: {n_signals} señales generadas")
    print(f"[run_single_backtest] Meta estrategia: {getattr(strat_res, 'meta', {})}")

    # 4) Configurar SL/TP dinámicos basados en ATR
    atr_pct = estimate_atr_pct_from_data(data, period=14)
    atr_sl_mult = 1.0   # 1x ATR de SL
    atr_tp_mult = 2.0   # 2x ATR de TP

    sl_pct = atr_sl_mult * atr_pct
    tp_pct = atr_tp_mult * atr_pct

    # Acotamos a rangos razonables (por seguridad)
    sl_pct = float(np.clip(sl_pct, 0.0002, 0.01))  # entre 0.02% y 1%
    tp_pct = float(np.clip(tp_pct, 0.0004, 0.02))  # entre 0.04% y 2%

    print(f"[run_single_backtest] ATR_pct estimado: {atr_pct:.6f}")
    print(f"[run_single_backtest] SL_pct usado:   {sl_pct:.6f}")
    print(f"[run_single_backtest] TP_pct usado:   {tp_pct:.6f}")

    # 5) Capital inicial y tamaño de posición
    initial_cash = 2000.0
    trade_size = estimate_trade_size_from_data(
        data,
        initial_cash=initial_cash,
        notional_fraction=0.5,  # ~50% del capital por operación
        min_size=0.01,
    )
    print(f"[run_single_backtest] Tamaño de posición estimado: {trade_size:.4f} unidades")

    config = BacktestConfig(
        initial_cash=initial_cash,
        commission_per_trade=1.0,
        trade_size=trade_size,
        slippage=0.0,
        sl_pct=sl_pct,
        tp_pct=tp_pct,
        max_bars_in_trade=60,   # hasta 60 minutos en mercado
        entry_threshold=0.0,    # no se usa en esta estrategia
    )

    # 6) Usamos el motor basado en señales externas
    result = run_backtest_with_signals(data, strat_res.signals, config=config)

    print("\n=== Resumen del backtest ===")
    print("Cash inicial:      ", config.initial_cash)
    print("Cash final:        ", result.cash)
    print("Posición final:    ", result.position)
    print("Número de trades:  ", result.extra.get("n_trades", 0))

    # 7) Conversión a pandas (usando los helpers del módulo reporting)
    eq_series = equity_to_series(result, data)
    trades_df = trades_to_dataframe(result, data)

    # 8) Métricas de equity (tipo Darwinex)
    eq_stats = equity_curve_metrics(eq_series)
    print("\n=== Métricas de equity (tipo Darwinex) ===")
    for k, v in eq_stats.items():
        print(f"{k:25s}: {v}")

    # 9) Métricas de trades
    tr_stats = trades_metrics(trades_df)
    print("\n=== Métricas de trades ===")
    for k, v in tr_stats.items():
        print(f"{k:25s}: {v}")

    # 10) Mostrar un pequeño resumen de series/tablas (últimos valores)
    print("\n=== Tail de la curva de equity ===")
    print(eq_series.tail())

    print("\n=== Primeros trades ===")
    print(trades_df.head())

    # 11) Guardar resumen a Excel + JSON
    excel_path, json_path = save_backtest_summary_to_excel(
        base_dir=reports_dir,
        filename=f"backtest_{symbol}_barrida_apertura",
        symbol=symbol,
        strategy_name="barrida_apertura_long_only",
        equity_series=eq_series,
        trades_df=trades_df,
        equity_stats=eq_stats,
        trade_stats=tr_stats,
        meta=getattr(strat_res, "meta", {}),
    )

    print(f"\n[run_single_backtest] Reporte Excel guardado en: {excel_path}")
    print(f"[run_single_backtest] Resumen JSON guardado en: {json_path}")

    # 12) Gráficas: equity + nº de trades por mes
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=False, figsize=(10, 7))

    plot_equity_curve(result, data, ax=ax1)
    plot_trades_per_month(result, data, ax=ax2)

    plt.tight_layout()
    plt.show()

    # 13) Plot de las 3 mejores y 3 peores operaciones
    best_worst_path = reports_dir / f"best_worst_trades_{symbol}.png"
    plot_best_and_worst_trades_local(
        result,
        data,
        n_best=3,
        n_worst=3,
        window_bars=60,
        save_path=best_worst_path,
    )


def main(symbol: str = "NDXm") -> None:
    """
    Punto de entrada principal del script.
    Por ahora ejecuta un único backtest sobre el símbolo indicado,
    usando la estrategia de barrida en aperturas.
    """
    run_single_backtest(symbol=symbol)


if __name__ == "__main__":
    main(symbol="NDXm")
