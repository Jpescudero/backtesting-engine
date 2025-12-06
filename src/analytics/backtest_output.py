# src/analytics/backtest_output.py
"""
Utilities para generar y guardar un resumen de backtest
en formatos fáciles de compartir (Excel + JSON), minimizando
al máximo el tiempo de generación.

Diseño "ultralight":
- Equity en Excel: como mucho 5.000 puntos (muestra/slice).
- Trades en Excel: como mucho 50.000 filas.
- JSON: sólo equity_stats, trade_stats y meta_summary ligero.
"""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple

import json
import numpy as np
import pandas as pd


# Límites para no matar Excel ni el tiempo de escritura
MAX_EQUITY_ROWS_EXCEL = 5_000
MAX_TRADES_ROWS_EXCEL = 50_000


# ============================================================
# Helpers de serialización ligera para JSON
# ============================================================


def _to_serializable(value: Any) -> Any:
    """
    Convierte valores (incluyendo estructuras anidadas) a tipos
    compatibles con JSON (dict, list, str, float, int, bool, None).

    Pensado para equity_stats y trade_stats (que deberían ser
    diccionarios relativamente pequeños).
    """
    # Dataclasses
    if is_dataclass(value):
        return {k: _to_serializable(v) for k, v in asdict(value).items()}

    # NumPy escalares
    if isinstance(value, (np.floating, np.integer)):
        return float(value)

    # NumPy arrays -> resumen (no guardamos arrays enormes en JSON)
    if isinstance(value, np.ndarray):
        return {
            "type": "ndarray",
            "shape": list(value.shape),
            "dtype": str(value.dtype),
        }

    # pandas Timestamp
    if isinstance(value, pd.Timestamp):
        return value.isoformat()

    # pandas Series / Index -> resumen
    if isinstance(value, (pd.Series, pd.Index)):
        return {
            "type": type(value).__name__,
            "length": len(value),
            "dtype": str(value.dtype),
        }

    # Mapping (dict, etc.)
    if isinstance(value, Mapping):
        return {str(k): _to_serializable(v) for k, v in value.items()}

    # Secuencias (lista/tupla/conjunto)
    if isinstance(value, (list, tuple, set)):
        return [_to_serializable(v) for v in list(value)]

    # Tipos básicos (str, int, float, bool, None) pasan tal cual
    return value


def _meta_summary(meta: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    """
    En lugar de volcar meta completa (que puede contener arrays muy grandes),
    guardamos sólo un resumen ligero por clave:
      - tipo de objeto
      - si es array/Series: su longitud o shape
      - si es mapping: nº de claves
    """
    meta = meta or {}
    summary: Dict[str, Any] = {}

    for k, v in meta.items():
        key = str(k)
        if isinstance(v, np.ndarray):
            summary[key] = {
                "type": "ndarray",
                "shape": list(v.shape),
                "dtype": str(v.dtype),
            }
        elif isinstance(v, (pd.Series, pd.Index)):
            summary[key] = {
                "type": type(v).__name__,
                "length": len(v),
                "dtype": str(v.dtype),
            }
        elif isinstance(v, Mapping):
            summary[key] = {
                "type": "mapping",
                "n_keys": len(v),
                "keys_sample": list(v.keys())[:10],
            }
        elif isinstance(v, (list, tuple, set)):
            summary[key] = {
                "type": type(v).__name__,
                "length": len(v),
            }
        else:
            # valor escalar o pequeño
            summary[key] = {
                "type": type(v).__name__,
                "value": v,
            }

    return summary


def build_backtest_summary(
    *,
    symbol: str,
    strategy_name: str,
    equity_stats: Mapping[str, Any],
    trade_stats: Mapping[str, Any],
    meta: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Construye un diccionario resumen del backtest (ligero)."""
    return {
        "symbol": symbol,
        "strategy_name": strategy_name,
        "meta_summary": _meta_summary(meta),
        "equity_stats": _to_serializable(equity_stats),
        "trade_stats": _to_serializable(trade_stats),
    }


# ============================================================
# Helpers para hacer DataFrames "Excel-safe" y pequeños
# ============================================================


def _make_df_excel_safe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Devuelve una copia de df donde:
      - Cualquier columna datetime con timezone se convierte a tz-naive
        (Excel no soporta timezones).
    """
    df = df.copy()

    # Columnas
    for col in df.columns:
        col_data = df[col]
        # Dtype con tz (pandas>=1.0: DatetimeTZDtype)
        if isinstance(col_data.dtype, pd.DatetimeTZDtype):
            df[col] = col_data.dt.tz_localize(None)

    # Índice
    idx = df.index
    if isinstance(idx, pd.DatetimeIndex) and idx.tz is not None:
        df.index = idx.tz_localize(None)

    return df


def _prepare_equity_df_for_excel(equity_series: pd.Series) -> Tuple[pd.DataFrame, Optional[int]]:
    """
    Prepara un DataFrame de equity para escribir a Excel:

    - Se quita la timezone del índice si existe.
    - Se limita a un máximo de MAX_EQUITY_ROWS_EXCEL filas
      tomando una muestra (submuestreo simple).
    """
    # 1) Asegurarnos de que el índice NO tiene timezone
    idx = equity_series.index
    if isinstance(idx, pd.DatetimeIndex) and idx.tz is not None:
        idx_naive = idx.tz_localize(None)
        equity_series = pd.Series(
            equity_series.values,
            index=idx_naive,
            name=equity_series.name,
        )

    equity_df = equity_series.to_frame(name="equity")
    equity_df.index.name = "timestamp"

    n_rows = len(equity_df)
    if n_rows <= MAX_EQUITY_ROWS_EXCEL:
        return equity_df, None

    # Submuestreo simple para quedarnos con ~MAX_EQUITY_ROWS_EXCEL puntos
    step = int(np.ceil(n_rows / MAX_EQUITY_ROWS_EXCEL))
    equity_df_ds = equity_df.iloc[::step].copy()
    # Si aún así queda alguna fila de más, cortamos por si acaso
    equity_df_ds = equity_df_ds.tail(MAX_EQUITY_ROWS_EXCEL)
    equity_df_ds["__sample_step__"] = step

    return equity_df_ds, step


def _prepare_trades_df_for_excel(trades_df: pd.DataFrame) -> Tuple[pd.DataFrame, Optional[int]]:
    """
    Prepara trades_df para Excel:
      - Quita timezones en columnas datetime.
      - Limita el nº de filas a MAX_TRADES_ROWS_EXCEL (head).
    """
    trades_df = _make_df_excel_safe(trades_df)

    n_rows = len(trades_df)
    if n_rows <= MAX_TRADES_ROWS_EXCEL:
        return trades_df, None

    trades_df_small = trades_df.head(MAX_TRADES_ROWS_EXCEL).copy()
    return trades_df_small, n_rows


# ============================================================
# Función principal de salida
# ============================================================


def save_backtest_summary_to_excel(
    *,
    base_dir: Path,
    filename: str,
    symbol: str,
    strategy_name: str,
    equity_series: pd.Series,
    trades_df: pd.DataFrame,
    equity_stats: Mapping[str, Any],
    trade_stats: Mapping[str, Any],
    meta: Optional[Mapping[str, Any]] = None,
) -> Tuple[Path, Path]:
    """
    Guarda un resumen del backtest en Excel + JSON (modo ultraligero).

    - Excel:
        - Hoja "equity_sample": equity submuestreada (<= 5.000 filas)
        - Hoja "trades": como máximo 50.000 trades
        - Hoja "equity_stats" y "trade_stats"
        - Hoja "meta" con información textual muy ligera
    - JSON:
        - Resumen ligero (equity_stats, trade_stats, meta_summary)
    """
    base_dir = Path(base_dir).resolve()
    base_dir.mkdir(parents=True, exist_ok=True)

    excel_path = base_dir / filename
    json_path = excel_path.with_suffix(".json")

    # ----- JSON ligero -----
    summary_dict = build_backtest_summary(
        symbol=symbol,
        strategy_name=strategy_name,
        equity_stats=equity_stats,
        trade_stats=trade_stats,
        meta=meta,
    )

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary_dict, f, ensure_ascii=False, indent=2)

    # ----- Excel ultraligero -----
    equity_df, step = _prepare_equity_df_for_excel(equity_series)
    equity_df = _make_df_excel_safe(equity_df)

    trades_df_excel, n_trades_original = _prepare_trades_df_for_excel(trades_df)

    eq_stats_df = pd.DataFrame(
        [{"metric": k, "value": _to_serializable(v)} for k, v in equity_stats.items()]
    )
    tr_stats_df = pd.DataFrame(
        [{"metric": k, "value": _to_serializable(v)} for k, v in trade_stats.items()]
    )

    meta_items = list((meta or {}).items())
    meta_extra_info = []

    if step is not None:
        meta_extra_info.append(
            (
                "equity_sample_step",
                f"Equity muestreada manteniendo 1 de cada {step} puntos, "
                f"máx {MAX_EQUITY_ROWS_EXCEL} filas en Excel.",
            )
        )

    if n_trades_original is not None and n_trades_original > MAX_TRADES_ROWS_EXCEL:
        meta_extra_info.append(
            (
                "trades_excel_limit",
                f"Se han guardado solo los primeros {MAX_TRADES_ROWS_EXCEL} trades "
                f"de un total de {n_trades_original}.",
            )
        )

    meta_all = meta_items + meta_extra_info

    meta_df = pd.DataFrame(
        [{"key": str(k), "value": v} for k, v in meta_all]
    )

    with pd.ExcelWriter(excel_path) as writer:
        equity_df.to_excel(writer, sheet_name="equity_sample")
        trades_df_excel.to_excel(writer, sheet_name="trades", index=False)
        eq_stats_df.to_excel(writer, sheet_name="equity_stats", index=False)
        tr_stats_df.to_excel(writer, sheet_name="trade_stats", index=False)
        meta_df.to_excel(writer, sheet_name="meta", index=False)

    return excel_path, json_path


