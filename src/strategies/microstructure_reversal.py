# src/strategies/microstructure_reversal.py

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict

import numpy as np
import pandas as pd

from src.data.feeds import OHLCVArrays
from src.strategies.base import StrategyResult


@dataclass
class MicrostructureParams:
    """
    Parámetros configurables para la estrategia de reversión por barrida
    basada en microestructura.
    """

    ema_short: int = 20
    ema_long: int = 50
    atr_period: int = 20

    min_pullback_atr: float = 0.4
    max_pullback_atr: float = 1.1
    max_pullback_bars: int = 8

    exhaustion_close_min: float = 0.4
    exhaustion_close_max: float = 0.6
    exhaustion_body_max_ratio: float = 0.4

    shift_body_atr: float = 0.6
    structure_break_lookback: int = 3

    volume_period: int = 20
    min_rvol: float = 1.0

    atr_stop_mult: float = 1.2
    atr_tp_mult: float = 2.2
    structure_buffer_atr: float = 0.2
    structure_stop_lookback: int = 6

    breakeven_atr_trigger: float = 0.5
    breakeven_lookahead: int = 10

    max_holding_bars: int = 60


class StrategyMicrostructureReversal:
    """
    Estrategia de reversión por microestructura (versión 2.0).

    Flujo general:
      1) Filtro de tendencia fuerte vía EMAs.
      2) Pullback en ATR dentro de una ventana máxima.
      3) Vela de agotamiento (cuerpo pequeño, cierre en zona media de rango).
      4) Vela de shift a favor de la tendencia que rompe estructura.
      5) Filtro de actividad por volumen relativo.
      6) SL / TP basados en ATR + límite de barras en el trade.
    """

    def __init__(self, **kwargs: Any) -> None:
        self.params = MicrostructureParams(**kwargs)

    # ---------------------------------------------------------
    # Helpers
    # ---------------------------------------------------------
    @staticmethod
    def _ema(series: np.ndarray, span: int) -> np.ndarray:
        if len(series) == 0:
            return np.zeros(0, dtype=float)
        return pd.Series(series).ewm(span=span, adjust=False).mean().to_numpy()

    @staticmethod
    def _atr(h: np.ndarray, l: np.ndarray, c: np.ndarray, period: int) -> np.ndarray:
        n = len(c)
        if n == 0:
            return np.zeros(0, dtype=float)

        tr = np.empty(n, dtype=float)
        tr[0] = h[0] - l[0]
        for i in range(1, n):
            high_low = h[i] - l[i]
            high_close_prev = abs(h[i] - c[i - 1])
            low_close_prev = abs(l[i] - c[i - 1])
            tr[i] = max(high_low, high_close_prev, low_close_prev)

        atr = np.full(n, np.nan, dtype=float)
        if period <= 0:
            return atr

        alpha = 2.0 / (period + 1)
        for i in range(period - 1, n):
            window_tr = tr[i - period + 1 : i + 1]
            ema = window_tr[0]
            for val in window_tr[1:]:
                ema = alpha * val + (1 - alpha) * ema
            atr[i] = ema

        return atr

    @staticmethod
    def _rolling_max(arr: np.ndarray, window: int) -> np.ndarray:
        return pd.Series(arr).rolling(window, min_periods=1).max().to_numpy()

    @staticmethod
    def _rolling_min(arr: np.ndarray, window: int) -> np.ndarray:
        return pd.Series(arr).rolling(window, min_periods=1).min().to_numpy()

    @staticmethod
    def _forward_rolling_max(arr: np.ndarray, window: int) -> np.ndarray:
        return pd.Series(arr[::-1]).rolling(window, min_periods=1).max().to_numpy()[::-1]

    @staticmethod
    def _forward_rolling_min(arr: np.ndarray, window: int) -> np.ndarray:
        return pd.Series(arr[::-1]).rolling(window, min_periods=1).min().to_numpy()[::-1]

    # ---------------------------------------------------------
    # API principal
    # ---------------------------------------------------------
    def generate_signals(self, data: OHLCVArrays) -> StrategyResult:
        o = np.asarray(data.o)
        h = np.asarray(data.h)
        l = np.asarray(data.l)
        c = np.asarray(data.c)
        v = np.asarray(data.v)

        n = len(c)
        if n == 0:
            return StrategyResult(signals=np.zeros(0, dtype=np.int8), meta={"n_entries": 0})

        p = self.params

        # 1) Tendencia
        ema_short = self._ema(c, p.ema_short)
        ema_long = self._ema(c, p.ema_long)
        uptrend_mask = ema_short > ema_long
        downtrend_mask = ema_short < ema_long

        # 2) ATR
        atr = self._atr(h=h, l=l, c=c, period=p.atr_period)

        # 3) Pullback en ATR
        swing_high = self._rolling_max(h, p.max_pullback_bars)
        swing_low = self._rolling_min(l, p.max_pullback_bars)

        with np.errstate(divide="ignore", invalid="ignore"):
            pullback_atr_long = (swing_high - c) / atr
            pullback_atr_short = (c - swing_low) / atr

        pullback_mask_long = (
            np.isfinite(pullback_atr_long)
            & (pullback_atr_long >= p.min_pullback_atr)
            & (pullback_atr_long <= p.max_pullback_atr)
            & (swing_high > c)
        )
        pullback_mask_short = (
            np.isfinite(pullback_atr_short)
            & (pullback_atr_short >= p.min_pullback_atr)
            & (pullback_atr_short <= p.max_pullback_atr)
            & (swing_low < c)
        )

        # 4) Exhaustion candle (vela previa al shift)
        range_ = h - l
        body = np.abs(c - o)
        tiny = 1e-12
        close_pos_from_low = (c - l) / np.maximum(range_, tiny)
        close_pos_from_high = (h - c) / np.maximum(range_, tiny)
        body_ratio = body / np.maximum(range_, tiny)

        exhaustion_long = (
            (c < o)
            & (close_pos_from_low >= p.exhaustion_close_min)
            & (close_pos_from_low <= p.exhaustion_close_max)
            & (body_ratio <= p.exhaustion_body_max_ratio)
        )
        exhaustion_short = (
            (c > o)
            & (close_pos_from_high >= p.exhaustion_close_min)
            & (close_pos_from_high <= p.exhaustion_close_max)
            & (body_ratio <= p.exhaustion_body_max_ratio)
        )

        exhaustion_long &= pullback_mask_long & uptrend_mask
        exhaustion_short &= pullback_mask_short & downtrend_mask

        # 5) Shift + ruptura de estructura en la vela actual
        body_signed = c - o
        shift_long = (body_signed > 0) & (body >= p.shift_body_atr * atr)
        shift_short = (body_signed < 0) & (body >= p.shift_body_atr * atr)

        prev_max_high = pd.Series(h).rolling(p.structure_break_lookback, min_periods=1).max().shift(1)
        prev_min_low = pd.Series(l).rolling(p.structure_break_lookback, min_periods=1).min().shift(1)
        prev_max_high.iloc[0] = h[0]
        prev_min_low.iloc[0] = l[0]

        structure_break_long = c > prev_max_high.to_numpy()
        structure_break_short = c < prev_min_low.to_numpy()

        shift_long &= structure_break_long & uptrend_mask
        shift_short &= structure_break_short & downtrend_mask

        # 6) Filtro de actividad por volumen
        vol_mean = pd.Series(v).rolling(p.volume_period, min_periods=1).mean().to_numpy()
        with np.errstate(divide="ignore", invalid="ignore"):
            rvol = np.divide(v, vol_mean, out=np.ones_like(v, dtype=float), where=vol_mean > 0)
        high_activity_mask = rvol >= p.min_rvol

        # 7) Entradas: vela previa agotamiento, vela actual shift + ruptura + volumen
        prev_exhaustion_long = np.roll(exhaustion_long, 1)
        prev_exhaustion_long[0] = False
        prev_exhaustion_short = np.roll(exhaustion_short, 1)
        prev_exhaustion_short[0] = False

        entries_long = shift_long & prev_exhaustion_long & high_activity_mask
        entries_short = shift_short & prev_exhaustion_short & high_activity_mask

        signals = np.zeros(n, dtype=np.int8)
        signals[entries_long] = 1
        signals[entries_short] = -1

        # 8) SL / TP / duración
        initial_stop_loss = np.full(n, np.nan, dtype=float)
        take_profit = np.full(n, np.nan, dtype=float)
        time_stop_bars = np.zeros(n, dtype=int)

        swing_low_stop = self._rolling_min(l, p.structure_stop_lookback)
        swing_high_stop = self._rolling_max(h, p.structure_stop_lookback)

        atr_entry = atr
        sl_long = np.minimum(
            c - p.atr_stop_mult * atr_entry,
            swing_low_stop - p.structure_buffer_atr * atr_entry,
        )
        sl_short = np.maximum(
            c + p.atr_stop_mult * atr_entry,
            swing_high_stop + p.structure_buffer_atr * atr_entry,
        )

        tp_long = c + p.atr_tp_mult * atr_entry
        tp_short = c - p.atr_tp_mult * atr_entry

        # Break-even tras movimiento a favor
        future_max_high = self._forward_rolling_max(h, p.breakeven_lookahead)
        future_min_low = self._forward_rolling_min(l, p.breakeven_lookahead)

        mfe_long = future_max_high - c
        mfe_short = c - future_min_low

        be_on_long = mfe_long >= p.breakeven_atr_trigger * atr_entry
        be_on_short = mfe_short >= p.breakeven_atr_trigger * atr_entry

        sl_long = np.where(be_on_long, np.maximum(sl_long, c), sl_long)
        sl_short = np.where(be_on_short, np.minimum(sl_short, c), sl_short)

        initial_stop_loss[entries_long] = sl_long[entries_long]
        take_profit[entries_long] = tp_long[entries_long]

        initial_stop_loss[entries_short] = sl_short[entries_short]
        take_profit[entries_short] = tp_short[entries_short]

        time_stop_bars[entries_long | entries_short] = p.max_holding_bars

        meta_summary: Dict[str, Any] = {
            "params": asdict(p),
            "n_entries_long": int(entries_long.sum()),
            "n_entries_short": int(entries_short.sum()),
            "uptrend_mask": uptrend_mask.tolist(),
            "downtrend_mask": downtrend_mask.tolist(),
            "pullback_mask_long": pullback_mask_long.tolist(),
            "pullback_mask_short": pullback_mask_short.tolist(),
            "exhaustion_mask_long": exhaustion_long.tolist(),
            "exhaustion_mask_short": exhaustion_short.tolist(),
            "shift_mask_long": shift_long.tolist(),
            "shift_mask_short": shift_short.tolist(),
            "structure_break_long": structure_break_long.tolist(),
            "structure_break_short": structure_break_short.tolist(),
            "high_activity_mask": high_activity_mask.tolist(),
            "entries_long": entries_long.tolist(),
            "entries_short": entries_short.tolist(),
            "initial_stop_loss": initial_stop_loss.tolist(),
            "take_profit": take_profit.tolist(),
            "time_stop_bars": time_stop_bars.tolist(),
        }

        return StrategyResult(signals=signals, meta=meta_summary)
