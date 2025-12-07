# src/strategies/microstructure_reversal.py

from __future__ import annotations

from dataclasses import dataclass
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


class StrategyMicrostructureReversal:
    """
    Estrategia long-only inspirada en barridas rápidas y reversión de
    microestructura:

    1) Contexto: tendencia alcista (precio > EMA50, EMA20 > EMA50, EMA50 ascendente).
    2) Pullback rápido: caída de 0.4–1.1 ATR en ≤8 velas desde el máximo reciente.
    3) Exhaustion bar: nueva mecha mínima, cuerpo pequeño, cierre en la mitad superior
       y sin incremento de volumen.
    4) Bullish shift candle: vela verde amplia (>0.6 ATR), cierra sobre el máximo
       previo y rompe la microestructura (nuevo máximo vs últimas 3 velas).
    5) Entrada en el cierre de la vela de shift.
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
        if n >= period:
            atr[period - 1] = tr[:period].mean()
            alpha = 1.0 / period
            for i in range(period, n):
                atr[i] = (1 - alpha) * atr[i - 1] + alpha * tr[i]

        return atr

    @staticmethod
    def _pullback_features(
        h: np.ndarray,
        l: np.ndarray,
        atr: np.ndarray,
        lookback: int,
    ) -> Dict[str, np.ndarray]:
        n = len(h)
        drop_atr = np.full(n, np.nan, dtype=float)
        bars_since_high = np.full(n, -1, dtype=int)
        ref_high = np.full(n, np.nan, dtype=float)

        for i in range(n):
            start = max(0, i - lookback + 1)
            window_high = h[start : i + 1]
            local_idx = int(np.argmax(window_high))
            peak_idx = start + local_idx

            bars_since_high[i] = i - peak_idx
            ref_high[i] = window_high[local_idx]

            if atr[i] > 0:
                drop_atr[i] = (ref_high[i] - l[i]) / atr[i]

        return {
            "drop_atr": drop_atr,
            "bars_since_high": bars_since_high,
            "ref_high": ref_high,
        }

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

        # 1) Contexto de tendencia
        ema_short = self._ema(c, p.ema_short)
        ema_long = self._ema(c, p.ema_long)

        ema_long_slope = np.zeros(n, dtype=bool)
        if n > 1:
            ema_long_slope[1:] = ema_long[1:] > ema_long[:-1]

        trend_mask = (c > ema_long) & (ema_short > ema_long) & ema_long_slope

        # 2) ATR y características de pullback
        atr = self._atr(h=h, l=l, c=c, period=p.atr_period)
        pullback = self._pullback_features(h=h, l=l, atr=atr, lookback=p.max_pullback_bars)

        pullback_mask = (
            (pullback["bars_since_high"] >= 1)
            & (pullback["bars_since_high"] <= p.max_pullback_bars)
            & (pullback["drop_atr"] >= p.min_pullback_atr)
            & (pullback["drop_atr"] <= p.max_pullback_atr)
            & (c < pullback["ref_high"])
        )

        # 3) Exhaustion bar (vela previa a la señal)
        range_ = h - l
        body = np.abs(c - o)
        with np.errstate(divide="ignore", invalid="ignore"):
            close_pos = np.where(range_ > 0, (c - l) / range_, 0.5)
            body_ratio = np.where(range_ > 0, body / range_, 0.0)

        prev_vol = np.roll(v, 1)
        prev_vol[0] = v[0]

        prev_low = np.roll(l, 1)
        prev_low[0] = l[0]

        exhaustion_mask = (
            (l < prev_low)
            & (close_pos >= p.exhaustion_close_min)
            & (close_pos <= p.exhaustion_close_max)
            & (body_ratio <= p.exhaustion_body_max_ratio)
            & (v <= prev_vol)
        )

        # 4) Bullish shift candle
        prev_high = np.roll(h, 1)
        prev_high[0] = h[0]

        shift_body = c - o
        shift_mask = (
            (shift_body > 0)
            & (atr > 0)
            & (shift_body >= p.shift_body_atr * atr)
            & (c > prev_high)
        )

        # 5) Ruptura de microestructura
        roll_max_prev = pd.Series(h).rolling(p.structure_break_lookback, min_periods=1).max().shift(1)
        roll_max_prev = roll_max_prev.to_numpy()
        structure_break = h > roll_max_prev

        # 6) Entradas: vela actual debe ser shift + romper estructura;
        # la vela previa debe ser exhaustion dentro de un pullback válido y bajo tendencia alcista.
        prev_exhaustion = np.roll(exhaustion_mask & pullback_mask & trend_mask, 1)
        prev_exhaustion[0] = False

        entries_mask = shift_mask & structure_break & prev_exhaustion & trend_mask

        signals = np.zeros(n, dtype=np.int8)
        signals[entries_mask] = 1

        meta: Dict[str, Any] = {
            "n_entries": int(entries_mask.sum()),
            "trend_mask": trend_mask,
            "pullback_mask": pullback_mask,
            "exhaustion_mask": exhaustion_mask,
            "shift_mask": shift_mask,
            "structure_break": structure_break,
            "params": p,
        }

        return StrategyResult(signals=signals, meta=meta)
