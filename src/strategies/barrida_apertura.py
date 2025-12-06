# src/strategies/barrida_apertura.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd

from src.data.feeds import OHLCVArrays  # tipo contenedor ts, o, h, l, c, v


@dataclass
class StrategyResult:
    """
    Contenedor sencillo para las señales generadas por la estrategia.
    """
    signals: np.ndarray          # 1 = entrar largo, 0 = no hacer nada
    meta: Dict[str, Any]


@dataclass
class BarridaParams:
    """
    Parámetros de la estrategia de "barrida" en aperturas.
    """
    volume_percentile: float = 80.0
    use_two_bearish_bars: bool = True

    # Ventanas horarias (hora local del activo) donde buscamos el patrón
    session_open_times: Tuple[str, ...] = ("09:00", "15:00")
    pre_open_minutes: int = 5
    post_open_minutes: int = 60

    # Filtro de pánico: cambio de precio mínimo aceptable en los últimos N minutos
    panic_lookback: int = 60           # en barras (1m)
    panic_threshold: float = -0.0075   # -0.75 %

    # Confirmación de reversal
    confirm_reversal: bool = True

    # Filtro de fuerza del reversal (en múltiplos de ATR)
    atr_period: int = 14
    min_reversal_strength_atr: float = 0.25  # 0 = desactivado


class StrategyBarridaApertura:
    """
    Estrategia de "barrida" en aperturas:

      1) Ventanas alrededor de las aperturas definidas (09:00 y 15:00).
      2) Busca 2 velas bajistas consecutivas con volumen alto.
      3) En la barra siguiente, si hay reversal (vela alcista) y
         la subida es suficientemente fuerte vs ATR, genera señal de entrada.
      4) Aplica un filtro de "no pánico" basado en el rendimiento
         de la última hora.

    Las salidas (SL, TP, tiempo máximo) las gestiona el motor de
    backtesting vía run_backtest_with_signals.
    """

    def __init__(
        self,
        volume_percentile: float = 80.0,
        use_two_bearish_bars: bool = True,
    ) -> None:
        self.params = BarridaParams(
            volume_percentile=volume_percentile,
            use_two_bearish_bars=use_two_bearish_bars,
        )

    # ---------------------------------------------------------
    # Helpers internos
    # ---------------------------------------------------------

    @staticmethod
    def _compute_atr(
        h: np.ndarray,
        l: np.ndarray,
        c: np.ndarray,
        period: int = 14,
    ) -> np.ndarray:
        """
        ATR clásico (tipo Wilder) en barras de 1m.
        Devuelve un array con NaN en las primeras `period-1` posiciones.
        """
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

    def _build_session_mask(self, ts: np.ndarray) -> np.ndarray:
        """
        Devuelve un array booleano con True en las barras que están
        dentro de las ventanas [open - pre_open_minutes, open + post_open_minutes]
        para cada hora de apertura configurada.
        """
        idx = pd.to_datetime(ts)
        minutes_in_day = idx.hour * 60 + idx.minute
        mask = np.zeros(len(ts), dtype=bool)

        for t_str in self.params.session_open_times:
            hour_str, minute_str = t_str.split(":")
            open_min = int(hour_str) * 60 + int(minute_str)

            start = open_min - self.params.pre_open_minutes
            end = open_min + self.params.post_open_minutes

            mask |= (minutes_in_day >= start) & (minutes_in_day <= end)

        return mask

    # ---------------------------------------------------------
    # API principal
    # ---------------------------------------------------------

    def generate_signals(self, data: OHLCVArrays) -> StrategyResult:
        """
        Genera el vector de señales (long-only) para todo el histórico.
        Cada 1 en `signals` se interpreta como una nueva entrada larga.
        """
        o = np.asarray(data.o)
        h = np.asarray(data.h)
        l = np.asarray(data.l)
        c = np.asarray(data.c)
        v = np.asarray(data.v)
        ts = np.asarray(data.ts)

        n = len(c)
        if n == 0:
            return StrategyResult(
                signals=np.zeros(0, dtype=np.int8),
                meta={"n_entries": 0},
            )

        params = self.params

        # 1) Máscara de sesiones (ventanas alrededor de las aperturas)
        session_mask = self._build_session_mask(ts)

        # 2) Umbral de volumen alto (percentil sobre barras de sesión)
        vol_in_session = v[session_mask]
        if vol_in_session.size > 0:
            vol_threshold = float(
                np.percentile(vol_in_session, params.volume_percentile)
            )
        else:
            vol_threshold = float(
                np.percentile(v, params.volume_percentile)
            )

        vol_high = v >= vol_threshold

        # 3) Patrón de velas bajistas con alto volumen
        bearish = c < o

        if params.use_two_bearish_bars:
            bearish_pair = bearish & np.roll(bearish, 1)
            vol_pair = vol_high & np.roll(vol_high, 1)
            pattern_bar = bearish_pair & vol_pair

            # La barra candidata de entrada es la siguiente
            candidate = np.roll(pattern_bar, 1)
            candidate[0] = False
        else:
            candidate = bearish & vol_high

        # 4) Filtro de pánico: evitar días con caída fuerte en la última hora
        ret_lookback = np.zeros(n, dtype=float)
        lb = params.panic_lookback
        if lb < n:
            ret_lookback[lb:] = c[lb:] / c[:-lb] - 1.0
        not_panic = ret_lookback > params.panic_threshold

        # 5) Confirmación de reversal en la barra de entrada
        prev_close = np.roll(c, 1)
        prev_close[0] = c[0]

        confirm_mask = np.ones(n, dtype=bool)
        if params.confirm_reversal:
            confirm_mask = (c > o) & (c > prev_close)

        # 6) Filtro de fuerza del reversal vía ATR
        atr = self._compute_atr(h=h, l=l, c=c, period=params.atr_period)
        strength_mask = np.ones(n, dtype=bool)
        if params.min_reversal_strength_atr > 0.0 and np.isfinite(atr).any():
            strength = c - prev_close
            required = params.min_reversal_strength_atr * atr
            strength_mask = strength >= required

        # 7) Señales finales: todas las condiciones a la vez
        entries_mask = (
            candidate
            & session_mask
            & not_panic
            & confirm_mask
            & strength_mask
        )

        signals = np.zeros(n, dtype=np.int8)
        signals[entries_mask] = 1  # long-only

        meta: Dict[str, Any] = {
            "volume_percentile": float(params.volume_percentile),
            "use_two_bearish_bars": bool(params.use_two_bearish_bars),
            "session_mask": session_mask,
            "vol_threshold": vol_threshold,
            "n_entries": int(entries_mask.sum()),
            "confirm_reversal": bool(params.confirm_reversal),
            "panic_lookback": int(params.panic_lookback),
            "panic_threshold": float(params.panic_threshold),
            "session_open_times": params.session_open_times,
            "pre_open_minutes": int(params.pre_open_minutes),
            "post_open_minutes": int(params.post_open_minutes),
            "atr_period": int(params.atr_period),
            "min_reversal_strength_atr": float(
                params.min_reversal_strength_atr
            ),
        }

        return StrategyResult(signals=signals, meta=meta)
