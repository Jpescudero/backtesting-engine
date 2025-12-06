# src/strategies/barrida_apertura.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Tuple

import numpy as np
import pandas as pd

from src.data.feeds import OHLCVArrays


@dataclass
class StrategyResult:
    """
    Resultado estándar de una estrategia discrecional:
      - signals: array int8 (-1, 0, +1)
      - meta:    diccionario con parámetros y diagnósticos
    """
    signals: np.ndarray
    meta: Dict[str, Any]


class StrategyBarridaApertura:
    """
    Estrategia de 'barrida' en aperturas:

    Idea básica:
      1) Solo mira alrededor de las aperturas (mañana y tarde).
      2) Busca dos velas bajistas con volumen extremo y caída fuerte
         en pocos minutos (flush / barrida).
      3) En la siguiente vela, exige un reversal potente (>= k * ATR).
      4) Lanza una señal de compra (+1) en esa vela de reversal.

    Las salidas (SL/TP/time_stop) las gestiona el motor de backtest.
    """

    def __init__(
        self,
        volume_percentile: float = 99.0,
        use_two_bearish_bars: bool = True,
        confirm_reversal: bool = True,
        panic_lookback: int = 60,
        panic_threshold: float = -0.0075,
        pre_open_minutes: int = 5,
        post_open_minutes: int = 20,
        atr_period: int = 14,
        min_reversal_strength_atr: float = 1.0,
        min_drop_pct: float = 0.005,        # 0.5% de caída mínima
        max_drop_bars: int = 5,
        session_open_times: Tuple[Tuple[int, int], ...] = ((9, 0), (15, 0)),
    ) -> None:
        self.volume_percentile = float(volume_percentile)
        self.use_two_bearish_bars = bool(use_two_bearish_bars)
        self.confirm_reversal = bool(confirm_reversal)
        self.panic_lookback = int(panic_lookback)
        self.panic_threshold = float(panic_threshold)
        self.pre_open_minutes = int(pre_open_minutes)
        self.post_open_minutes = int(post_open_minutes)
        self.atr_period = int(atr_period)
        self.min_reversal_strength_atr = float(min_reversal_strength_atr)
        self.min_drop_pct = float(min_drop_pct)
        self.max_drop_bars = int(max_drop_bars)
        self.session_open_times = tuple(session_open_times)

    # ============================================================
    # Utilidades internas
    # ============================================================

    def _build_session_mask(self, dt_index: pd.DatetimeIndex) -> np.ndarray:
        """
        Devuelve un boolean mask de las barras que están dentro de las
        ventanas de pre/post apertura definidas.

        Para cada hora de apertura h:
          - incluimos barras con minutos en [h*60 - pre_open, h*60 + post_open]
        """
        minutes_of_day = dt_index.hour * 60 + dt_index.minute
        mask = np.zeros(len(dt_index), dtype=bool)

        for (open_hour, open_minute) in self.session_open_times:
            open_min = open_hour * 60 + open_minute
            lo = open_min - self.pre_open_minutes
            hi = open_min + self.post_open_minutes
            mask |= (minutes_of_day >= lo) & (minutes_of_day <= hi)

        return mask

    def _compute_atr(self, h: np.ndarray, l: np.ndarray, c: np.ndarray) -> np.ndarray:
        """
        ATR clásico (Wilder) aproximado con media móvil simple.
        Devolvemos un array de la misma longitud que c.
        """
        n = c.shape[0]
        if n == 0:
            return np.zeros(0, dtype=float)

        tr = np.empty(n, dtype=float)
        tr[0] = h[0] - l[0]
        for i in range(1, n):
            tr1 = h[i] - l[i]
            tr2 = abs(h[i] - c[i - 1])
            tr3 = abs(l[i] - c[i - 1])
            tr[i] = max(tr1, tr2, tr3)

        if self.atr_period <= 1 or self.atr_period >= n:
            atr = np.full(n, tr.mean(), dtype=float)
        else:
            kernel = np.ones(self.atr_period, dtype=float) / float(self.atr_period)
            atr_valid = np.convolve(tr, kernel, mode="valid")
            atr = np.empty(n, dtype=float)
            atr[: self.atr_period - 1] = atr_valid[0]
            atr[self.atr_period - 1 :] = atr_valid

        return atr

    # ============================================================
    # Núcleo de la estrategia
    # ============================================================

    def generate_signals(self, data: OHLCVArrays) -> StrategyResult:
        """
        Genera señales (+1, 0, -1) para todo el histórico de 'data'.

        Por diseño:
          - Solo generamos +1 (entradas largas). No forzamos salidas (-1),
            éstas las gestiona el motor con SL/TP/time_stop.
        """
        # Convertimos timestamps a DateTimeIndex de pandas
        dt_index = pd.to_datetime(data.ts)

        o = data.o.astype(float)
        h = data.h.astype(float)
        l = data.l.astype(float)
        c = data.c.astype(float)
        v = data.v.astype(float)

        n = c.shape[0]
        signals = np.zeros(n, dtype=np.int8)

        # 1) Máscara de barras en ventana de apertura
        session_mask = self._build_session_mask(dt_index)

        # 2) Umbral de volumen extremo (percentil alto)
        if session_mask.any():
            vol_threshold = float(
                np.percentile(v[session_mask], self.volume_percentile)
            )
        else:
            vol_threshold = float(np.percentile(v, self.volume_percentile))

        # 3) ATR para medir fuerza del reversal
        atr = self._compute_atr(h, l, c)

        # 4) Búsqueda de patrones: dos velas bajistas + flush + reversal fuerte
        start_idx = max(self.max_drop_bars + 2, self.atr_period + 1, self.panic_lookback + 2)

        for i in range(start_idx, n):
            # Indices: flush1 (i-2), flush2 (i-1), reversal (i)
            f1 = i - 2
            f2 = i - 1
            rev = i

            # Deben estar todos dentro de la ventana de apertura
            if not (session_mask[f1] and session_mask[f2] and session_mask[rev]):
                continue

            # Comprobamos velas bajistas de flush
            if self.use_two_bearish_bars:
                if not (c[f1] < o[f1] and c[f2] < o[f2]):
                    continue
            else:
                if not (c[f2] < o[f2]):
                    continue

            # Volumen extremo en la segunda vela de flush
            if v[f2] < vol_threshold:
                continue

            # Caída fuerte en pocas barras:
            #   desde el máximo de las últimas max_drop_bars hasta el cierre de f2
            start_drop = max(0, f2 - self.max_drop_bars + 1)
            recent_high = np.max(h[start_drop : f2 + 1])
            if recent_high <= 0 or not np.isfinite(recent_high):
                continue

            drop_pct = (recent_high - c[f2]) / recent_high
            if drop_pct < self.min_drop_pct:
                # No consideramos esto una 'barrida' seria
                continue

            # Filtro de pánico extremo: evitamos caídas demasiado grandes
            if self.panic_lookback > 0 and f2 - self.panic_lookback >= 0:
                base_price = c[f2 - self.panic_lookback]
                if base_price > 0 and np.isfinite(base_price):
                    panic_ret = (c[f2] - base_price) / base_price
                    if panic_ret <= self.panic_threshold:
                        # Movimiento tipo crash/pánico, preferimos no entrar
                        continue

            # Vela de reversal (rev): verde + rango >= k * ATR
            if self.confirm_reversal:
                if not (c[rev] > o[rev]):
                    continue

                if not np.isfinite(atr[rev]) or atr[rev] <= 0:
                    continue

                prev_close = c[rev - 1]
                true_range = max(
                    h[rev] - l[rev],
                    abs(h[rev] - prev_close),
                    abs(l[rev] - prev_close),
                )

                if true_range < self.min_reversal_strength_atr * atr[rev]:
                    continue

            # Si hemos llegado hasta aquí, consideramos que hay patrón de barrida + reversal
            signals[rev] = 1

        meta: Dict[str, Any] = {
            "volume_percentile": self.volume_percentile,
            "use_two_bearish_bars": self.use_two_bearish_bars,
            "confirm_reversal": self.confirm_reversal,
            "panic_lookback": self.panic_lookback,
            "panic_threshold": self.panic_threshold,
            "session_open_times": self.session_open_times,
            "pre_open_minutes": self.pre_open_minutes,
            "post_open_minutes": self.post_open_minutes,
            "atr_period": self.atr_period,
            "min_reversal_strength_atr": self.min_reversal_strength_atr,
            "min_drop_pct": self.min_drop_pct,
            "max_drop_bars": self.max_drop_bars,
            "session_mask": session_mask,
            "vol_threshold": vol_threshold,
            "n_entries": int((signals == 1).sum()),
        }

        return StrategyResult(signals=signals, meta=meta)
