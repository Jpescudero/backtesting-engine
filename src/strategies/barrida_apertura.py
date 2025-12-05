# src/strategies/barrida_apertura.py

from __future__ import annotations

from typing import Dict, Any

import numpy as np
import pandas as pd

from src.data.feeds import OHLCVArrays
from src.strategies.base import BaseStrategy, StrategyResult


class StrategyBarridaApertura(BaseStrategy):
    """
    Estrategia tipo "barrida" en apertura:

    Idea (versión 1, simplificada):
      - Sólo miramos ventanas horarias alrededor de las 9:00 y 15:00 (hora Europa/Madrid).
      - Buscamos velas bajistas con volumen alto (por encima de un cierto percentil).
      - Tras esa "barrida" bajista, marcamos una señal de compra (+1) en la propia barra.
      - El cierre lo gestiona el motor vía SL/TP/time-stop.

    NOTA: es una primera versión para poder backtestear y ver estadísticas;
    luego podemos refinar condiciones (2 velas seguidas, confirmación de reversal, etc.).
    """

    def __init__(
        self,
        volume_percentile: float = 80.0,
        use_two_bearish_bars: bool = False,
    ) -> None:
        """
        Parámetros:
          - volume_percentile: umbral de volumen alto (percentil global del sample).
          - use_two_bearish_bars: si True, exige 2 velas bajistas consecutivas
                                  con volumen alto.
        """
        self.volume_percentile = volume_percentile
        self.use_two_bearish_bars = use_two_bearish_bars

    def _compute_session_mask(self, idx_local: pd.DatetimeIndex) -> np.ndarray:
        """
        Devuelve un boolean array que marca barras entre:
          - 8:55 y 10:00
          - 14:55 y 16:00
        (Hora Europa/Madrid)
        """
        hour = idx_local.hour
        minute = idx_local.minute
        minutes_from_midnight = hour * 60 + minute

        # Ventanas (en minutos desde medianoche)
        # 9:00   ->  9*60  = 540
        # 10:00  -> 10*60  = 600
        # 15:00  -> 15*60 = 900
        # 16:00  -> 16*60 = 960
        mask_9 = (minutes_from_midnight >= 9 * 60 - 5) & (minutes_from_midnight <= 10 * 60)
        mask_15 = (minutes_from_midnight >= 15 * 60 - 5) & (minutes_from_midnight <= 16 * 60)

        return (mask_9 | mask_15).astype(bool)

    def generate_signals(self, data: OHLCVArrays) -> StrategyResult:
        n = data.c.shape[0]
        signals = np.zeros(n, dtype=np.int8)

        # Convertimos timestamps a índice datetime en Europa/Madrid
        idx_utc = pd.to_datetime(data.ts, utc=True)
        idx_local = idx_utc.tz_convert("Europe/Madrid")

        session_mask = self._compute_session_mask(idx_local)

        # Serie de precios y volumen en DataFrame para manipular fácil
        df = pd.DataFrame(
            {
                "open": data.o,
                "high": data.h,
                "low": data.l,
                "close": data.c,
                "volume": data.v,
            },
            index=idx_local,
        )

        # Velas bajistas
        bearish = df["close"] < df["open"]

        # Volumen alto según percentil global
        vol_threshold = np.nanpercentile(df["volume"].values, self.volume_percentile)
        high_vol = df["volume"] >= vol_threshold

        if self.use_two_bearish_bars:
            # Requiere dos velas bajistas consecutivas con volumen alto
            cond = (
                bearish
                & high_vol
                & bearish.shift(1).fillna(False)
                & high_vol.shift(1).fillna(False)
            )
        else:
            # Una sola vela bajista y vol alto
            cond = bearish & high_vol

        # Restricción a las ventanas de apertura
        cond = cond & session_mask

        entry_idx = np.where(cond.values)[0]
        if entry_idx.size > 0:
            signals[entry_idx] = 1

        meta: Dict[str, Any] = {
            "volume_percentile": self.volume_percentile,
            "use_two_bearish_bars": self.use_two_bearish_bars,
            "session_mask": session_mask,
            "vol_threshold": vol_threshold,
            "n_entries": int(entry_idx.size),
        }

        return StrategyResult(signals=signals, meta=meta)
