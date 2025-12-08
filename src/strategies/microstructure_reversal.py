# src/strategies/microstructure_reversal.py

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional

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
    atr_timeframe: Optional[str] = "1m"
    atr_timeframe_period: int = 10

    min_pullback_atr: float = 0.3
    max_pullback_atr: float = 1.3
    max_pullback_bars: int = 12

    exhaustion_close_min: float = 0.35
    exhaustion_close_max: float = 0.65
    exhaustion_body_max_ratio: float = 0.5

    shift_body_atr: float = 0.45
    structure_break_lookback: int = 3

    volume_period: int = 20
    min_rvol: float = 0.8

    vol_percentile_low: float = 0.20
    vol_percentile_high: float = 0.98

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
    def _atr(h: np.ndarray, low: np.ndarray, c: np.ndarray, period: int) -> np.ndarray:
        n = len(c)
        if n == 0:
            return np.zeros(0, dtype=float)

        tr = np.empty(n, dtype=float)
        tr[0] = h[0] - low[0]
        for i in range(1, n):
            high_low = h[i] - low[i]
            high_close_prev = abs(h[i] - c[i - 1])
            low_close_prev = abs(low[i] - c[i - 1])
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
    def _align_indicator_to_target_ts(
        indicator: np.ndarray, source_ts: np.ndarray, target_ts: np.ndarray
    ) -> np.ndarray:
        """
        Reindexa un array de indicador calculado en otro timeframe para que
        coincida con los timestamps objetivo (relleno forward-fill).
        """
        if indicator.shape[0] == 0:
            return np.zeros_like(target_ts, dtype=float)

        source_index = pd.to_datetime(source_ts, utc=True)
        target_index = pd.to_datetime(target_ts, utc=True)

        aligned = pd.Series(indicator, index=source_index).reindex(target_index, method="ffill").to_numpy()
        return aligned

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

    def compute_lower_timeframe_atr(self, lower_data: OHLCVArrays, target_ts: np.ndarray) -> np.ndarray:
        """
        Calcula un ATR en un timeframe inferior (por ejemplo 1m) y lo reindexa
        a los timestamps del dataframe objetivo.
        """

        atr_lower = self._atr(
            h=np.asarray(lower_data.h),
            low=np.asarray(lower_data.low),
            c=np.asarray(lower_data.c),
            period=self.params.atr_timeframe_period,
        )

        return self._align_indicator_to_target_ts(indicator=atr_lower, source_ts=lower_data.ts, target_ts=target_ts)

    # ---------------------------------------------------------
    # API principal
    # ---------------------------------------------------------
    def generate_signals(self, data: OHLCVArrays, external_atr: Optional[np.ndarray] = None) -> StrategyResult:
        o = np.asarray(data.o)
        h = np.asarray(data.h)
        low = np.asarray(data.low)
        c = np.asarray(data.c)
        v = np.asarray(data.v)
        ts = np.asarray(data.ts)

        n = len(c)
        if n == 0:
            return StrategyResult(signals=np.zeros(0, dtype=np.int8), meta={"n_entries": 0})

        p = self.params

        # Ventanas horarias (Europe/Madrid con horario de verano/invierno automático)
        idx_local = pd.to_datetime(ts, utc=True).tz_convert("Europe/Madrid")
        minutes_in_day = idx_local.hour * 60 + idx_local.minute
        session_mask = ((minutes_in_day >= 8 * 60 + 50) & (minutes_in_day <= 10 * 60)) | (
            (minutes_in_day >= 15 * 60 + 20) & (minutes_in_day <= 16 * 60 + 30)
        )

        day_index = idx_local.normalize()

        # 1) Tendencia
        ema_short = self._ema(c, p.ema_short)
        ema_long = self._ema(c, p.ema_long)
        uptrend_mask = ema_short > ema_long
        downtrend_mask = ema_short < ema_long

        # 2) ATR
        if external_atr is not None:
            if external_atr.shape[0] != n:
                raise ValueError("external_atr debe tener la misma longitud que los datos del timeframe objetivo")
            atr = external_atr
        else:
            atr = self._atr(h=h, low=low, c=c, period=p.atr_period)

        # 3) Pullback en ATR
        swing_high = self._rolling_max(h, p.max_pullback_bars)
        swing_low = self._rolling_min(low, p.max_pullback_bars)

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
        range_ = h - low
        body = np.abs(c - o)
        tiny = 1e-12
        close_pos_from_low = (c - low) / np.maximum(range_, tiny)
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
        prev_min_low = pd.Series(low).rolling(p.structure_break_lookback, min_periods=1).min().shift(1)
        prev_max_high.iloc[0] = h[0]
        prev_min_low.iloc[0] = low[0]

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

        # 6b) Filtro de régimen de volatilidad: evitar extremos intradía
        atr_series = pd.Series(atr)
        atr_intraday_median = atr_series.groupby(day_index).transform("median")
        atr_filter = np.isfinite(atr) & np.isfinite(atr_intraday_median)
        atr_filter &= atr <= 2.3 * atr_intraday_median.to_numpy()

        vol_series = pd.Series(v, dtype=float)
        vol_q_low = vol_series.groupby(day_index).transform(lambda x: x.quantile(p.vol_percentile_low))
        vol_q_high = vol_series.groupby(day_index).transform(lambda x: x.quantile(p.vol_percentile_high))
        vol_filter = (vol_series >= vol_q_low) & (vol_series <= vol_q_high)

        entries_long &= session_mask & atr_filter & vol_filter.to_numpy()
        entries_short = np.zeros_like(entries_short, dtype=bool)

        # Limitar a 2 trades por día y evitar solapamiento de posiciones
        entries_filtered = np.zeros_like(entries_long, dtype=bool)
        flat_until = -1
        daily_counts: Dict[pd.Timestamp.date, int] = {}

        for i, is_entry in enumerate(entries_long):
            if not is_entry:
                continue

            if i <= flat_until:
                continue

            day = idx_local[i].date()
            count_today = daily_counts.get(day, 0)
            if count_today >= 2:
                continue

            entries_filtered[i] = True
            daily_counts[day] = count_today + 1
            flat_until = i + p.max_holding_bars - 1

        entries_long = entries_filtered

        signals = np.zeros(n, dtype=np.int8)
        signals[entries_long] = 1
        signals[entries_short] = -1

        # 8) SL / TP / duración
        initial_stop_loss = np.full(n, np.nan, dtype=float)
        take_profit = np.full(n, np.nan, dtype=float)
        time_stop_bars = np.zeros(n, dtype=int)

        atr_entry = atr
        atr_stop_dist = self.params.atr_stop_mult * atr_entry
        structure_buffer = self.params.structure_buffer_atr * atr_entry

        swing_low = pd.Series(low).rolling(self.params.structure_stop_lookback, min_periods=1).min().to_numpy()
        swing_high = pd.Series(h).rolling(self.params.structure_stop_lookback, min_periods=1).max().to_numpy()

        atr_sl_long = c - atr_stop_dist
        atr_sl_short = c + atr_stop_dist

        structure_sl_long = swing_low - structure_buffer
        structure_sl_short = swing_high + structure_buffer

        sl_long = np.minimum(atr_sl_long, structure_sl_long)
        sl_short = np.maximum(atr_sl_short, structure_sl_short)

        rr = self.params.atr_tp_mult
        tp_long = c + rr * (c - sl_long)
        tp_short = c - rr * (sl_short - c)

        # Break-even tras movimiento a favor
        future_max_high = self._forward_rolling_max(h, p.breakeven_lookahead)
        future_min_low = self._forward_rolling_min(low, p.breakeven_lookahead)

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
            "atr_intraday_median": atr_intraday_median.tolist(),
            "atr_filter": atr_filter.tolist(),
            "vol_q_low": vol_q_low.tolist(),
            "vol_q_high": vol_q_high.tolist(),
            "vol_filter": vol_filter.tolist(),
            "session_mask": session_mask.tolist(),
            "entries_long": entries_long.tolist(),
            "entries_short": entries_short.tolist(),
            "daily_entry_counts": {str(day): count for day, count in daily_counts.items()},
            "initial_stop_loss": initial_stop_loss.tolist(),
            "take_profit": take_profit.tolist(),
            "time_stop_bars": time_stop_bars.tolist(),
            "atr": atr.tolist(),
            "atr_source": "external" if external_atr is not None else "local",
            "atr_timeframe": p.atr_timeframe,
            "atr_timeframe_period": p.atr_timeframe_period,
        }

        return StrategyResult(signals=signals, meta=meta_summary)
