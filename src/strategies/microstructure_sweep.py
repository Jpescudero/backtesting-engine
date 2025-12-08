from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from src.data.feeds import OHLCVArrays
from src.strategies.base import StrategyResult


@dataclass
class SweepParams:
    # Timeframes / ATR
    ema_short: int = 20
    ema_long: int = 50
    atr_period: int = 20
    atr_timeframe: Optional[str] = "1m"
    atr_timeframe_period: int = 10

    # Barrida (stop-hunt)
    sweep_lookback: int = 15
    min_sweep_break_atr: float = 0.3
    min_lower_wick_body_ratio: float = 1.5
    min_sweep_range_atr: float = 0.5

    # Confirmación (absorción)
    confirm_body_atr: float = 0.35
    confirm_close_above_mid: bool = True

    # Volumen
    volume_period: int = 20
    min_rvol: float = 1.0
    vol_percentile_min: float = 0.80
    vol_percentile_max: float = 1.00

    # Filtros adicionales
    use_trend_filter: bool = True
    max_atr_mult_intraday: float = 3.0
    max_trades_per_day: int = 4
    max_holding_bars: int = 60

    # SL/TP
    atr_stop_mult: float = 0.2
    rr_multiple: float = 2.5


class StrategyMicrostructureSweep:
    def __init__(self, **kwargs: Any) -> None:
        self.params = SweepParams(**kwargs)

    # Helpers reutilizados de microstructure_reversal
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
        if indicator.shape[0] == 0:
            return np.zeros_like(target_ts, dtype=float)

        source_index = pd.to_datetime(source_ts, utc=True)
        target_index = pd.to_datetime(target_ts, utc=True)

        aligned = (
            pd.Series(indicator, index=source_index)
            .reindex(target_index, method="ffill")
            .to_numpy()
        )
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

    def compute_lower_timeframe_atr(
        self, lower_data: OHLCVArrays, target_ts: np.ndarray
    ) -> np.ndarray:
        atr_lower = self._atr(
            h=np.asarray(lower_data.h),
            low=np.asarray(lower_data.low),
            c=np.asarray(lower_data.c),
            period=self.params.atr_timeframe_period,
        )

        return self._align_indicator_to_target_ts(
            indicator=atr_lower, source_ts=lower_data.ts, target_ts=target_ts
        )

    def generate_signals(
        self, data: OHLCVArrays, external_atr: Optional[np.ndarray] = None
    ) -> StrategyResult:
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

        idx_local = pd.to_datetime(ts, utc=True).tz_convert("Europe/Madrid")
        minutes_in_day = idx_local.hour * 60 + idx_local.minute
        session_mask = ((minutes_in_day >= 8 * 60 + 50) & (minutes_in_day <= 10 * 60)) | (
            (minutes_in_day >= 15 * 60 + 20) & (minutes_in_day <= 16 * 60 + 30)
        )
        day_index = idx_local.normalize()

        if p.use_trend_filter:
            ema_short = self._ema(c, p.ema_short)
            ema_long = self._ema(c, p.ema_long)
            uptrend_mask = ema_short > ema_long
        else:
            uptrend_mask = np.ones_like(c, dtype=bool)
            ema_short = ema_long = np.zeros_like(c, dtype=float)

        if external_atr is not None:
            if external_atr.shape[0] != n:
                raise ValueError(
                    "external_atr debe tener la misma longitud que los datos del timeframe objetivo"
                )
            atr = external_atr
            atr_source = "external"
        else:
            atr = self._atr(h=h, low=low, c=c, period=p.atr_period)
            atr_source = "local"

        swing_low_prev = self._rolling_min(low, p.sweep_lookback)
        swing_low_prev = np.roll(swing_low_prev, 1)
        swing_low_prev[0] = low[0]

        break_amount = swing_low_prev - low
        with np.errstate(divide="ignore", invalid="ignore"):
            break_atr = break_amount / atr
        broke_prev_low = np.isfinite(break_atr) & (break_atr >= p.min_sweep_break_atr)

        range_ = h - low
        body = np.abs(c - o)
        tiny = 1e-12

        lower_wick = np.where(c >= o, o - low, c - low)
        wick_body_cond = lower_wick >= p.min_lower_wick_body_ratio * np.maximum(body, tiny)

        with np.errstate(divide="ignore", invalid="ignore"):
            range_atr = range_ / atr
        range_cond = np.isfinite(range_atr) & (range_atr >= p.min_sweep_range_atr)

        mid_level = low + 0.5 * range_
        close_recovered = c >= mid_level

        sweep_long = broke_prev_low & wick_body_cond & range_cond & close_recovered & uptrend_mask

        body_next = np.abs(np.roll(c, -1) - np.roll(o, -1))
        body_signed_next = np.roll(c, -1) - np.roll(o, -1)

        confirm_long = sweep_long & (body_signed_next > 0)
        with np.errstate(divide="ignore", invalid="ignore"):
            body_next_atr = body_next / atr
        confirm_long &= np.isfinite(body_next_atr) & (body_next_atr >= p.confirm_body_atr)
        if p.confirm_close_above_mid:
            confirm_long &= np.roll(c, -1) >= mid_level

        confirm_long[-1] = False

        entries_long = np.roll(confirm_long, 1)
        entries_long[0] = False
        entries_long[-1] = False

        entries_short = np.zeros_like(entries_long, dtype=bool)

        vol_mean = pd.Series(v).rolling(p.volume_period, min_periods=1).mean().to_numpy()
        with np.errstate(divide="ignore", invalid="ignore"):
            rvol = np.divide(v, vol_mean, out=np.ones_like(v, dtype=float), where=vol_mean > 0)

        vol_series = pd.Series(v, dtype=float)
        vol_q_min = vol_series.groupby(day_index).transform(
            lambda x: x.quantile(p.vol_percentile_min)
        )
        vol_q_max = vol_series.groupby(day_index).transform(
            lambda x: x.quantile(p.vol_percentile_max)
        )
        vol_filter = (vol_series >= vol_q_min) & (vol_series <= vol_q_max)

        high_rvol = rvol >= p.min_rvol
        volume_mask = high_rvol & vol_filter.to_numpy()

        atr_series = pd.Series(atr)
        atr_intraday_median = atr_series.groupby(day_index).transform("median")
        atr_filter = np.isfinite(atr) & np.isfinite(atr_intraday_median)
        atr_filter &= atr <= p.max_atr_mult_intraday * atr_intraday_median.to_numpy()

        entries_long &= session_mask & volume_mask & atr_filter

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
            if count_today >= p.max_trades_per_day:
                continue

            entries_filtered[i] = True
            daily_counts[day] = count_today + 1
            flat_until = i + p.max_holding_bars - 1

        entries_long = entries_filtered

        signals = np.zeros(n, dtype=np.int8)
        signals[entries_long] = 1
        signals[entries_short] = -1

        sweep_low_for_entry = np.zeros(n, dtype=float)
        sweep_low_for_entry[entries_long] = np.roll(low, 1)[entries_long]

        initial_stop_loss = np.full(n, np.nan, dtype=float)
        take_profit = np.full(n, np.nan, dtype=float)
        time_stop_bars = np.zeros(n, dtype=int)

        atr_entry = atr
        sl_buffer = p.atr_stop_mult * atr_entry

        sl_long = sweep_low_for_entry - sl_buffer
        rr = p.rr_multiple
        tp_long = c + rr * (c - sl_long)

        initial_stop_loss[entries_long] = sl_long[entries_long]
        take_profit[entries_long] = tp_long[entries_long]
        time_stop_bars[entries_long] = p.max_holding_bars

        meta_summary: Dict[str, Any] = {
            "params": asdict(p),
            "n_entries": int(entries_long.sum()),
            "sweep_mask_long": sweep_long.tolist(),
            "confirm_mask_long": confirm_long.tolist(),
            "volume_mask": volume_mask.tolist(),
            "atr_filter": atr_filter.tolist(),
            "session_mask": session_mask.tolist(),
            "entries_long": entries_long.tolist(),
            "entries_short": entries_short.tolist(),
            "daily_entry_counts": {str(day): count for day, count in daily_counts.items()},
            "initial_stop_loss": initial_stop_loss.tolist(),
            "take_profit": take_profit.tolist(),
            "time_stop_bars": time_stop_bars.tolist(),
            "atr": atr.tolist(),
            "atr_source": atr_source,
            "atr_timeframe": p.atr_timeframe,
            "atr_timeframe_period": p.atr_timeframe_period,
        }

        return StrategyResult(signals=signals, meta=meta_summary)
