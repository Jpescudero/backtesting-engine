"""Opening Sweep V4 strategy compatible with the backtesting engine."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd

from src.data.feeds import OHLCVArrays
from src.strategies.base import SignalEntry, Strategy, StrategyResult


@dataclass
class OpeningSweepV4Params:
    """Parámetros configurables de la estrategia Opening Sweep V4."""

    wick_factor: float = 1.5
    atr_percentile: float = 0.5
    volume_percentile: float = 0.4
    sl_buffer_atr: float = 0.3
    sl_buffer_relative: float = 0.1
    tp_multiplier: float = 1.2
    max_horizon: int = 30
    session_windows: Tuple[Tuple[int, int], ...] = (
        (8 * 60 + 50, 10 * 60),
        (15 * 60 + 20, 16 * 60 + 30),
    )
    max_trades_per_day: int = 2
    trading_timezone: str = "Europe/Madrid"


DEFAULTS: Dict[str, Any] = asdict(OpeningSweepV4Params())


class OpeningSweepV4(Strategy):
    """Opening sweep strategy with precomputed signals and ATR buffers."""

    def __init__(self, config: Mapping[str, Any] | None = None) -> None:
        if isinstance(config, OpeningSweepV4Params):
            config_dict = asdict(config)
        else:
            config_dict = dict(config or {})

        super().__init__(config_dict)

        if isinstance(config, OpeningSweepV4Params):
            self.params = config
        else:
            merged = DEFAULTS | config_dict
            self.params = OpeningSweepV4Params(**merged)
        self.signals: np.ndarray | None = None
        self.atr: np.ndarray | None = None
        self.atr_norm: np.ndarray | None = None
        self.opens: np.ndarray | None = None
        self.highs: np.ndarray | None = None
        self.lows: np.ndarray | None = None
        self.closes: np.ndarray | None = None
        self.volumes: np.ndarray | None = None
        self.daily_entry_counts: Dict[str, int] = {}

    def preload(self, df: pd.DataFrame) -> None:
        """Precompute ATR, normalized ATR, and sweep entry signals."""

        self.opens = df["open"].to_numpy(dtype=float)
        self.highs = df["high"].to_numpy(dtype=float)
        self.lows = df["low"].to_numpy(dtype=float)
        self.closes = df["close"].to_numpy(dtype=float)
        self.volumes = df["volume"].to_numpy(dtype=float)

        self.atr = self._compute_atr(self.highs, self.lows, self.closes)
        self.atr_norm = self._normalize_atr(self.atr)

        self.signals = self._precalc_signals(
            o=self.opens,
            c=self.closes,
            lows=self.lows,
            v=self.volumes,
            atr_norm=self.atr_norm,
            ts=df.index.view(np.int64),
        )

    def generate_signals(self, idx: int, row: Dict[str, Any]) -> SignalEntry | None:
        """Return a long entry when the precomputed signal is active."""

        if self.signals is None:
            return None

        if self.signals[idx] == 1 and not self.position.is_open:
            return SignalEntry(direction="long", size=1.0)
        return None

    def on_fill(self, trade: Any) -> None:
        """Set stop loss and take profit once the trade is filled."""

        if self.atr is None or self.atr_norm is None:
            return

        idx = trade.entry_index
        entry = trade.entry_price

        sl, tp = self.compute_sl_tp(idx, entry)
        self.set_stop_loss(price=sl)
        self.set_take_profit(price=tp)

    def compute_sl_tp(self, idx: int, entry: float) -> tuple[float, float]:
        """Compute stop-loss and take-profit levels for the given entry."""

        if self.lows is None or self.atr is None or self.atr_norm is None:
            raise ValueError("Strategy not preloaded with market data")

        low_sweep = float(self.lows[idx])
        atr_value = float(self.atr[idx])
        atr_norm_value = float(self.atr_norm[idx])

        sl_buffer_atr = float(self.params.sl_buffer_atr)
        sl_buffer_relative = float(self.params.sl_buffer_relative)
        tp_multiplier = float(self.params.tp_multiplier)

        buffer_total = sl_buffer_atr * atr_value + (sl_buffer_relative * atr_norm_value * atr_value)
        sl = low_sweep - buffer_total
        if sl >= entry:
            sl = entry - abs(buffer_total) - 1e-8

        tp = entry + tp_multiplier * (entry - sl)
        if tp <= entry:
            tp = entry + abs(tp_multiplier) * max(entry - sl, 1e-8)

        return sl, tp

    def generate_strategy_result(self, data: OHLCVArrays) -> StrategyResult:
        """Compute signals and SL/TP arrays compatible with the backtest engine."""

        o = np.asarray(data.o, dtype=float)
        h = np.asarray(data.h, dtype=float)
        lows = np.asarray(data.low, dtype=float)
        c = np.asarray(data.c, dtype=float)
        v = np.asarray(data.v, dtype=float)

        self.opens = o
        self.highs = h
        self.lows = lows
        self.closes = c
        self.volumes = v

        self.atr = self._compute_atr(h, lows, c)
        self.atr_norm = self._normalize_atr(self.atr)

        signals = self._precalc_signals(
            o=o, c=c, lows=lows, v=v, atr_norm=self.atr_norm, ts=np.asarray(data.ts)
        )

        stop_losses = np.full_like(c, np.nan, dtype=float)
        take_profits = np.full_like(c, np.nan, dtype=float)

        if signals.size:
            buffer_total = self.params.sl_buffer_atr * self.atr + (
                self.params.sl_buffer_relative * self.atr_norm * self.atr
            )
            sl_candidates = lows - buffer_total

            entry_prices = c
            sl_candidates = np.where(
                sl_candidates >= entry_prices, entry_prices - 1e-8, sl_candidates
            )
            tp_candidates = entry_prices + self.params.tp_multiplier * (
                entry_prices - sl_candidates
            )
            tp_candidates = np.where(
                tp_candidates <= entry_prices,
                entry_prices
                + self.params.tp_multiplier * np.maximum(entry_prices - sl_candidates, 1e-8),
                tp_candidates,
            )

            entries_mask = signals == 1
            stop_losses[entries_mask] = sl_candidates[entries_mask]
            take_profits[entries_mask] = tp_candidates[entries_mask]

        meta = {
            "params": asdict(self.params),
            "atr": self.atr.tolist(),
            "atr_norm": self.atr_norm.tolist(),
            "initial_stop_loss": stop_losses.tolist(),
            "take_profit": take_profits.tolist(),
            "session_windows": self.params.session_windows,
            "session_mask": getattr(self, "_session_mask", []).tolist()
            if hasattr(self, "_session_mask")
            else [],
            "daily_entry_counts": self.daily_entry_counts,
        }

        return StrategyResult(signals=signals.astype(np.int8), meta=meta)

    def _normalize_atr(self, atr: np.ndarray) -> np.ndarray:
        atr_mean = pd.Series(atr).rolling(1000, min_periods=1).mean().to_numpy(dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            atr_norm = np.nan_to_num(atr / atr_mean, nan=0.0, posinf=0.0, neginf=0.0)
        return atr_norm

    def _compute_atr(
        self, highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, period: int = 20
    ) -> np.ndarray:
        """Compute a simple ATR using rolling mean of true range."""

        n = len(closes)
        if n == 0:
            return np.zeros(0, dtype=float)

        prev_close = np.roll(closes, 1)
        prev_close[0] = closes[0]
        tr = np.maximum(
            highs - lows, np.maximum(np.abs(highs - prev_close), np.abs(lows - prev_close))
        )
        atr_series = pd.Series(tr).rolling(period, min_periods=1).mean()
        atr_values = atr_series.to_numpy(dtype=float)
        return np.nan_to_num(atr_values, nan=0.0, posinf=0.0, neginf=0.0)

    def _precalc_signals(
        self,
        *,
        o: np.ndarray,
        c: np.ndarray,
        lows: np.ndarray,
        v: np.ndarray,
        atr_norm: np.ndarray,
        ts: Sequence[int] | np.ndarray,
    ) -> np.ndarray:
        """Precompute sweep entry signals following the V4 heuristic."""

        n = len(c)
        signals = np.zeros(n, dtype=np.int8)
        if n == 0:
            return signals

        idx_local = pd.to_datetime(np.asarray(ts), unit="ns", utc=True).tz_convert(
            self.params.trading_timezone
        )
        minutes_in_day = idx_local.hour * 60 + idx_local.minute
        session_mask = self._session_filter(minutes_in_day)

        atr_threshold = float(np.quantile(atr_norm, self.params.atr_percentile))
        vol_threshold = float(np.quantile(v, self.params.volume_percentile))

        for i in range(2, n):
            body = abs(c[i] - o[i])
            wick = (o[i] - lows[i]) if c[i] >= o[i] else (c[i] - lows[i])
            wick_ratio = wick / (body + 1e-12)

            if wick_ratio < self.params.wick_factor:
                continue
            if v[i] < vol_threshold:
                continue
            if atr_norm[i] < atr_threshold:
                continue
            if not (c[i - 1] < o[i - 1] and c[i - 2] < o[i - 2]):
                continue

            mid = lows[i] + 0.5 * wick
            if c[i] < mid:
                continue

            signals[i] = 1

        signals = signals * session_mask.astype(np.int8)
        signals, daily_counts = self._apply_daily_limits(
            signals=signals, idx_local=idx_local, max_horizon=self.params.max_horizon
        )
        self._session_mask = session_mask
        self.daily_entry_counts = daily_counts

        return signals

    def _session_filter(self, minutes_in_day: np.ndarray) -> np.ndarray:
        if minutes_in_day.size == 0:
            return np.zeros(0, dtype=bool)

        mask = np.zeros_like(minutes_in_day, dtype=bool)
        for start, end in self.params.session_windows:
            mask |= (minutes_in_day >= start) & (minutes_in_day <= end)
        return mask

    def _apply_daily_limits(
        self, *, signals: np.ndarray, idx_local: pd.DatetimeIndex, max_horizon: int
    ) -> tuple[np.ndarray, Dict[str, int]]:
        filtered = np.zeros_like(signals, dtype=np.int8)
        flat_until = -1
        daily_counts: Dict[str, int] = {}

        for i, signal in enumerate(signals):
            if signal != 1:
                continue

            if i <= flat_until:
                continue

            day_str = str(idx_local[i].date())
            count_today = daily_counts.get(day_str, 0)
            if self.params.max_trades_per_day > 0 and count_today >= self.params.max_trades_per_day:
                continue

            filtered[i] = 1
            daily_counts[day_str] = count_today + 1
            flat_until = i + max(max_horizon, 1) - 1

        return filtered, daily_counts
