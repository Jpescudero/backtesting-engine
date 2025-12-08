from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

import matplotlib
import numpy as np

from src.analytics.reporting import equity_to_series, trades_to_dataframe
from src.config.paths import REPORTS_DIR
from src.data.feeds import NPZOHLCVFeed, OHLCVArrays
from src.engine.core import BacktestConfig, run_backtest_with_signals
from src.pipeline.data_pipeline import prepare_npz_dataset
from src.pipeline.reporting import (
    BacktestReports,
    compute_analytics,
    generate_main_plots,
    generate_report_files,
    generate_trade_plots,
)
from src.strategies.microstructure_reversal import StrategyMicrostructureReversal
from src.strategies.microstructure_sweep import SweepParams, StrategyMicrostructureSweep
from src.utils.timing import timed_step


logger = logging.getLogger(__name__)


@dataclass
class StrategyParams:
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


@dataclass
class BacktestRunConfig:
    symbol: str = "NDXm"
    timeframe: str = "1m"
    strategy_name: str = "microstructure_reversal"
    generate_report_files: bool = True
    generate_main_plots: bool = True
    generate_trade_plots: bool = True
    headless: bool = False
    reports_dir: Optional[Path] = None
    train_years: Optional[list[int]] = None
    test_years: Optional[list[int]] = None
    use_test_years: bool = False
    strategy_params: object = field(default_factory=StrategyParams)
    backtest_config: BacktestConfig = field(
        default_factory=lambda: BacktestConfig(
            initial_cash=100_000.0,
            commission_per_trade=1.0,
            trade_size=1.0,
            min_trade_size=0.01,
            max_trade_size=100.0,
            risk_per_trade_pct=0.0025,
            atr_stop_mult=1.0,
            atr_tp_mult=2.0,
            slippage=0.0,
            sl_pct=0.01,
            tp_pct=0.02,
            point_value=1.0,
            max_bars_in_trade=60,
            entry_threshold=0.0,
        )
    )


@dataclass
class BacktestArtifacts:
    equity_series: object
    trades_df: object
    equity_stats: Dict
    trade_stats: Dict
    timings: Dict[str, float]
    reports: BacktestReports


def _configure_matplotlib(headless: bool) -> None:
    if headless:
        matplotlib.use("Agg")


def run_single_backtest(config: BacktestRunConfig) -> BacktestArtifacts:
    _configure_matplotlib(config.headless)

    if config.strategy_name not in {"microstructure_reversal", "microstructure_sweep"}:
        raise ValueError(
            "Solo se soportan las estrategias 'microstructure_reversal' y 'microstructure_sweep'"
        )

    timings: Dict[str, float] = {}
    reports_dir = config.reports_dir or (REPORTS_DIR / config.symbol).resolve()

    logger.info("Capital inicial configurado: %s", config.backtest_config.initial_cash)

    with timed_step(timings, "01_datos_preparacion"):
        bars_csv_path, npz_path = prepare_npz_dataset(config.symbol, timeframe=config.timeframe)
        atr_npz_path: Optional[Path] = None
        atr_tf = getattr(config.strategy_params, "atr_timeframe", None)
        if atr_tf and atr_tf != config.timeframe:
            _, atr_npz_path = prepare_npz_dataset(config.symbol, timeframe=atr_tf)
    logger.info("CSV 1m listo: %s", bars_csv_path)
    logger.info("NPZ listo: %s", npz_path)
    if atr_npz_path:
        logger.info("NPZ ATR (tf %s) listo: %s", atr_tf, atr_npz_path)

    atr_data: Optional[OHLCVArrays] = None
    with timed_step(timings, "02_carga_feed_npz"):
        feed = NPZOHLCVFeed(symbol=config.symbol, timeframe=config.timeframe)

        use_test_years = config.use_test_years or (
            config.test_years is not None and not config.train_years
        )

        if use_test_years and not config.test_years:
            raise ValueError("'use_test_years' está a True pero no se han definido test_years")

        if use_test_years:
            data = feed.load_years(config.test_years)
            logger.info("Usando años de prueba: %s", config.test_years)
        elif config.train_years:
            data = feed.load_years(config.train_years)
            logger.info("Usando años de entrenamiento: %s", config.train_years)
        else:
            data = feed.load_all()

        atr_tf = getattr(config.strategy_params, "atr_timeframe", None)
        if atr_tf and atr_tf != config.timeframe:
            atr_feed = NPZOHLCVFeed(symbol=config.symbol, timeframe=atr_tf)
            if use_test_years:
                atr_data = atr_feed.load_years(config.test_years)
            elif config.train_years:
                atr_data = atr_feed.load_years(config.train_years)
            else:
                atr_data = atr_feed.load_all()

    with timed_step(timings, "03_generar_senales_estrategia"):
        if config.strategy_name == "microstructure_reversal":
            strategy = StrategyMicrostructureReversal(
                ema_short=config.strategy_params.ema_short,
                ema_long=config.strategy_params.ema_long,
                atr_period=config.strategy_params.atr_period,
                atr_timeframe=config.strategy_params.atr_timeframe,
                atr_timeframe_period=config.strategy_params.atr_timeframe_period,
                min_pullback_atr=config.strategy_params.min_pullback_atr,
                max_pullback_atr=config.strategy_params.max_pullback_atr,
                max_pullback_bars=config.strategy_params.max_pullback_bars,
                exhaustion_close_min=config.strategy_params.exhaustion_close_min,
                exhaustion_close_max=config.strategy_params.exhaustion_close_max,
                exhaustion_body_max_ratio=config.strategy_params.exhaustion_body_max_ratio,
                shift_body_atr=config.strategy_params.shift_body_atr,
                structure_break_lookback=config.strategy_params.structure_break_lookback,
            )
        else:
            assert isinstance(config.strategy_params, SweepParams)
            strategy = StrategyMicrostructureSweep(
                ema_short=config.strategy_params.ema_short,
                ema_long=config.strategy_params.ema_long,
                atr_period=config.strategy_params.atr_period,
                atr_timeframe=config.strategy_params.atr_timeframe,
                atr_timeframe_period=config.strategy_params.atr_timeframe_period,
                sweep_lookback=config.strategy_params.sweep_lookback,
                min_sweep_break_atr=config.strategy_params.min_sweep_break_atr,
                min_lower_wick_body_ratio=config.strategy_params.min_lower_wick_body_ratio,
                min_sweep_range_atr=config.strategy_params.min_sweep_range_atr,
                confirm_body_atr=config.strategy_params.confirm_body_atr,
                confirm_close_above_mid=config.strategy_params.confirm_close_above_mid,
                volume_period=config.strategy_params.volume_period,
                min_rvol=config.strategy_params.min_rvol,
                vol_percentile_min=config.strategy_params.vol_percentile_min,
                vol_percentile_max=config.strategy_params.vol_percentile_max,
                use_trend_filter=config.strategy_params.use_trend_filter,
                max_atr_mult_intraday=config.strategy_params.max_atr_mult_intraday,
                max_trades_per_day=config.strategy_params.max_trades_per_day,
                max_holding_bars=config.strategy_params.max_holding_bars,
                atr_stop_mult=config.strategy_params.atr_stop_mult,
                rr_multiple=config.strategy_params.rr_multiple,
            )

        atr_override = None
        if atr_data is not None:
            atr_override = strategy.compute_lower_timeframe_atr(
                lower_data=atr_data,
                target_ts=data.ts,
            )

        strat_res = strategy.generate_signals(data, external_atr=atr_override)
    n_signals = int((strat_res.signals != 0).sum())
    logger.info("Estrategia %s: %s señales generadas", config.strategy_name, n_signals)

    with timed_step(timings, "04_backtest_motor"):
        atr_array = None
        if "atr" in strat_res.meta:
            atr_array = np.asarray(strat_res.meta["atr"], dtype=float)

        stop_losses = None
        take_profits = None
        if "initial_stop_loss" in strat_res.meta:
            stop_losses = np.asarray(strat_res.meta["initial_stop_loss"], dtype=float)
        if "take_profit" in strat_res.meta:
            take_profits = np.asarray(strat_res.meta["take_profit"], dtype=float)

        result = run_backtest_with_signals(
            data,
            strat_res.signals,
            atr=atr_array,
            stop_losses=stop_losses,
            take_profits=take_profits,
            config=config.backtest_config,
        )
    logger.info(
        "Cash final: %s | Posición final: %s | Número de trades: %s",
        result.cash,
        result.position,
        result.extra.get("n_trades", 0),
    )

    with timed_step(timings, "05_conversion_pandas"):
        equity_series = equity_to_series(result, data)
        trades_df = trades_to_dataframe(result, data)

    with timed_step(timings, "06_metricas"):
        equity_series, trades_df, equity_stats, trade_stats = compute_analytics(
            result, data, equity_series=equity_series, trades_df=trades_df
        )

    report_paths = BacktestReports(equity_stats=equity_stats, trade_stats=trade_stats)
    if config.generate_report_files:
        with timed_step(timings, "07_reportes_excel_json"):
            excel_path, json_path = generate_report_files(
                reports_dir=reports_dir,
                symbol=config.symbol,
                strategy_name=config.strategy_name,
                equity_series=equity_series,
                trades_df=trades_df,
                equity_stats=equity_stats,
                trade_stats=trade_stats,
                meta=getattr(strat_res, "meta", {}),
            )
        report_paths.excel_path = excel_path
        report_paths.json_path = json_path
    else:
        timings["07_reportes_excel_json"] = 0.0

    if config.generate_main_plots:
        with timed_step(timings, "08_plots_equity_trades_mes"):
            equity_plot_path = generate_main_plots(
                result=result,
                data=data,
                strategy_name=config.strategy_name,
                reports_dir=reports_dir,
                show=not config.headless,
            )
        report_paths.equity_path = equity_plot_path
    else:
        timings["08_plots_equity_trades_mes"] = 0.0

    if config.generate_trade_plots:
        with timed_step(timings, "09_plots_mejores_peores_trades"):
            best_path, worst_path = generate_trade_plots(
                trades_df=trades_df,
                data=data,
                strategy_name=config.strategy_name,
                reports_dir=reports_dir,
                show=not config.headless,
            )
        report_paths.best_trade_plot_path = best_path
        report_paths.worst_trade_plot_path = worst_path
    else:
        timings["09_plots_mejores_peores_trades"] = 0.0

    return BacktestArtifacts(
        equity_series=equity_series,
        trades_df=trades_df,
        equity_stats=equity_stats,
        trade_stats=trade_stats,
        timings=timings,
        reports=report_paths,
    )
