from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

import matplotlib

from src.analytics.reporting import equity_to_series, trades_to_dataframe
from src.config.paths import REPORTS_DIR
from src.data.feeds import NPZOHLCVFeed
from src.engine.core import BacktestConfig, run_backtest_with_signals
from src.pipeline.data_pipeline import prepare_npz_dataset
from src.pipeline.reporting import (
    BacktestReports,
    compute_analytics,
    generate_main_plots,
    generate_report_files,
    generate_trade_plots,
)
from src.strategies.barrida_apertura import StrategyBarridaApertura
from src.utils.timing import timed_step


logger = logging.getLogger(__name__)


@dataclass
class StrategyParams:
    volume_percentile: float = 80.0
    use_two_bearish_bars: bool = True


@dataclass
class BacktestRunConfig:
    symbol: str = "NDXm"
    timeframe: str = "1m"
    generate_report_files: bool = True
    generate_main_plots: bool = True
    generate_trade_plots: bool = True
    headless: bool = False
    reports_dir: Optional[Path] = None
    train_years: Optional[list[int]] = None
    test_years: Optional[list[int]] = None
    use_test_years: bool = False
    strategy_params: StrategyParams = field(default_factory=StrategyParams)
    backtest_config: BacktestConfig = field(
        default_factory=lambda: BacktestConfig(
            initial_cash=100_000.0,
            commission_per_trade=1.0,
            trade_size=1.0,
            slippage=0.0,
            sl_pct=0.01,
            tp_pct=0.02,
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

    timings: Dict[str, float] = {}
    reports_dir = config.reports_dir or (REPORTS_DIR / config.symbol).resolve()

    logger.info("Capital inicial configurado: %s", config.backtest_config.initial_cash)

    with timed_step(timings, "01_datos_preparacion"):
        bars_csv_path, npz_path = prepare_npz_dataset(config.symbol, timeframe=config.timeframe)
    logger.info("CSV 1m listo: %s", bars_csv_path)
    logger.info("NPZ listo: %s", npz_path)

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

    with timed_step(timings, "03_generar_senales_estrategia"):
        strategy = StrategyBarridaApertura(
            volume_percentile=config.strategy_params.volume_percentile,
            use_two_bearish_bars=config.strategy_params.use_two_bearish_bars,
        )
        strat_res = strategy.generate_signals(data)
    n_signals = int((strat_res.signals != 0).sum())
    logger.info("Estrategia Barrida: %s señales generadas", n_signals)

    with timed_step(timings, "04_backtest_motor"):
        result = run_backtest_with_signals(data, strat_res.signals, config=config.backtest_config)
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
                strategy_name="barrida_apertura",
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
                reports_dir=reports_dir,
                show=not config.headless,
            )
        report_paths.equity_path = equity_plot_path
    else:
        timings["08_plots_equity_trades_mes"] = 0.0

    if config.generate_trade_plots:
        with timed_step(timings, "09_plots_mejores_peores_trades"):
            trade_plot_path = generate_trade_plots(
                trades_df=trades_df,
                data=data,
                reports_dir=reports_dir,
                show=not config.headless,
            )
        report_paths.trade_plot_path = trade_plot_path
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
