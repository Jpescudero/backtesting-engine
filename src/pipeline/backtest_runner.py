from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any, Dict, Literal, Mapping, Optional, Sequence, Union, overload

import matplotlib
import numpy as np

from src.analytics.reporting import equity_to_series, trades_to_dataframe
from src.config.paths import REPORTS_DIR
from src.data.feeds import NPZOHLCVFeed, OHLCVArrays
from src.engine.core import BacktestConfig, BacktestSnapshot, run_backtest_with_signals
from src.pipeline.data_pipeline import prepare_npz_dataset
from src.pipeline.reporting import (
    BacktestReports,
    compute_analytics,
    generate_main_plots,
    generate_report_files,
    generate_trade_plots,
    _strategy_suffix,
)
from src.strategies.microstructure_reversal import StrategyMicrostructureReversal
from src.strategies.microstructure_sweep import StrategyMicrostructureSweep, SweepParams
from src.strategies.opening_sweep_v4 import OpeningSweepV4, OpeningSweepV4Params
from src.utils.seeding import seed_everything
from src.utils.timing import timed_step

logger = logging.getLogger(__name__)


def _load_snapshot(path: Path | str) -> BacktestSnapshot:
    snapshot_path = Path(path)
    if not snapshot_path.exists():
        raise FileNotFoundError(f"Snapshot no encontrado en {snapshot_path}")

    with open(snapshot_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    if isinstance(payload, list):
        if not payload:
            raise ValueError(f"El snapshot {snapshot_path} está vacío")
        payload = payload[-1]

    return BacktestSnapshot.from_dict(payload)


def _save_snapshots(path: Path, snapshots) -> None:
    if not snapshots:
        return

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump([s.to_dict() for s in snapshots], f, ensure_ascii=False, indent=2)


def _build_run_metadata(
    *, config: BacktestRunConfig, seeds: Mapping[str, object], snapshot_path: Path | None
) -> Dict[str, object]:
    strategy_params = config.strategy_params
    strategy_params_dict = (
        asdict(strategy_params)
        if is_dataclass(strategy_params) and not isinstance(strategy_params, type)
        else _params_mapping(strategy_params)
    )

    return {
        "symbol": config.symbol,
        "timeframe": config.timeframe,
        "strategy_name": config.strategy_name,
        "strategy_params": strategy_params_dict,
        "backtest_config": asdict(config.backtest_config),
        "train_years": config.train_years,
        "test_years": config.test_years,
        "use_test_years": config.use_test_years,
        "seed": config.seed,
        "seeds": dict(seeds),
        "snapshot_interval": config.snapshot_interval,
        "snapshot_path": str(snapshot_path) if snapshot_path else None,
        "resume_snapshot": str(config.resume_snapshot) if config.resume_snapshot else None,
    }


def _save_run_metadata(path: Path, meta: Mapping[str, object]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


def _effective_strategy_name(config_name: str, meta: Optional[Mapping[str, object]]) -> str:
    if meta and isinstance(meta, Mapping):
        meta_name = meta.get("strategy_name")
        if meta_name:
            return str(meta_name)
    return config_name


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


StrategyParamsType = Union[StrategyParams, SweepParams, OpeningSweepV4Params]


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
    seed: Optional[int] = None
    snapshot_interval: Optional[int] = None
    snapshot_path: Optional[Path] = None
    resume_snapshot: Optional[Path] = None
    run_metadata_path: Optional[Path] = None
    replay_metadata: Optional[Path] = None
    strategy_params: StrategyParamsType = field(default_factory=StrategyParams)
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
    trade_level_stats: Dict
    timings: Dict[str, float]
    reports: BacktestReports


def _params_mapping(params: object) -> Dict[str, Any]:
    if is_dataclass(params) and not isinstance(params, type):
        return {k: v for k, v in asdict(params).items()}
    if isinstance(params, Mapping):
        return {str(k): v for k, v in params.items()}
    return {
        key: getattr(params, key)
        for key in dir(params)
        if not key.startswith("_") and not callable(getattr(params, key))
    }


@overload
def _coerce_strategy_params(
    params: StrategyParamsType | Mapping[str, object],
    *,
    strategy_name: Literal["opening_sweep_v4"],
) -> OpeningSweepV4Params: ...


@overload
def _coerce_strategy_params(
    params: StrategyParamsType | Mapping[str, object],
    *,
    strategy_name: Literal["microstructure_sweep"],
) -> SweepParams: ...


@overload
def _coerce_strategy_params(
    params: StrategyParamsType | Mapping[str, object],
    *,
    strategy_name: Literal["microstructure_reversal"],
) -> StrategyParams: ...


@overload
def _coerce_strategy_params(
    params: StrategyParamsType | Mapping[str, object], *, strategy_name: str
) -> StrategyParamsType: ...


def _coerce_strategy_params(
    params: StrategyParamsType | Mapping[str, object], *, strategy_name: str
) -> StrategyParamsType:
    params_dict: Dict[str, Any] = _params_mapping(params)
    if strategy_name == "opening_sweep_v4":
        if isinstance(params, OpeningSweepV4Params):
            return params
        return OpeningSweepV4Params(**params_dict)
    if strategy_name == "microstructure_sweep":
        if isinstance(params, SweepParams):
            return params
        return SweepParams(**params_dict)

    if isinstance(params, StrategyParams):
        return params
    return StrategyParams(**params_dict)


def _validated_years(years: Optional[Sequence[int]], *, label: str) -> Sequence[int]:
    if years is None:
        raise ValueError(f"'{label}' está a None pero se esperaba una lista de años")
    return years


def _configure_matplotlib(headless: bool) -> None:
    if headless:
        matplotlib.use("Agg")


def run_single_backtest(config: BacktestRunConfig) -> BacktestArtifacts:
    _configure_matplotlib(config.headless)

    config.strategy_params = _coerce_strategy_params(
        config.strategy_params, strategy_name=config.strategy_name
    )
    strategy_params = config.strategy_params
    use_test_years = config.use_test_years or (
        config.test_years is not None and not config.train_years
    )
    seeds = seed_everything(config.seed)

    resume_snapshot: BacktestSnapshot | None = None
    if config.resume_snapshot:
        resume_snapshot = _load_snapshot(config.resume_snapshot)

    if config.strategy_name not in {
        "microstructure_reversal",
        "microstructure_sweep",
        "opening_sweep_v4",
    }:
        raise ValueError(
            "Solo se soportan las estrategias 'microstructure_reversal', "
            "'microstructure_sweep' y 'opening_sweep_v4'"
        )

    timings: Dict[str, float] = {}
    base_reports_dir = config.reports_dir or (REPORTS_DIR / config.symbol).resolve()

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

        if use_test_years:
            data = feed.load_years(_validated_years(config.test_years, label="test_years"))
            logger.info("Usando años de prueba: %s", config.test_years)
        elif config.train_years is not None:
            data = feed.load_years(config.train_years)
            logger.info("Usando años de entrenamiento: %s", config.train_years)
        else:
            data = feed.load_all()

        atr_tf = getattr(strategy_params, "atr_timeframe", None)
        if atr_tf and atr_tf != config.timeframe:
            atr_feed = NPZOHLCVFeed(symbol=config.symbol, timeframe=atr_tf)
            if use_test_years:
                atr_data = atr_feed.load_years(
                    _validated_years(config.test_years, label="test_years")
                )
            elif config.train_years is not None:
                atr_data = atr_feed.load_years(config.train_years)
            else:
                atr_data = atr_feed.load_all()

    with timed_step(timings, "03_generar_senales_estrategia"):
        strategy: StrategyMicrostructureReversal | StrategyMicrostructureSweep | OpeningSweepV4
        if config.strategy_name == "microstructure_reversal":
            assert isinstance(strategy_params, StrategyParams)
            strategy = StrategyMicrostructureReversal(
                ema_short=strategy_params.ema_short,
                ema_long=strategy_params.ema_long,
                atr_period=strategy_params.atr_period,
                atr_timeframe=strategy_params.atr_timeframe,
                atr_timeframe_period=strategy_params.atr_timeframe_period,
                min_pullback_atr=strategy_params.min_pullback_atr,
                max_pullback_atr=strategy_params.max_pullback_atr,
                max_pullback_bars=strategy_params.max_pullback_bars,
                exhaustion_close_min=strategy_params.exhaustion_close_min,
                exhaustion_close_max=strategy_params.exhaustion_close_max,
                exhaustion_body_max_ratio=strategy_params.exhaustion_body_max_ratio,
                shift_body_atr=strategy_params.shift_body_atr,
                structure_break_lookback=strategy_params.structure_break_lookback,
            )
        elif config.strategy_name == "microstructure_sweep":
            assert isinstance(strategy_params, SweepParams)
            strategy = StrategyMicrostructureSweep(
                ema_short=strategy_params.ema_short,
                ema_long=strategy_params.ema_long,
                atr_period=strategy_params.atr_period,
                atr_timeframe=strategy_params.atr_timeframe,
                atr_timeframe_period=strategy_params.atr_timeframe_period,
                sweep_lookback=strategy_params.sweep_lookback,
                min_sweep_break_atr=strategy_params.min_sweep_break_atr,
                min_lower_wick_body_ratio=strategy_params.min_lower_wick_body_ratio,
                min_sweep_range_atr=strategy_params.min_sweep_range_atr,
                confirm_body_atr=strategy_params.confirm_body_atr,
                confirm_close_above_mid=strategy_params.confirm_close_above_mid,
                volume_period=strategy_params.volume_period,
                min_rvol=strategy_params.min_rvol,
                vol_percentile_min=strategy_params.vol_percentile_min,
                vol_percentile_max=strategy_params.vol_percentile_max,
                use_trend_filter=strategy_params.use_trend_filter,
                max_atr_mult_intraday=strategy_params.max_atr_mult_intraday,
                max_trades_per_day=strategy_params.max_trades_per_day,
                max_holding_bars=strategy_params.max_holding_bars,
                atr_stop_mult=strategy_params.atr_stop_mult,
                rr_multiple=strategy_params.rr_multiple,
            )
        else:
            assert isinstance(strategy_params, OpeningSweepV4Params)
            strategy = OpeningSweepV4(config=strategy_params)

        if config.strategy_name == "opening_sweep_v4":
            config.backtest_config.max_bars_in_trade = strategy_params.max_horizon
            strat_res = strategy.generate_strategy_result(data)
        else:
            atr_override = None
            if atr_data is not None:
                atr_override = strategy.compute_lower_timeframe_atr(
                    lower_data=atr_data,
                    target_ts=data.ts,
                )

            strat_res = strategy.generate_signals(data, external_atr=atr_override)
    strategy_label = _effective_strategy_name(
        config.strategy_name, getattr(strat_res, "meta", None)
    )
    reports_dir = (base_reports_dir / _strategy_suffix(strategy_label)).resolve()
    reports_dir.mkdir(parents=True, exist_ok=True)
    n_signals = int((strat_res.signals != 0).sum())
    logger.info("Estrategia %s: %s señales generadas", strategy_label, n_signals)

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
            snapshot_interval=config.snapshot_interval,
            resume_from=resume_snapshot,
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
        (
            equity_series,
            trades_df,
            equity_stats,
            trade_stats,
            level_stats,
        ) = compute_analytics(result, data, equity_series=equity_series, trades_df=trades_df)

    report_paths = BacktestReports(
        equity_stats=equity_stats, trade_stats=trade_stats, trade_level_stats=level_stats
    )

    snapshot_path: Optional[Path] = None
    if result.snapshots:
        snapshot_path = config.snapshot_path or (reports_dir / "snapshots.json")
        _save_snapshots(snapshot_path, result.snapshots)

    run_meta = _build_run_metadata(
        config=config,
        seeds=seeds,
        snapshot_path=snapshot_path,
    )
    meta_path = config.run_metadata_path or (reports_dir / "run_metadata.json")
    _save_run_metadata(meta_path, run_meta)
    if config.generate_report_files:
        with timed_step(timings, "07_reportes_excel_json"):
            excel_path, json_path, levels_path = generate_report_files(
                reports_dir=reports_dir,
                symbol=config.symbol,
                strategy_name=strategy_label,
                equity_series=equity_series,
                trades_df=trades_df,
                equity_stats=equity_stats,
                trade_stats=trade_stats,
                level_stats=level_stats,
                meta={
                    **getattr(strat_res, "meta", {}),
                    "run_metadata": run_meta,
                },
            )
        report_paths.excel_path = excel_path
        report_paths.json_path = json_path
        report_paths.levels_report_path = levels_path
    else:
        timings["07_reportes_excel_json"] = 0.0

    if config.generate_main_plots:
        with timed_step(timings, "08_plots_equity_trades_mes"):
            equity_plot_path = generate_main_plots(
                result=result,
                data=data,
                strategy_name=strategy_label,
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
                strategy_name=strategy_label,
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
        trade_level_stats=level_stats,
        timings=timings,
        reports=report_paths,
    )


def load_run_config_from_metadata(path: Path | str) -> BacktestRunConfig:
    meta_path = Path(path)
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    strategy_name = meta.get("strategy_name", "microstructure_reversal")
    strat_params_dict = meta.get("strategy_params", {}) or {}
    strategy_params: StrategyParamsType
    if strategy_name == "microstructure_sweep":
        strategy_params = SweepParams(**strat_params_dict)
    elif strategy_name == "opening_sweep_v4":
        strategy_params = OpeningSweepV4Params(**strat_params_dict)
    else:
        strategy_params = StrategyParams(**strat_params_dict)

    backtest_cfg = BacktestConfig(**(meta.get("backtest_config", {}) or {}))

    return BacktestRunConfig(
        symbol=meta.get("symbol", "NDXm"),
        timeframe=meta.get("timeframe", "1m"),
        strategy_name=strategy_name,
        train_years=meta.get("train_years"),
        test_years=meta.get("test_years"),
        use_test_years=bool(meta.get("use_test_years", False)),
        seed=meta.get("seed"),
        snapshot_interval=meta.get("snapshot_interval"),
        snapshot_path=Path(meta["snapshot_path"]) if meta.get("snapshot_path") else None,
        resume_snapshot=Path(meta["resume_snapshot"]) if meta.get("resume_snapshot") else None,
        strategy_params=strategy_params,
        backtest_config=backtest_cfg,
    )
