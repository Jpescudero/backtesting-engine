"""Batch runner para ejecutar múltiples configuraciones de una estrategia.

Permite construir trabajos a partir de un `BacktestRunConfig`, planificarlos
en paralelo con `multiprocessing.Pool` y exportar un ranking reproducible con
parámetros y métricas clave.
"""

from __future__ import annotations

import copy
import logging
import multiprocessing as mp
from collections.abc import Iterable, Mapping, MutableMapping, Sequence
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path

import pandas as pd

from src.config.paths import REPORTS_DIR
from src.pipeline.backtest_runner import (
    BacktestArtifacts,
    BacktestRunConfig,
    run_single_backtest,
)

logger = logging.getLogger(__name__)


@dataclass
class BatchJob:
    """Trabajo individual de backtest dentro de un lote."""

    job_id: str
    run_config: BacktestRunConfig
    param_overrides: Mapping[str, object]


@dataclass
class BatchJobOutcome:
    """Resultado de un trabajo de backtest ejecutado."""

    job: BatchJob
    artifacts: BacktestArtifacts


@dataclass
class RankingRule:
    """Regla de ordenación para el ranking reproducible."""

    column: str
    ascending: bool


DEFAULT_RANKING_RULES: tuple[RankingRule, ...] = (
    RankingRule(column="equity.total_return", ascending=False),
    RankingRule(column="equity.max_drawdown", ascending=True),
    RankingRule(column="trade.n_trades", ascending=False),
)


def _apply_overrides(config: BacktestRunConfig, overrides: Mapping[str, object]) -> None:
    """Aplica overrides sobre strategy_params, backtest_config o el propio run_config."""

    for key, value in overrides.items():
        if hasattr(config, key):
            setattr(config, key, value)
            continue

        if hasattr(config.backtest_config, key):
            setattr(config.backtest_config, key, value)
            continue

        params_obj = config.strategy_params
        if hasattr(params_obj, key):
            setattr(params_obj, key, value)
            continue

        raise AttributeError(f"El parámetro '{key}' no existe en la configuración")


def _strategy_params_dict(strategy_params: object) -> MutableMapping[str, object]:
    if is_dataclass(strategy_params):
        return asdict(strategy_params)
    return {
        k: getattr(strategy_params, k)
        for k in dir(strategy_params)
        if not k.startswith("_") and not callable(getattr(strategy_params, k))
    }


def build_parametrized_jobs(
    base_config: BacktestRunConfig,
    param_grid: Iterable[Mapping[str, object]],
    *,
    reports_dir: Path | None = None,
) -> list[BatchJob]:
    """
    Genera una lista de trabajos a partir de un config base y un grid de parámetros.

    Cada trabajo recibe un `reports_dir` único para evitar colisiones de ficheros.
    """

    jobs: list[BatchJob] = []
    target_reports_dir = reports_dir or (REPORTS_DIR / "batch_runs")
    for idx, overrides in enumerate(param_grid, start=1):
        run_cfg = copy.deepcopy(base_config)
        _apply_overrides(run_cfg, overrides)

        job_reports_dir = target_reports_dir / f"job_{idx:03d}"
        run_cfg.reports_dir = job_reports_dir

        job = BatchJob(
            job_id=f"job_{idx:03d}",
            run_config=run_cfg,
            param_overrides=dict(overrides),
        )
        jobs.append(job)
    return jobs


def _limit_memory(memory_limit_mb: int | None) -> None:
    if memory_limit_mb is None:
        return
    try:
        import resource

        soft = hard = int(memory_limit_mb * 1024 * 1024)
        resource.setrlimit(resource.RLIMIT_AS, (soft, hard))
        logger.info("Límite de memoria aplicado: %s MB", memory_limit_mb)
    except Exception as exc:  # pragma: no cover - dependiente de SO
        logger.warning("No se pudo aplicar límite de memoria: %s", exc)


def _run_job(job: BatchJob) -> BatchJobOutcome:
    artifacts = run_single_backtest(job.run_config)
    return BatchJobOutcome(job=job, artifacts=artifacts)


def schedule_jobs(
    jobs: Sequence[BatchJob],
    *,
    max_workers: int = 1,
    memory_limit_mb: int | None = None,
) -> list[BatchJobOutcome]:
    """Ejecuta los trabajos con un pool multiproceso respetando límites de memoria."""

    if max_workers <= 1:
        _limit_memory(memory_limit_mb)
        return [_run_job(job) for job in jobs]

    with mp.Pool(
        processes=max_workers, initializer=_limit_memory, initargs=(memory_limit_mb,)
    ) as pool:
        results = pool.map(_run_job, jobs)
    return results


def _flatten_metrics(prefix: str, stats: Mapping[str, object]) -> dict[str, object]:
    return {f"{prefix}.{k}": v for k, v in stats.items()}


def build_results_dataframe(
    outcomes: Sequence[BatchJobOutcome],
    ranking_rules: Sequence[RankingRule] | None = None,
) -> pd.DataFrame:
    """Convierte los resultados en un DataFrame y calcula el ranking reproducible."""

    records: list[dict[str, object]] = []
    for outcome in outcomes:
        cfg = outcome.job.run_config
        record: dict[str, object] = {
            "job_id": outcome.job.job_id,
            "strategy_name": cfg.strategy_name,
            "symbol": cfg.symbol,
            "timeframe": cfg.timeframe,
            "reports_dir": str(cfg.reports_dir) if cfg.reports_dir else None,
        }

        record.update(
            {f"param.{k}": v for k, v in _strategy_params_dict(cfg.strategy_params).items()}
        )
        record.update(_flatten_metrics("equity", outcome.artifacts.equity_stats))
        record.update(_flatten_metrics("trade", outcome.artifacts.trade_stats))
        records.append(record)

    df = pd.DataFrame.from_records(records)
    ranking_rules = list(ranking_rules or DEFAULT_RANKING_RULES)

    sort_cols = [rule.column for rule in ranking_rules] + ["job_id"]
    ascending = [rule.ascending for rule in ranking_rules] + [True]
    df = df.sort_values(by=sort_cols, ascending=ascending, kind="mergesort")
    df.insert(0, "rank", range(1, len(df) + 1))
    return df


def export_ranked_results(
    outcomes: Sequence[BatchJobOutcome],
    *,
    ranking_rules: Sequence[RankingRule] | None = None,
    output_path: Path | None = None,
) -> Path:
    """Genera un CSV con parámetros, métricas y ranking reproducible."""

    df = build_results_dataframe(outcomes, ranking_rules=ranking_rules)
    path = output_path or (REPORTS_DIR / "batch_runs" / "batch_results.csv")
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    logger.info("Resultados agregados exportados a %s", path)
    return path
