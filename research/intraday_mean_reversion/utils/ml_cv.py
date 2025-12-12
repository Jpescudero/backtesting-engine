"""Temporal cross-validation helpers for meta-labeling models."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import timedelta
from typing import List, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class FoldSplit:
    """Train/test index split for temporal validation."""

    train_idx: pd.Index
    test_idx: pd.Index
    label: str


def _year_range(start_year: int, end_year: int) -> range:
    if end_year < start_year:
        raise ValueError("end_year must be greater than or equal to start_year")
    return range(start_year, end_year + 1)


def generate_walk_forward_splits(
    events: pd.DataFrame,
    train_start_year: int,
    train_end_year: int,
    test_start_year: int,
    test_end_year: int,
    fold_years: int,
    min_train_days: int = 1,
    embargo_days: int = 0,
) -> List[FoldSplit]:
    """Create rolling train/test splits by calendar years.

    A fold trains on events from ``train_start_year`` up to ``current_end`` and
    tests on the following ``fold_years``. The window advances by ``fold_years``
    until reaching ``test_end_year``.
    """

    if events.empty:
        logger.warning("No events available for CV; returning empty splits")
        return []

    folds: List[FoldSplit] = []

    for start in _year_range(test_start_year, test_end_year):
        test_years = list(_year_range(start, min(start + fold_years - 1, test_end_year)))
        train_years = list(_year_range(train_start_year, min(train_end_year, start - 1)))
        if not train_years:
            continue

        train_mask = events.index.year.isin(train_years)
        test_mask = events.index.year.isin(test_years)
        if not test_mask.any():
            continue

        train_dates = events.index.normalize()[train_mask].unique()
        if len(train_dates) < min_train_days:
            logger.info("Skipping fold %s-%s due to insufficient train days", train_years[0], train_years[-1])
            continue

        train_idx = events.index[train_mask]
        test_idx = events.index[test_mask]

        if embargo_days > 0:
            embargo = timedelta(days=embargo_days)
            max_train_date = test_idx.min() - embargo
            train_idx = train_idx[train_idx.normalize() <= max_train_date.normalize()]

        if train_idx.empty:
            logger.info("Skipping fold %s-%s after embargo (no training samples)", train_years[0], train_years[-1])
            continue

        folds.append(
            FoldSplit(
                train_idx=train_idx,
                test_idx=test_idx,
                label=f"train_{train_years[0]}_{train_years[-1]}__test_{test_years[0]}_{test_years[-1]}",
            )
        )

    logger.info("Generated %s walk-forward folds", len(folds))
    return folds


def time_series_train_test_split(
    X: pd.DataFrame, y: pd.Series, split: FoldSplit
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Slice feature matrix and labels using a ``FoldSplit`` definition."""

    return X.loc[split.train_idx], X.loc[split.test_idx], y.loc[split.train_idx], y.loc[split.test_idx]


def evaluate_classifier(
    clf, X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series
) -> dict[str, float]:
    """Fit a classifier and compute standard classification metrics."""

    clf.fit(X_train, y_train)
    proba_test = clf.predict_proba(X_test)[:, 1]
    proba_train = clf.predict_proba(X_train)[:, 1]

    try:
        from sklearn.metrics import (
            roc_auc_score,
            average_precision_score,
            brier_score_loss,
            log_loss,
        )
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("scikit-learn is required for evaluation") from exc

    metrics = {
        "auc_test": roc_auc_score(y_test, proba_test) if len(np.unique(y_test)) > 1 else np.nan,
        "auc_train": roc_auc_score(y_train, proba_train) if len(np.unique(y_train)) > 1 else np.nan,
        "ap_test": average_precision_score(y_test, proba_test),
        "ap_train": average_precision_score(y_train, proba_train),
        "brier_test": brier_score_loss(y_test, proba_test),
        "brier_train": brier_score_loss(y_train, proba_train),
        "logloss_test": log_loss(y_test, proba_test),
        "logloss_train": log_loss(y_train, proba_train),
    }
    return metrics
