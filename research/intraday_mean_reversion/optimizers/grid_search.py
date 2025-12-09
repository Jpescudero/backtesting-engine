"""Generic grid search optimizer for research workflows."""

from __future__ import annotations

from itertools import product
from typing import Any, Callable

import pandas as pd


class GridSearchOptimizer:
    """Brute-force parameter grid search runner.

    Parameters
    ----------
    param_grid : dict[str, list[Any]]
        Dictionary where keys are parameter names and values are lists of possible values.
    objective_fn : Callable[[dict[str, Any]], dict[str, Any]]
        Function that executes the pipeline for a single parameter combination and returns metrics.
    """

    def __init__(self, param_grid: dict[str, list[Any]], objective_fn: Callable[[dict[str, Any]], dict[str, Any]]):
        self.param_grid = param_grid
        self.objective_fn = objective_fn

    def _iter_params(self):
        keys = list(self.param_grid.keys())
        for values in product(*self.param_grid.values()):
            yield dict(zip(keys, values))

    def run(self) -> pd.DataFrame:
        """Run the grid search and collect results into a DataFrame."""

        results = []
        for params in self._iter_params():
            metrics = self.objective_fn(params)
            results.append({**params, **metrics})
        return pd.DataFrame(results)
