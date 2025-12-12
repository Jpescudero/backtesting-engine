import logging
from pathlib import Path

import pandas as pd
import pytest

from research.intraday_mean_reversion.optimizers.grid_search import GridSearchOptimizer
from research.intraday_mean_reversion.utils.config_loader import load_params
from research.intraday_mean_reversion.utils.data_loader import load_intraday_data
from research.intraday_mean_reversion.utils.events import detect_mean_reversion_events
from research.intraday_mean_reversion.utils.labeling import label_events
from research.intraday_mean_reversion.utils.metrics import compute_metrics
from research.intraday_mean_reversion.utils.plotting import (
    plot_heatmap_param_space,
    plot_return_distribution,
    plot_zscore_vs_success,
)


def test_load_params_parses_types(tmp_path: Path) -> None:
    config_content = """
    SYMBOL=TEST
    START_YEAR=2020
    END_YEAR=2021
    DATA_PATH=data
    DATA_FILE_PATTERN={symbol}.csv
    LOOKBACK_MINUTES=5
    ZSCORE_ENTRY=1.0
    HOLD_TIME_BARS=2
    SESSION_START_TIME=09:00
    SESSION_END_TIME=17:30
    GRID_LOOKBACK_MINUTES=5,10
    GRID_ZSCORE_ENTRY=1.0,2.0
    GRID_HOLD_TIME_BARS=1,3
    """
    config_file = tmp_path / "params.txt"
    config_file.write_text(config_content)

    params = load_params(str(config_file))

    assert params["START_YEAR"] == 2020
    assert params["ZSCORE_ENTRY"] == 1.0
    assert params["GRID_LOOKBACK_MINUTES"] == [5, 10]
    assert params["SESSION_END_TIME"] == "17:30"


def test_load_params_strips_inline_comments(tmp_path: Path) -> None:
    config_content = """
    SYMBOL=TEST
    START_YEAR=2020
    END_YEAR=2021
    DATA_PATH=data    # ruta base relativa
    DATA_FILE_PATTERN={symbol}.csv  # patron
    LOOKBACK_MINUTES=5
    ZSCORE_ENTRY=1.0
    HOLD_TIME_BARS=2
    SESSION_START_TIME=09:00
    SESSION_END_TIME=17:30
    """
    config_file = tmp_path / "params_with_comments.txt"
    config_file.write_text(config_content)

    params = load_params(str(config_file))

    assert params["DATA_PATH"] == "data"
    assert params["DATA_FILE_PATTERN"] == "{symbol}.csv"


def test_load_intraday_data_filters_and_deduplicates(tmp_path: Path) -> None:
    data_path = tmp_path / "data"
    data_path.mkdir()
    file_path = data_path / "TEST.csv"
    timestamps = pd.date_range("2020-01-01 09:00", periods=3, freq="T")
    df = pd.DataFrame(
        {
            "timestamp": [timestamps[0], timestamps[1], timestamps[1]],
            "open": [1.0, 2.0, 2.0],
            "high": [1.0, 2.5, 2.2],
            "low": [1.0, 1.5, 1.8],
            "close": [1.0, 2.0, 2.1],
            "volume": [10, 20, 30],
        }
    )
    df.to_csv(file_path, index=False)

    params = {
        "DATA_PATH": str(data_path),
        "DATA_FILE_PATTERN": "{symbol}.csv",
        "START_YEAR": 2020,
        "END_YEAR": 2020,
    }
    loaded = load_intraday_data("TEST", 2020, 2020, params)

    assert loaded.index.is_monotonic_increasing
    assert len(loaded) == 2
    assert loaded.loc[timestamps[1], "volume"] == 50


def test_load_intraday_data_resolves_via_data_hubs(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    data_file = tmp_path / "mirror" / "TEST.csv"
    data_file.parent.mkdir(parents=True)
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2020-01-01 00:00", periods=2, freq="min"),
            "open": [1.0, 2.0],
            "high": [1.1, 2.1],
            "low": [0.9, 1.9],
            "close": [1.05, 1.95],
            "volume": [100, 200],
        }
    )
    df.to_csv(data_file, index=False)

    def _fake_resolve(path: Path) -> Path:
        assert path == Path("data") / "TEST.csv"
        return data_file

    monkeypatch.setattr(
        "research.intraday_mean_reversion.utils.data_loader.resolve_data_path",
        _fake_resolve,
    )

    params = {
        "DATA_PATH": "data",
        "DATA_FILE_PATTERN": "{symbol}.csv",
        "START_YEAR": 2020,
        "END_YEAR": 2020,
    }

    loaded = load_intraday_data("TEST", 2020, 2020, params)

    assert loaded.index[0].year == 2020
    assert loaded.iloc[0]["volume"] == 100


def test_load_intraday_data_remaps_absolute_path_to_mirror(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    primary_hub = tmp_path / "primary"
    mirror_hub = tmp_path / "mirror"
    base_path = primary_hub / "data"
    data_file = mirror_hub / "data" / "TEST.csv"
    data_file.parent.mkdir(parents=True)

    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2020-01-01 00:00", periods=1, freq="min"),
            "open": [1.0],
            "high": [1.1],
            "low": [0.9],
            "close": [1.05],
            "volume": [100],
        }
    )
    df.to_csv(data_file, index=False)

    monkeypatch.setattr(
        "research.intraday_mean_reversion.utils.data_loader.DATA_DIR",
        primary_hub,
    )
    monkeypatch.setattr(
        "research.intraday_mean_reversion.utils.data_loader.DATA_MIRRORS",
        [mirror_hub],
    )

    params = {
        "DATA_PATH": str(base_path),
        "DATA_FILE_PATTERN": "{symbol}.csv",
        "START_YEAR": 2020,
        "END_YEAR": 2020,
    }

    loaded = load_intraday_data("TEST", 2020, 2020, params)

    assert loaded.iloc[0]["close"] == 1.05


def test_load_intraday_data_remaps_case_insensitive(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    primary_hub = tmp_path / "primary"
    mirror_hub = tmp_path / "mirror"
    base_path = primary_hub.parent / "Primary" / "data"
    data_file = mirror_hub / "data" / "TEST.csv"
    data_file.parent.mkdir(parents=True)

    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2020-01-01 00:00", periods=1, freq="min"),
            "open": [1.0],
            "high": [1.1],
            "low": [0.9],
            "close": [1.05],
            "volume": [100],
        }
    )
    df.to_csv(data_file, index=False)

    monkeypatch.setattr(
        "research.intraday_mean_reversion.utils.data_loader.DATA_DIR",
        primary_hub,
    )
    monkeypatch.setattr(
        "research.intraday_mean_reversion.utils.data_loader.DATA_MIRRORS",
        [mirror_hub],
    )

    params = {
        "DATA_PATH": str(base_path),
        "DATA_FILE_PATTERN": "{symbol}.csv",
        "START_YEAR": 2020,
        "END_YEAR": 2020,
    }

    loaded = load_intraday_data("TEST", 2020, 2020, params)

    assert loaded.iloc[0]["high"] == 1.1


def test_load_intraday_data_searches_symbol_subdirectory(tmp_path: Path) -> None:
    base_path = tmp_path / "Market Data" / "parquet" / "ticks"
    symbol = "NDXm"
    data_dir = base_path / symbol
    data_dir.mkdir(parents=True)

    data_file = data_dir / f"{symbol}.csv"
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2020-01-01 00:00", periods=2, freq="min"),
            "open": [1.0, 2.0],
            "high": [1.1, 2.1],
            "low": [0.9, 1.9],
            "close": [1.05, 1.95],
            "volume": [100, 200],
        }
    )
    df.to_csv(data_file, index=False)

    params = {
        "DATA_PATH": str(base_path),
        "DATA_FILE_PATTERN": "{symbol}.csv",
        "START_YEAR": 2020,
        "END_YEAR": 2020,
    }

    loaded = load_intraday_data(symbol, 2020, 2020, params)

    assert loaded.iloc[1]["close"] == 1.95


def test_events_labeling_and_metrics(tmp_path: Path) -> None:
    timestamps = pd.date_range("2020-01-01 09:00", periods=5, freq="min")
    prices = [100.0, 100.0, 90.0, 95.0, 100.0]
    df = pd.DataFrame(
        {"close": prices, "open": prices, "high": prices, "low": prices, "volume": 100},
        index=timestamps,
    )
    params = {
        "LOOKBACK_MINUTES": 2,
        "ZSCORE_ENTRY": 1.0,
        "SESSION_START_TIME": "09:00",
        "SESSION_END_TIME": "10:00",
        "HOLD_TIME_BARS": 1,
        "SPREAD_POINTS": 0.0,
        "SLIPPAGE_POINTS": 0.0,
        "COMMISSION_PER_CONTRACT": 0.0,
        "CONTRACT_MULTIPLIER": 1.0,
    }

    events = detect_mean_reversion_events(df, params)
    assert len(events) >= 1
    labeled = label_events(df, events, params)
    metrics = compute_metrics(labeled)

    assert labeled["r_next"].iloc[0] > 0
    assert metrics["n_events"] == 1
    assert metrics["p_next_bar_pos"] == 1.0


def test_plotting_outputs(tmp_path: Path) -> None:
    data = pd.DataFrame(
        {
            "r_H_net": [0.1, -0.05, 0.02],
            "side": [1, -1, 1],
            "z_score": [1.0, -1.5, 0.5],
            "is_r_H_net_positive": [True, False, True],
        }
    )

    plot_return_distribution(data, tmp_path / "ret.png", by_side=True)
    plot_zscore_vs_success(data, tmp_path / "zscore.png")

    assert (tmp_path / "ret.png").exists()
    assert (tmp_path / "zscore.png").exists()


@pytest.mark.filterwarnings("ignore:.*")
def test_grid_search_runs(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO)

    def objective(params: dict[str, float]) -> dict[str, float]:
        return {"E_r_H_net": params["LOOKBACK_MINUTES"] + params["ZSCORE_ENTRY"]}

    grid = {"LOOKBACK_MINUTES": [1, 2], "ZSCORE_ENTRY": [0.5, 1.0]}
    optimizer = GridSearchOptimizer(grid, objective)
    results = optimizer.run()

    assert len(results) == 4
    assert {"LOOKBACK_MINUTES", "ZSCORE_ENTRY", "E_r_H_net"}.issubset(results.columns)

    plot_heatmap_param_space(results, tmp_path / "heatmap.png", value_col="E_r_H_net")
    assert (tmp_path / "heatmap.png").exists()
