import pandas as pd

from src.visualization.trades_dashboard import build_trades_dashboard


def test_build_trades_dashboard(tmp_path) -> None:
    trades = pd.DataFrame(
        {
            "entry_time": pd.date_range("2024-01-01", periods=3, freq="D"),
            "exit_time": pd.date_range("2024-01-02", periods=3, freq="D"),
            "pnl": [10.0, -5.5, 20.0],
            "qty": [1, -2, 1.5],
            "exit_reason": ["tp", "sl", "manual"],
            "holding_bars": [3, 2, 5],
            "volatility": [0.8, 1.2, 0.5],
        }
    )
    output = build_trades_dashboard(
        trades,
        tmp_path / "dashboard.html",
        volatility_col="volatility",
    )

    html = output.read_text(encoding="utf-8")
    assert output.exists()
    assert "Dashboard de trades" in html
    assert "volatility" in html
    assert "Ganadoras" in html
