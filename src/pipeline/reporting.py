from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt

from src.analytics.backtest_output import save_backtest_summary_to_excel
from src.analytics.metrics import equity_curve_metrics, trades_metrics
from src.analytics.plots import plot_equity_curve, plot_trades_per_month
from src.analytics.reporting import equity_to_series, trades_to_dataframe
from src.analytics.trade_plots import plot_best_and_worst_trades


@dataclass
class BacktestReports:
    equity_stats: Dict
    trade_stats: Dict
    equity_path: Path | None = None
    best_trade_plot_path: Path | None = None
    worst_trade_plot_path: Path | None = None
    excel_path: Path | None = None
    json_path: Path | None = None


def compute_analytics(result, data, equity_series=None, trades_df=None):
    equity_series = equity_series if equity_series is not None else equity_to_series(result, data)
    trades_df = trades_df if trades_df is not None else trades_to_dataframe(result, data)
    equity_stats = equity_curve_metrics(equity_series)
    trade_stats = trades_metrics(trades_df)
    return equity_series, trades_df, equity_stats, trade_stats


def generate_report_files(reports_dir: Path, symbol: str, strategy_name: str, equity_series, trades_df,
                          equity_stats: Dict, trade_stats: Dict, meta: Dict) -> Tuple[Path, Path]:
    excel_path, json_path = save_backtest_summary_to_excel(
        base_dir=reports_dir,
        filename=f"backtest_{symbol}_{strategy_name}.xlsx",
        symbol=symbol,
        strategy_name=strategy_name,
        equity_series=equity_series,
        trades_df=trades_df,
        equity_stats=equity_stats,
        trade_stats=trade_stats,
        meta=meta,
    )
    return excel_path, json_path


def generate_main_plots(result, data, reports_dir: Path | None, show: bool) -> Path | None:
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=False)
    plot_equity_curve(result, data, ax=ax1)
    plot_trades_per_month(result, data, ax=ax2)
    plt.tight_layout()

    save_path = None
    if reports_dir is not None:
        reports_dir.mkdir(parents=True, exist_ok=True)
        save_path = reports_dir / "equity_trades.png"
        fig.savefig(save_path, dpi=150)

    if show:
        plt.show()
    else:
        plt.close(fig)
    return save_path


def generate_trade_plots(trades_df, data, reports_dir: Path | None, show: bool) -> Tuple[Path | None, Path | None]:
    best_path = (reports_dir / "best_trades.png") if reports_dir else None
    worst_path = (reports_dir / "worst_trades.png") if reports_dir else None

    best_fig, worst_fig = plot_best_and_worst_trades(
        trades_df=trades_df,
        data=data,
        n_best=6,
        pnl_col="pnl",
        entry_col="entry_idx",
        exit_col="exit_idx",
        direction_col="direction",
        window=30,
        figsize=(14, 10),
        save_best_path=best_path,
        save_worst_path=worst_path,
    )

    if show:
        plt.show()
    else:
        plt.close(best_fig)
        plt.close(worst_fig)
    return best_path, worst_path
