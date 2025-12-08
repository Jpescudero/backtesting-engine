"""Exportadores ligeros para colecciones de métricas."""

from __future__ import annotations

import base64
from io import BytesIO
from pathlib import Path
from typing import Iterable, Mapping

import pandas as pd

from src.analytics.collectors import MetricCollector


def _ensure_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def export_collector_to_csv(collector: MetricCollector, path: str | Path) -> Path:
    """Guarda el contenido del collector a CSV."""

    output_path = Path(path)
    _ensure_dir(output_path)
    collector.to_frame().to_csv(output_path, index=False)
    return output_path


def export_collector_to_parquet(collector: MetricCollector, path: str | Path) -> Path:
    """Guarda el contenido del collector a Parquet."""

    output_path = Path(path)
    _ensure_dir(output_path)
    collector.to_frame().to_parquet(output_path, index=False)
    return output_path


def export_collectors(
    collectors: Iterable[MetricCollector], output_dir: str | Path, fmt: str = "csv"
) -> Mapping[str, Path]:
    """Exporta múltiples collectors a una carpeta en CSV o Parquet."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results: dict[str, Path] = {}
    for collector in collectors:
        suffix = ".parquet" if fmt.lower() == "parquet" else ".csv"
        export_fn = export_collector_to_parquet if suffix == ".parquet" else export_collector_to_csv
        results[collector.name] = export_fn(collector, output_dir / f"{collector.name}{suffix}")
    return results


def _plot_with_plotly(df: pd.DataFrame, y: str, title: str) -> str | None:
    try:
        import plotly.graph_objects as go
    except Exception:  # pragma: no cover - dependencia opcional
        return None

    fig = go.Figure(go.Scatter(x=df["timestamp"], y=df[y], mode="lines", name=y))
    fig.update_layout(title=title, template="plotly_white")
    return fig.to_html(include_plotlyjs="cdn", full_html=False)


def _plot_with_matplotlib(df: pd.DataFrame, y: str, title: str) -> str:
    import matplotlib.pyplot as plt  # import local para no añadir dependencia global

    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(df["timestamp"], df[y], label=y)
    ax.set_title(title)
    ax.grid(True)
    ax.legend()

    buffer = BytesIO()
    fig.tight_layout()
    fig.savefig(buffer, format="png")
    plt.close(fig)
    buffer.seek(0)
    encoded = base64.b64encode(buffer.read()).decode("ascii")
    return f'<img src="data:image/png;base64,{encoded}" alt="{title}">'


def build_light_html_report(
    collectors: Mapping[str, MetricCollector], output_path: str | Path
) -> Path:
    """Genera un HTML liviano con gráficos de equity y drawdown."""

    output_path = Path(output_path)
    _ensure_dir(output_path)

    pnl = collectors.get("pnl")
    drawdown = collectors.get("drawdown")
    trades = collectors.get("trades")

    sections: list[str] = ["<h1>Reporte de backtest</h1>"]

    if pnl is not None:
        pnl_df = pnl.to_frame()
        html_plot = _plot_with_plotly(pnl_df, "equity", "Equity curve")
        if html_plot is None:
            html_plot = _plot_with_matplotlib(pnl_df, "equity", "Equity curve")
        sections.append("<h2>Equity</h2>")
        sections.append(html_plot)
        sections.append(pnl_df.head(5).to_html(index=False))

    if drawdown is not None:
        dd_df = drawdown.to_frame()
        html_plot = _plot_with_plotly(dd_df, "drawdown", "Drawdown")
        if html_plot is None:
            html_plot = _plot_with_matplotlib(dd_df, "drawdown", "Drawdown")
        sections.append("<h2>Drawdown</h2>")
        sections.append(html_plot)
        sections.append(dd_df.head(5).to_html(index=False))

    if trades is not None:
        trades_df = trades.to_frame()
        sections.append("<h2>Trades</h2>")
        sections.append(trades_df.head(10).to_html(index=False))

    html = "\n".join(sections)
    output_path.write_text(html, encoding="utf-8")
    return output_path
