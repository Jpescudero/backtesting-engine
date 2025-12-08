"""Dashboard web ligero para analizar trades con filtros interactivos."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


def _to_iso(value: Any) -> str:
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    return str(value)


def _prepare_records(trades: pd.DataFrame, volatility_col: str | None) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    qty_exists = "qty" in trades.columns

    for row in trades.to_dict("records"):
        pnl = float(row["pnl"])
        qty = float(row["qty"]) if qty_exists else 0.0
        record: dict[str, Any] = {
            "entry_time": _to_iso(row["entry_time"]),
            "exit_time": _to_iso(row["exit_time"]),
            "pnl": pnl,
            "qty": qty,
            "is_win": pnl >= 0,
            "direction": "long" if qty >= 0 else "short",
        }
        for optional_key in ("entry_price", "exit_price", "exit_reason", "holding_bars"):
            if optional_key in row and row[optional_key] is not None:
                record[optional_key] = row[optional_key]
        if volatility_col:
            record[volatility_col] = float(row[volatility_col])
        records.append(record)

    return records


def build_trades_dashboard(
    trades: pd.DataFrame, output_path: str | Path, *, volatility_col: str | None = None
) -> Path:
    """Genera un HTML interactivo para explorar trades.

    Parámetros
    ----------
    trades:
        DataFrame con, al menos, ``entry_time``, ``exit_time`` y ``pnl``. Las columnas
        adicionales como ``qty``, ``entry_price``, ``exit_price``, ``exit_reason`` y
        ``holding_bars`` se usarán en los tooltips si están presentes. Si se proporciona
        ``volatility_col``, la columna debe existir y contener valores numéricos.
    output_path:
        Ruta donde se guardará el HTML generado.
    volatility_col:
        Nombre de la columna de volatilidad (o cualquier métrica continua) para aplicar
        un filtro por rango en la interfaz.
    """

    required_cols = {"entry_time", "exit_time", "pnl"}
    missing = required_cols - set(trades.columns)
    if missing:
        raise ValueError(f"Faltan columnas obligatorias en trades: {sorted(missing)}")
    if trades.empty:
        raise ValueError("No hay trades para mostrar en el dashboard")

    trades = trades.copy()
    trades["entry_time"] = pd.to_datetime(trades["entry_time"])
    trades["exit_time"] = pd.to_datetime(trades["exit_time"])

    if volatility_col:
        if volatility_col not in trades.columns:
            raise ValueError(
                f"La columna de volatilidad '{volatility_col}' no está en el DataFrame"
            )
        trades[volatility_col] = pd.to_numeric(trades[volatility_col])

    trade_records = _prepare_records(trades, volatility_col)

    exit_reasons = sorted(
        {rec.get("exit_reason", "") for rec in trade_records if rec.get("exit_reason")}
    )
    volatility_min = float(trades[volatility_col].min()) if volatility_col else None
    volatility_max = float(trades[volatility_col].max()) if volatility_col else None

    exit_options_html = "".join(
        f"<option value='{reason}'>{reason}</option>" for reason in exit_reasons
    )
    vol_min_input = (
        f"<div><label for='vol-min'>{volatility_col} (mín)</label>"
        f"<input type='number' step='any' id='vol-min' placeholder='{volatility_min:.3f}'></div>"
        if volatility_col
        else ""
    )
    vol_max_input = (
        f"<div><label for='vol-max'>{volatility_col} (máx)</label>"
        f"<input type='number' step='any' id='vol-max' placeholder='{volatility_max:.3f}'></div>"
        if volatility_col
        else ""
    )

    dashboard_template = """
<!DOCTYPE html>
<html lang="es">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Trades dashboard</title>
  <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
  <style>
    body {{
      font-family: 'Inter', system-ui, -apple-system, sans-serif;
      margin: 0;
      background: #f6f8fb;
    }}
    header {{
      background: #0f172a;
      color: white;
      padding: 20px 28px;
      box-shadow: 0 1px 8px rgba(0,0,0,0.15);
    }}
    h1 {{ margin: 0 0 4px 0; font-size: 24px; }}
    .container {{ padding: 24px 28px 32px; max-width: 1400px; margin: auto; }}
    .panel {{
      background: white;
      border-radius: 12px;
      padding: 16px 18px;
      box-shadow: 0 8px 20px rgba(15,23,42,0.08);
      margin-bottom: 18px;
    }}
    .filters {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 12px;
      align-items: end;
    }}
    label {{
      font-size: 13px;
      color: #334155;
      font-weight: 600;
      display: block;
      margin-bottom: 4px;
    }}
    input, select {{
      width: 100%;
      padding: 8px 10px;
      border: 1px solid #e2e8f0;
      border-radius: 8px;
      font-size: 14px;
    }}
      .checkbox-group {{ display: flex; gap: 12px; align-items: center; }}
      .kpi-grid {{
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
        gap: 12px;
      }}
    .kpi {{
      background: linear-gradient(135deg, #0ea5e9, #2563eb);
      color: white;
      padding: 14px;
      border-radius: 12px;
      box-shadow: inset 0 1px 0 rgba(255,255,255,0.25);
    }}
    .kpi small {{ display: block; opacity: 0.85; }}
    .kpi strong {{ font-size: 22px; }}
      .chart {{ width: 100%; height: 420px; }}
      .table {{ width: 100%; border-collapse: collapse; margin-top: 8px; font-size: 13px; }}
      .table th, .table td {{
        border: 1px solid #e2e8f0;
        padding: 6px 8px;
        text-align: left;
      }}
    .table th {{ background: #f8fafc; color: #475569; }}
  </style>
</head>
<body>
  <header>
    <h1>Dashboard de trades</h1>
    <div>Filtra, explora y entiende tus operaciones rápidamente.</div>
  </header>
  <div class="container">
    <div class="panel">
      <div class="filters">
        <div class="checkbox-group">
          <label><input type="checkbox" id="winners" checked> Ganadoras</label>
          <label><input type="checkbox" id="losers" checked> Perdedoras</label>
        </div>
        <div>
          <label for="from">Desde (salida)</label>
          <input type="date" id="from">
        </div>
        <div>
          <label for="to">Hasta (salida)</label>
          <input type="date" id="to">
        </div>
        <div>
          <label for="reason">Motivo de salida</label>
          <select id="reason">
            <option value="">Todos</option>
            __EXIT_OPTIONS__
          </select>
        </div>
        __VOL_MIN__
        __VOL_MAX__
      </div>
    </div>

    <div class="panel kpi-grid" id="kpis"></div>

    <div class="panel">
      <div id="equity" class="chart"></div>
    </div>

    <div class="panel">
      <div id="pnl-scatter" class="chart"></div>
      <table class="table" id="table"></table>
    </div>
  </div>

  <script>
    const trades = __TRADES__;
    const volatilityField = __VOL_FIELD__;

    function parseDate(value) {{
      return value ? new Date(value) : null;
    }}

    function filterTrades() {{
      const winners = document.getElementById('winners').checked;
      const losers = document.getElementById('losers').checked;
      const from = parseDate(document.getElementById('from').value);
      const to = parseDate(document.getElementById('to').value);
      const reason = document.getElementById('reason').value;
      const volMinEl = document.getElementById('vol-min');
      const volMaxEl = document.getElementById('vol-max');
      const volMin = volMinEl ? Number(volMinEl.value) : null;
      const volMax = volMaxEl ? Number(volMaxEl.value) : null;

      return trades.filter(trade => {{
        if (trade.is_win && !winners) return false;
        if (!trade.is_win && !losers) return false;
        const exitDate = new Date(trade.exit_time);
        if (from && exitDate < from) return false;
        if (to) {{
          const end = new Date(to);
          end.setHours(23,59,59,999);
          if (exitDate > end) return false;
        }}
        if (reason && trade.exit_reason !== reason) return false;
        if (volatilityField) {{
          const vol = trade[volatilityField];
          if (volMinEl && volMinEl.value && vol < volMin) return false;
          if (volMaxEl && volMaxEl.value && vol > volMax) return false;
        }}
        return true;
      }});
    }

    function renderKPIs(filtered) {{
      const total = filtered.length;
      const wins = filtered.filter(t => t.is_win).length;
      const pnl = filtered.reduce((acc, t) => acc + t.pnl, 0);
      const avgHold = filtered.reduce((acc, t) => acc + (t.holding_bars ?? 0), 0) / (total || 1);
      const winRate = total ? (wins / total) * 100 : 0;

        document.getElementById('kpis').innerHTML = `
          <div class="kpi"><small>Total trades</small><strong>${{total}}</strong></div>
          <div class="kpi"><small>Win rate</small><strong>${{winRate.toFixed(1)}}%</strong></div>
          <div class="kpi"><small>PnL total</small><strong>${{pnl.toFixed(2)}}</strong></div>
          <div class="kpi">
            <small>Holding medio (barras)</small>
            <strong>${{avgHold.toFixed(1)}}</strong>
          </div>
        `;
      }

    function renderCharts(filtered) {{
      const sorted = [...filtered].sort((a, b) => new Date(a.exit_time) - new Date(b.exit_time));
      let acc = 0;
      const equityY = sorted.map(t => acc += t.pnl);

      Plotly.react('equity', [{{
        x: sorted.map(t => t.exit_time),
        y: equityY,
        mode: 'lines+markers',
        marker: {{ color: '#2563eb' }},
        line: {{ width: 2 }},
        name: 'Equity'
      }}], {{
        title: 'Evolución del PnL',
        template: 'plotly_white',
        margin: {{ t: 32, l: 48, r: 24, b: 48 }},
        yaxis: {{ zeroline: true, zerolinecolor: '#94a3b8' }}
      }});

      Plotly.react('pnl-scatter', [{{
        x: sorted.map(t => t.exit_time),
        y: sorted.map(t => t.pnl),
        mode: 'markers',
        marker: {{
          color: sorted.map(t => t.is_win ? '#10b981' : '#ef4444'),
          size: sorted.map(t => Math.max(6, Math.min(18, Math.abs(t.pnl) / 50))),
          opacity: 0.85,
        }},
        text: sorted.map(t => `PnL: ${{t.pnl.toFixed(2)}} | ${{t.direction}}`),
        hovertemplate: '<b>%{text}</b><br>Salida: %{x}<extra></extra>',
        name: 'Trades'
      }}], {{
        title: 'PnL por trade',
        template: 'plotly_white',
        margin: {{ t: 32, l: 48, r: 24, b: 48 }},
        yaxis: {{ zeroline: true, zerolinecolor: '#94a3b8' }}
      }});
    }

    function renderTable(filtered) {{
      const head = `
        <tr>
          <th>Salida</th>
          <th>PnL</th>
          <th>Sentido</th>
          <th>Qty</th>
          <th>Motivo</th>
        </tr>
      `;
      const rows = filtered.slice(-25).reverse().map(trade => `
        <tr>
          <td>${{trade.exit_time}}</td>
          <td style="color:${{trade.is_win ? '#10b981' : '#ef4444'}}">${{trade.pnl.toFixed(2)}}</td>
          <td>${{trade.direction}}</td>
          <td>${{trade.qty.toFixed(2)}}</td>
          <td>${{trade.exit_reason || ''}}</td>
        </tr>
      `).join('');
      document.getElementById('table').innerHTML = head + rows;
    }

    function refresh() {{
      const filtered = filterTrades();
      renderKPIs(filtered);
      renderCharts(filtered);
      renderTable(filtered);
    }

    document.querySelectorAll('input, select').forEach(el => el.addEventListener('input', refresh));
    refresh();
  </script>
</body>
</html>
"""

    dashboard_html = (
        dashboard_template.replace("__EXIT_OPTIONS__", exit_options_html)
        .replace("__VOL_MIN__", vol_min_input)
        .replace("__VOL_MAX__", vol_max_input)
        .replace("__TRADES__", json.dumps(trade_records))
        .replace("__VOL_FIELD__", json.dumps(volatility_col))
    )
    dashboard_html = dashboard_html.replace("{{", "{").replace("}}", "}")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(dashboard_html, encoding="utf-8")
    return output_path

