# Microstructure opening sweep research

Este módulo contiene tres versiones del estudio de "opening sweeps" y un
optimizador común para calibrar sus parámetros. Ninguno de estos scripts es un
motor de backtesting; su único objetivo es generar estadísticas y parámetros que
luego se implementarán en estrategias del motor principal.

## Versiones del estudio

- **V1 (`study_opening_sweeps_v1.py`)**: versión base con filtros mínimos de
  wick, ATR y volumen. Calcula retornos forward fijos.
- **V2 (`study_opening_sweeps_v2.py`)**: añade filtros microestructurales y un
  horizonte forward dinámico dependiente del ATR.
- **V3 (`study_opening_sweeps_v3.py`)**: incorpora buffers dinámicos de SL/TP,
  simulación de trades y puntuación de calidad del sweep.

Cada script escribe sus salidas en `reports/research/microstructure/reports/<vx>/`
con un subdirectorio fechado (`YYYYMMDD_HHMMSS`). Los reportes incluyen:

- `trades.csv`: cada trade individual con su retorno (y `r_multiple` en V3).
- `summary.csv`: recuento, media, desviación estándar y winrate.
- En V3 se añade `equity_curve.png` con la curva de equity simulada.

## Optimizador

`optimizer_strategies.py` expone una CLI única:

```bash
python research/microstructure/optimizer_strategies.py --strategy v3 --n-jobs 4
```

Características principales:

- Registro de estrategias mediante adaptadores (`v1`, `v2`, `v3`).
- Grid de parámetros extensible y tipado, pensado para añadir nuevos estudios.
- Multiproceso con pre-carga única de datos para minimizar el overhead.
- Reportes completos en `reports/research/microstructure/reports/optimizer/` con:
  - `optimizer_results.csv`: todas las combinaciones evaluadas.
  - `optimizer_top.csv`: top-20 por *Sharpe*.
  - `summary.json`: métrica agregada, mejor set de parámetros y tiempos.
  - Mapas de calor específicos por estrategia (p.ej. buffers SL/TP en V3).

Utiliza estos reportes para seleccionar los parámetros que luego se integrarán en
el motor de backtesting.
