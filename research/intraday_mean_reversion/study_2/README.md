# Study 2 — Intraday Mean Reversion Half-Life

## Objetivo
Este estudio cuantifica de forma empírica cuánto tarda el precio en revertir intradía bajo régimen de **mean reversion (MR)**, usando exactamente la misma lógica de detección y filtrado validada en el Study 1. El resultado proporciona tiempos de `holding` y *time stops* estadísticos no optimizables que se integrarán directamente en el research principal de `intraday_mean_reversion`.

## Relación con Study 1
- Se reutilizan las detecciones de eventos MR, los filtros de régimen (volatilidad, tendencia, shocks y horario) y la carga de datos del Study 1.
- La diferencia es que aquí **no se optimiza nada**: solo se mide el tiempo que tarda la desviación inicial en reducirse a la mitad.
- Los parámetros derivados (p. ej. `HOLD_TIME_BY_TIME_BUCKET`, `MAX_TIME_STOP_BY_TIME_BUCKET`) se fijarán como conocimiento estructural en el research principal.

## Definición de half-life
Para cada evento en `t0`:
- Desviación inicial: `D0 = price(t0) - reference_mean(t0)`.
- La media de referencia se congela en `t0` y se calcula como la media móvil simple de `close` de longitud `LOOKBACK_MINUTES`.
- Half-life: primer minuto `t > t0` tal que `|price(t) - reference_mean(t0)| <= HALF_LIFE_THRESHOLD * |D0|`.
- Si no ocurre en `MAX_LOOKAHEAD_MIN`: `half_life_min = NaN` y `reverted = False`.

## Segmentación horaria
Los eventos se asignan a franjas fijas definidas en `TIME_BUCKETS` (conocimiento estructural, no optimizable), por defecto:
```
OPEN:09:35-10:30
MID_AM:10:30-12:00
LUNCH:12:00-14:00
PM:14:00-15:45
```

## Parámetros de entrada
- Se leen desde `study_2_params.txt` en el mismo formato `key=value` que el Study 1.
- Incluyen parámetros base del research MR, filtros de régimen y los específicos del Study 2:
  - `HALF_LIFE_THRESHOLD`
  - `MAX_LOOKAHEAD_MIN`
  - `TIME_BUCKETS`
- Si falta algún parámetro crítico, el runner aborta con error explícito.

## Outputs
Todos los archivos se guardan en `research/intraday_mean_reversion/output/study_2/<run_id>/`:
- `event_half_life_log.csv`: detalle por evento (timestamp, bucket, side, desviación inicial, half-life y `reverted`).
- `half_life_by_time_bucket.csv`: estadísticos agregados (media, mediana, p75, p90 y `%_no_reversion`).
- `reversion_probability_by_horizon.csv`: probabilidad de revertir en 5/10/15/30 minutos por franja (define *time stops* estadísticos).
- `regime_context.json`: `% tiempo en régimen MR`, nº de eventos analizados y definiciones de half-life y franjas.
- `params.json`: volcado exacto de parámetros usados (reproducibilidad).

## Uso e interpretación
1. Ejecutar `runner.py` con un único `.txt` de parámetros (sin grids ni optimizaciones).
2. De `half_life_by_time_bucket.csv` se derivan los `holding time` óptimos por franja (p. ej. mediana o p75 según criterio de robustez).
3. De `reversion_probability_by_horizon.csv` se extraen *time stops* estadísticos (`MAX_TIME_STOP_BY_TIME_BUCKET`).
4. Estos valores se fijan como parámetros **no optimizables** en el research principal.

## Advertencias
- Solo se consideran eventos y régimen MR (idéntico a Study 1); días direccionales o alta volatilidad quedan excluidos por construcción.
- No se usan salidas reales de estrategia ni lógica de PnL: es un estudio descriptivo.
- Si no hay suficientes eventos en alguna franja, interprete los resultados con cautela y valide con más datos.
