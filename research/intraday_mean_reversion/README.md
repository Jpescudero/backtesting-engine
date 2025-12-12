# Investigación: Intraday Mean Reversion

Resumen del conjunto de scripts usados para explorar estrategias intradía de reversión a la media. El código vive en GitHub en: https://github.com/Jpescudero/backtesting-engine/tree/main/research/intraday_mean_reversion

## Estructura principal

- `intraday_mean_reversion_research.py`: punto de entrada CLI. Carga parámetros, descarga/lee los datos intradía y ejecuta una evaluación simple o una búsqueda en rejilla (`--run-grid-search`). Genera métricas y gráficos en el directorio de salida. Incluye cálculo automático de umbrales recomendados a partir de estadísticas por bin de z-score y marcas en los gráficos de P(éxito) y esperanza.
- `intraday_mean_reversion_research_params.txt`: archivo de parámetros base (símbolo, ventanas, umbrales de z-score, horarios de sesión, costes, etc.). Puedes modificarlo o pasar otro vía `--params-file`.
- `optimizers/grid_search.py`: recorrido exhaustivo de combinaciones de parámetros. Recibe un `param_grid` y una función objetivo; devuelve un `DataFrame` con las métricas de cada combinación.
- `optimizers/ml_optimizer.py`: esqueleto para un optimizador asistido por ML (p. ej. RandomForest) que sugerirá nuevos parámetros a partir de resultados previos.
- `utils/data_loader.py`: resolución flexible de rutas y carga de datos OHLCV intradía, gestionando índices de tiempo duplicados y normalización de columnas.
- `utils/events.py`: detección de eventos de reversión usando z-scores de retornos con filtro de horario de sesión y soporte para modos asimétricos (`MODE=both|fade_up_only|fade_down_only`) con filtros duros `Z_MIN_SHORT` / `Z_MIN_LONG`.
- `utils/labeling.py`: etiqueta eventos con resultados posteriores (retornos a horizonte fijo, stop, take profit) para análisis y métricas.
- `utils/metrics.py`: calcula KPIs principales (tasa de acierto, retorno medio, drawdown, etc.) y genera estadísticas por bin de z-score para alimentar la recomendación de umbrales.
- `utils/thresholding.py`: aplica restricciones estadísticas (mínimo de eventos, CI inferior, esperanza neta, límite de cola) para sugerir umbrales de z-score diferenciados por lado y guarda `recommended_thresholds.csv`.
- `utils/plotting.py`: genera gráficos de distribución de retornos, relación z-score/éxito y mapas de calor de búsquedas de parámetros.
- `utils/config_loader.py`: lectura de archivos de parámetros tipo `KEY=VALUE`, convirtiendo tipos numéricos y listas cuando es posible.
- `utils/costs.py`: helpers para aplicar costes de transacción en las métricas/eventos.
- `optimizers/ml_meta_labeling.py`: pipeline de meta-labeling que construye features sin leakage, entrena un clasificador (Logistic Regression o RandomForest), valida con walk-forward y guarda predicciones y comparativas de uplift.
- `utils/ml_features.py`: generación de la matriz de características alineada con el timestamp del evento (tiempo del día, tendencia, momentum reciente, distancia al máximo intradía, etc.).
- `utils/ml_cv.py`: utilidades de validación temporal (walk-forward con embargo opcional) y métricas de clasificación.
- `utils/ml_reporting.py`: resumen baseline vs filtrado por ML (P&L diario, Sharpe, histograma de probabilidades, curvas ROC y gráficos de uplift).
- `__init__.py`: marca el paquete de investigación y expone utilidades compartidas.

## Uso básico

1. Asegúrate de tener los datos intradía accesibles según las rutas configuradas en `intraday_mean_reversion_research_params.txt`.
2. Ejecuta una evaluación simple:
   ```bash
   python -m research.intraday_mean_reversion.intraday_mean_reversion_research --params-file intraday_mean_reversion_research_params.txt
   ```
3. Ejecuta una búsqueda en rejilla sobre el espacio definido en el archivo de parámetros:
   ```bash
   python -m research.intraday_mean_reversion.intraday_mean_reversion_research --run-grid-search
   ```
4. Revisa los artefactos (CSV y gráficos) en el directorio `output/` o en el que especifiques con `--output-dir`.
5. Para activar el meta-labeling como filtro de eventos, añade `RUN_ML=true` en el archivo de parámetros o pasa el flag `--run-ml` (o `--ml-only`). Los artefactos de ML se guardan en `output/ml/`.

## Modos asimétricos y umbrales recomendados

- Define el modo en `intraday_mean_reversion_research_params.txt` con `MODE=both|fade_up_only|fade_down_only`.
- Activa filtros duros con `Z_MIN_SHORT` (fade de subidas extremas) y `Z_MIN_LONG` (fade de caídas extremas). Estos filtros se aplican tras la detección para que las métricas y el PnL reflejen únicamente el lado elegido.
- El script calcula `recommended_thresholds.csv` a partir de las estadísticas por bin (`zscore_bins.csv`) usando restricciones configurables (`MIN_EVENTS_PER_BIN`, `MIN_CI_LOW`, `MIN_EXPECTANCY_NET`, `MAX_TAIL_LOSS`).
- En `zscore_vs_success.png` y `zscore_expected_return.png` se dibuja la línea vertical del umbral recomendado cuando procede.

## Notas

- El módulo `optimizers/ml_optimizer.py` está preparado para incorporar un modelo de ML, pero todavía no entrena ni sugiere parámetros.
- Ajusta los horarios de sesión y los niveles de z-score en el archivo de parámetros para adaptarlos al símbolo y liquidez de interés.
