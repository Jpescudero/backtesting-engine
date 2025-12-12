# Investigación: Intraday Mean Reversion

Resumen del conjunto de scripts usados para explorar estrategias intradía de reversión a la media. El código vive en GitHub en: https://github.com/Jpescudero/backtesting-engine/tree/main/research/intraday_mean_reversion

## Estructura principal

- `intraday_mean_reversion_research.py`: punto de entrada CLI. Carga parámetros, descarga/lee los datos intradía y ejecuta una evaluación simple o una búsqueda en rejilla (`--run-grid-search`). Genera métricas y gráficos en el directorio de salida.
- `intraday_mean_reversion_research_params.txt`: archivo de parámetros base (símbolo, ventanas, umbrales de z-score, horarios de sesión, costes, etc.). Puedes modificarlo o pasar otro vía `--params-file`.
- `optimizers/grid_search.py`: recorrido exhaustivo de combinaciones de parámetros. Recibe un `param_grid` y una función objetivo; devuelve un `DataFrame` con las métricas de cada combinación.
- `optimizers/ml_optimizer.py`: esqueleto para un optimizador asistido por ML (p. ej. RandomForest) que sugerirá nuevos parámetros a partir de resultados previos.
- `utils/data_loader.py`: resolución flexible de rutas y carga de datos OHLCV intradía, gestionando índices de tiempo duplicados y normalización de columnas.
- `utils/events.py`: detección de eventos de reversión usando z-scores de retornos con filtro de horario de sesión.
- `utils/labeling.py`: etiqueta eventos con resultados posteriores (retornos a horizonte fijo, stop, take profit) para análisis y métricas.
- `utils/metrics.py`: calcula KPIs principales (tasa de acierto, retorno medio, drawdown, etc.).
- `utils/plotting.py`: genera gráficos de distribución de retornos, relación z-score/éxito y mapas de calor de búsquedas de parámetros.
- `utils/config_loader.py`: lectura de archivos de parámetros tipo `KEY=VALUE`, convirtiendo tipos numéricos y listas cuando es posible.
- `utils/costs.py`: helpers para aplicar costes de transacción en las métricas/eventos.
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

## Notas

- El módulo `optimizers/ml_optimizer.py` está preparado para incorporar un modelo de ML, pero todavía no entrena ni sugiere parámetros.
- Ajusta los horarios de sesión y los niveles de z-score en el archivo de parámetros para adaptarlos al símbolo y liquidez de interés.
