# Guía rápida de links en GitHub

Esta guía sirve para compartir con ChatGPT (u otros colaboradores) los enlaces clave del repositorio una vez alojado en GitHub. Sustituye `<OWNER>/<REPO>` por la organización y nombre reales.

## Enlaces esenciales
- **Repositorio:** `https://github.com/<OWNER>/<REPO>`
- **README (visión general y estructura):** `https://github.com/<OWNER>/<REPO>/blob/main/README.md`
- **Entrada principal (CLI):** `https://github.com/<OWNER>/<REPO>/blob/main/main.py`
- **Configuración de rutas:** `https://github.com/<OWNER>/<REPO>/blob/main/src/config/paths.py`
- **Pipeline completo:** `https://github.com/<OWNER>/<REPO>/blob/main/src/pipeline/backtest_runner.py`
- **Pipeline de datos (ticks → CSV → NPZ):** `https://github.com/<OWNER>/<REPO>/blob/main/src/pipeline/data_pipeline.py`
- **Motor de backtesting (Numba):** `https://github.com/<OWNER>/<REPO>/blob/main/src/engine/core.py`
- **Métricas y reporting:** `https://github.com/<OWNER>/<REPO>/blob/main/src/analytics/metrics.py` y `https://github.com/<OWNER>/<REPO>/blob/main/src/analytics/reporting.py`
- **Estrategias disponibles:**
  - Barrida de apertura: `https://github.com/<OWNER>/<REPO>/blob/main/src/strategies/barrida_apertura.py`
  - Microstructure Reversal: `https://github.com/<OWNER>/<REPO>/blob/main/src/strategies/microstructure_reversal.py`
  - Microstructure Sweep: `https://github.com/<OWNER>/<REPO>/blob/main/src/strategies/microstructure_sweep.py`
- **Utilidades comunes:** `https://github.com/<OWNER>/<REPO>/blob/main/src/utils`
- **Ejemplo de configuración:** `https://github.com/<OWNER>/<REPO>/blob/main/run_settings.example.txt`
- **Pruebas automatizadas:** `https://github.com/<OWNER>/<REPO>/tree/main/tests`

## Enlaces útiles para documentación y apoyo
- **Estructura comentada del proyecto:** `https://github.com/<OWNER>/<REPO>/blob/main/backtesting_project_structure.txt`
- **Notebooks de experimentación:** `https://github.com/<OWNER>/<REPO>/tree/main/notebooks`
- **Informes generados (outputs):** `https://github.com/<OWNER>/<REPO>/tree/main/reports`

## Consejos de uso
- Incluye estos enlaces en las instrucciones de ChatGPT para que el modelo tenga contexto rápido sobre dónde están los componentes clave.
- Si cambias la rama principal (por ejemplo, `main` → `master`), ajusta `blob/main` o `tree/main` en los enlaces.
- Añade enlaces adicionales cuando crees nuevos módulos o dashboards.
