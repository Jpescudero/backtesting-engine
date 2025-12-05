📘 Backtesting Engine — Python & Numba (Darwinex-Oriented)

Motor de backtesting optimizado en Python + Numba, diseñado para trabajar con datos BID/ASK tick de Darwinex, generar barras intradía y ejecutar estrategias de forma eficiente.
El objetivo final es desarrollar estrategias aptas para convertirse en DARWIN (Darwinex).

🚀 Características principales
✔️ Pipeline completo de datos

Logs raw Darwinex (BID/ASK)

→ Limpieza y fusión en Parquet de ticks

→ Generación de barras de 1 minuto

→ Conversión a arrays NPZ optimizados para Numba

→ Ejecución del motor de backtesting

Todo el pipeline se ejecuta automáticamente desde main.py.

✔️ Motor de backtesting (Numba)

Incluye:

Estrategia de ejemplo basada en momentum

Stop Loss (%)

Take Profit (%)

Límite de duración de un trade

Registro detallado de todos los trades

Curva de equity completa

📊 Métricas tipo Darwinex

Generadas desde src/analytics/metrics.py:

Retorno total

Retorno anualizado

Max Drawdown

Return/Drawdown Ratio

Sharpe Ratio

Sortino Ratio

Volatilidad anual

VaR mensual 95% (histórico)

Y métricas de trades:

Número total de trades

Winrate

Payoff ratio

Expectancy por trade

Duración media

Distribución de salidas (SL, TP, TimeStop, Señal)

📈 Visualizaciones

Incluidas en src/analytics/plots.py:

Curva de equity

Número de trades por mes


▶️ Cómo ejecutar
1. Instalar dependencias
pip install numpy pandas matplotlib numba pyarrow

2. Colocar los datos raw de Darwinex en:
data/raw/darwinex/<SYMBOL>/BID/
data/raw/darwinex/<SYMBOL>/ASK/


Si tus logs están mezclados:

python -m src.data.organize_darwinex_files

3. Ejecutar el backtest
python main.py


El pipeline:

Genera Parquet si no existe

Genera barras 1m CSV

Convierte a NPZ

Ejecuta el motor

Imprime métricas

Muestra gráficos

🔧 Configuración del backtest

Editar dentro de run_single_backtest en main.py:

config = BacktestConfig(
    initial_cash=100000,
    commission_per_trade=1.0,
    trade_size=1.0,
    slippage=0.0,
    sl_pct=0.01,
    tp_pct=0.02,
    max_bars_in_trade=60,
    entry_threshold=0.001,
)

🔮 Próximos pasos del proyecto

Añadir estrategia de barrida personalizada

Simulación de spread dinámico y slippage realista

Backtesting multi-símbolo

Grid search para optimización de parámetros

Validación walk-forward

Compatibilidad con despliegue tipo DARWIN

👤 Autor

Jpescudero
Backtesting y Trading Algorítmico en Python.

🤝 Licencia

Proyecto privado y personal.
No apto para redistribución sin permiso.
