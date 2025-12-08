# ¿Qué se configuró recientemente?

Esta es una explicación en español, en lenguaje sencillo, de los cambios de
mantenimiento que se hicieron en el proyecto:

1. **Formatos y estilo automáticos**: se añadieron Black, Ruff e isort para que
   el código quede ordenado y limpio de forma automática. Así no dependemos de
   que cada persona recuerde cómo ordenar imports o aplicar formateo.
2. **Chequeo de tipos**: se activó mypy para revisar que las funciones usen los
   tipos de datos esperados. Esto ayuda a encontrar errores antes de ejecutar el
   código.
3. **Cobertura de tests**: al ejecutar los tests se calcula cuánta parte del
   código está cubierta. Si la cobertura baja de un umbral, la tubería de CI
   falla para que no se pierda calidad.
4. **Pre-commit**: se añadieron ganchos (hooks) que corren isort, Black y Ruff
   antes de permitir un commit. Así evitamos que entren cambios con imports
   desordenados o estilo inconsistente.
5. **CI en GitHub Actions**: se creó un workflow que ejecuta lint, chequeo de
   tipos y tests con cobertura. También publica el reporte de cobertura como
   artefacto para poder descargarlo y revisarlo.

En resumen, todo esto automatiza tareas repetitivas (ordenar imports, formatear,
revisar tipos y medir cobertura) para que el equipo se concentre en la lógica de
negocio y tenga menos sorpresas en producción.
