# Registro de decisiones

| # | Decisión | Estado | Notas |
|---|----------|--------|-------|
| 1 | Fuente de datos de opciones | Probable: API de Massive | Confirmar que el plan incluye cotizaciones históricas con bid y ask, y el rango de fechas |
| 2 | Universo de eventos | Pendiente | 10 a 15 empresas líquidas, 3 o 4 trimestres, unos 30 eventos como mínimo |
| 3 | Calendario de anuncios | Pendiente | CSV versionado con ticker, fecha y antes de apertura o después de cierre. Verificar en dos fuentes |
| 4 | Modelo a calibrar | Propuesto | Merton y Bates con evento, Λ_e fijo y λ_c = 0 |
| 5 | Tasa libre de riesgo y dividendos | Propuesto | Tasa de letras del Tesoro de FRED. El forward sale de la paridad put-call |
| 6 | Herramientas | Decidido | Python 3.13, numpy, scipy, pandas, pyarrow, matplotlib, pytest |
| 7 | Idioma | Propuesto | Código en inglés, texto del proyecto en español |
| 8 | Fecha de entrega y formato final | Pendiente | |
| 9 | Flujo en git | Decidido | Una rama por tarea, pull request revisado, `main` con pruebas en verde. Los push los hace cada persona |
