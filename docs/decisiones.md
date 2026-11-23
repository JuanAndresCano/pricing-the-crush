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
| 10 | Vencimientos a descargar y tratamiento de κ y θ | Propuesto | Incluir al menos un vencimiento de 21 días o más, idealmente uno de 40 a 60. Fijar θ con la volatilidad ATM de largo plazo o regularizar la calibración. No interpretar κ ni θ por separado. Ver hallazgos abajo |

## Hallazgos de la calibración sintética (prueba 10)

Se generaron cadenas de opciones con el modelo Bates-evento (parámetros conocidos: v₀ = 0,09, κ = 6, θ = 0,0625, ξ = 0,8, ρ = −0,6, Λ_e = 1,5, μ_e = −0,02, σ_e = 0,07), se les agregó ruido y se recalibró con 3 arranques independientes. Son resultados con datos sintéticos, no con datos reales.

**Sin ruido, con vencimientos de 5, 10 y 21 días:** los 8 parámetros se recuperan exactos en los 3 arranques (error de ajuste de 1e-14 en volatilidad implícita).

**Con ruido de 0,2 puntos de volatilidad y vencimientos de 5, 10 y 21 días:**

| Parámetro | Verdadero | Estimado |
|-----------|-----------|----------|
| v₀ | 0,090 | 0,089 |
| κ | 6,0 | 7,5 |
| θ | 0,0625 | 0,0683 |
| ξ | 0,80 | 0,91 |
| ρ | −0,60 | −0,55 |
| Λ_e | 1,50 | 1,507 |
| μ_e | −0,020 | −0,020 |
| σ_e | 0,070 | 0,070 |

Los parámetros del evento y v₀ se recuperan casi exactos. Los demás tienen un sesgo moderado pero estable entre arranques. ε_repl queda con error de 0,3% y K_repl de 0,1% a 5, 10 y 21 días.

**Con ruido de 0,2 puntos y solo vencimientos de 5 y 10 días:** κ y θ **no se identifican**. Los 3 arranques logran el mismo ajuste (error de 2,98e-6) con parámetros distintos: κ entre 0,9 y 1,4 (el verdadero es 6) y θ entre 0,167 y 0,212 (el verdadero es 0,0625, es decir, una volatilidad de largo plazo de 41% a 46% en lugar de 25%). Aun así, ε_repl y K_repl a 5 y 10 días se recuperan con error de 0,2% a 1,5%.

**Consecuencias:**

- La Pregunta 1 (qué modelo se acerca más al strike del mercado) no depende de κ ni de θ, porque el strike a los plazos observados se recupera bien.
- La Pregunta 2 (cómo reacciona Bates al crush) sí depende de ellos: la caída gradual de la varianza se debe a la reversión hacia θ a velocidad κ. Con solo plazos de 5 y 10 días no hay base para interpretar esa parte.
- No se debe extrapolar K_repl a plazos más largos que los observados con κ y θ mal identificados.

Prueba automática de la recuperación sin ruido: `tests/test_calibration.py` (la de Bates está marcada como lenta, se corre con `pytest -m slow`).
