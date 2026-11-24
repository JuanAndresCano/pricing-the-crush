# Experimentos de robustez y siguientes pasos

Lista de experimentos sobre los resultados principales (Preguntas 1 y 2), con qué prueba cada uno, cómo correrlo y su estado. Ninguno cambia los resultados principales ni los archivos del pipeline: cada uno escribe en archivos propios. Los contrastes de esta lista son descriptivos y no entran en la corrección por comparaciones múltiples de la Decisión 17 (los 13 contrastes principales).

## Hechos o en curso

| # | Experimento | Pregunta que contesta | Cómo correrlo | Salida | Estado |
|---|---|---|---|---|---|
| E1 | Robustez por subconjuntos y beta con validación cruzada | ¿La conclusión depende de un año, del choque arancelario de abril de 2025, del horario del anuncio o de una empresa? ¿El beta estimado con los mismos datos infla el resultado de la Pregunta 2? | `python scripts/13_robustness.py` | `results/robustness_q1.csv`, `results/robustness_q2.csv` | Hecho con 79 eventos: la ventaja de Bates en la Pregunta 1 y la caída extra de la Pregunta 2 se mantienen en todos los subconjuntos (2025, 2026, sin abril de 2025, amc, bmo, sin cada empresa) y con beta de validación cruzada |
| E2 | Bates con θ libre y con κ más amplio | ¿La ventaja (o falta de ella) de Bates sobre Merton cambia si θ no se fija en la varianza del vencimiento lejano? | `python scripts/14_bates_variants.py --workers 5` | `data/processed/bates_variants.csv`, `results/bates_variants_q1.csv` | Hecho con 79 eventos: con θ libre Bates se acerca a Merton (p = 0,044) y no mejora a Bates principal (p = 0,35); con κ más ancho gana a Merton (p = 0,0001), pero κ queda en la cota inferior en el 54 % de los eventos |
| E3 | Control QQQ en lugar de SPY | ¿El crush gradual sobrevive a otro control de mercado (más cercano a las tecnológicas)? | `python scripts/10_control.py --symbol QQQ --strike-range 40`, luego `python scripts/15_alt_control.py --control QQQ` | `results/q2_*_QQQ.csv` | Hecho: beta de QQQ 0,50; caída extra de Bates −5,3 % y de Merton −3,9 % (p < 0,001), casi igual que con SPY (−5,5 % y −4,0 %) |
| E4 | Placebo de ventana | ¿La caída extra después del día 0 es crush gradual o decaimiento normal de la volatilidad implícita cerca del vencimiento? | `python scripts/16_placebo_download.py` (descarga, Theta Terminal abierto), luego `python scripts/17_placebo.py` | `results/placebo_groups.csv`, `results/placebo_effect.csv`, `results/placebo_table.csv` | Hecho con 73 eventos y 58 fechas: placebo +0,3 % (p = 0,65), real −4,2 %, diferencia −4,5 % (p < 0,001). La hipótesis se cumple |
| E5 | Beta del control por empresa y sector | ¿El beta único del control es confiable o varía por sector? | `python scripts/18_beta_groups.py` | `results/beta_by_sector.csv`, `results/beta_by_ticker.csv`, `results/beta_groups_tests.csv` | Hecho: los betas difieren (0,20 a 0,67; igualdad rechazada, p = 0,007 por sector) pero con 7 u 8 eventos por empresa son imprecisos; la caída extra no cambia (−4,05 % contra −4,03 %) |
| E6 | Validación fuera de muestra | ¿La ventaja de Bates se mantiene con strikes que no vio en la calibración? | `python scripts/19_oos.py --workers 5` | `data/processed/oos.csv`, `results/oos_summary.csv` | Hecho con 79 eventos: Bates ajusta mejor los strikes que no vio en el 83,5 % de los eventos (RMSE de volatilidad implícita 1,75 contra 2,09 puntos, p < 0,001). En el strike, vencimiento 2: Bates más cerca en el 70 % (p = 0,001); vencimiento 1: 59 % (p = 0,20, no significativo). Misma conclusión que dentro de muestra, algo más débil en el vencimiento 2 |
| E7 | Monte Carlo sobre los eventos | ¿Las fórmulas cerradas coinciden con la simulación con los parámetros calibrados? ¿Cómo se distribuye el pago del swap? | `python scripts/20_monte_carlo.py` | `results/mc_events.csv`, `results/mc_summary.csv` | Hecho: diferencia media de 0,06 % a 0,2 %, martingala 1,0; pago medio del comprador cercano a 0 con pérdida en el 69 % a 73 % de los eventos |

## Hipótesis de E4 (escrita antes de ver el resultado)

**Diseño.** Para cada evento se toma un día 0 de placebo 20 días hábiles después del día 0 real (sin anuncio cerca: el siguiente está a unos 40 días hábiles) y se repite la medición de la Pregunta 2 con los días +1, +2 y +4, la misma regla de selección de vencimientos y el mismo control SPY. Como no hay evento, la comparación es contra el análisis real de Merton (`g_model = 1`).

**Hipótesis.** Si la caída extra es crush gradual, el placebo debe dar una caída extra cercana a 0 y la diferencia real menos placebo (coeficiente `c` de la regresión conjunta y Wilcoxon de la diferencia por evento) debe ser negativa y significativa. Si el placebo da una caída parecida a la real (alrededor de −4 %), la Pregunta 2 está captando decaimiento normal y la conclusión de crush gradual se debilita. Un resultado intermedio se reporta tal cual.

**Limitaciones previstas.** Los días restantes de los vencimientos del placebo pueden diferir de los reales (se reporta el promedio y se controla en una regresión); el placebo usa una sola ventana por evento; el contraste es descriptivo y no entra en las 13 pruebas corregidas.

## Ideas pendientes (no implementadas)

1. **Beta con shrinkage.** E5 muestra que los betas por empresa son imprecisos; un beta con shrinkage hacia el global sería la versión razonable si se quisiera usar uno por empresa.
2. **Sensibilidad al filtro de calidad.** Repetir con filtros de spread bid-ask más estrictos y con otras ventanas de strikes alrededor del dinero para la volatilidad ATM.
3. **Más años y más empresas.** La tasa libre de ThetaData empieza en 2024-01-01, así que se pueden agregar los anuncios de 2024 de las mismas empresas. Después, más empresas (por ejemplo GOOGL, AMD, COST, BAC) con la misma verificación de dos fuentes.
4. **Merton con saltos genéricos.** Comparar contra un Merton con λ_c libre, para ver cuánto del resultado depende de fijar λ_c = 0.
5. **Prueba de signo exacta y bootstrap por fecha.** Alternativas al Wilcoxon, sin depender de simetría.
6. **Corrección jerárquica (Holm con la familia ampliada).** Si se agregan E1 a E7 a los resultados reportados, decidir antes qué familia corrige a cuál.

## Reglas para agregar experimentos

- Un script nuevo por experimento, con la salida en archivos con nombre propio.
- No modificar los módulos del pipeline para un experimento: usar módulos nuevos o parámetros.
- Documentar aquí la hipótesis antes de ver el resultado, y reportar también los resultados que no confirman.
