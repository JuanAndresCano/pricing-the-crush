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


## Hallazgos con datos reales (ThetaData, plan gratis)

Prueba del 23-nov-2026 con la cuenta gratis y el Theta Terminal v3 (unos 35 pedidos en total). No son resultados del estudio: el evento de prueba (AAPL, anuncio supuesto el 2026-11-02 después del cierre) se eligió porque los datos lo muestran como un anuncio, y su fecha y hora **no están verificadas** con fuentes externas.

**Acceso**

- `Stock`, `Options`, `Index` y `Rate` salen como `FREE`. El plan gratis devuelve cadenas completas de fin de día con `bid`, `ask` y sus tamaños, incluidas fechas de 2023 (2023-08-01, 2025-06-03, 2025-11-24 y 2026) y contratos ya vencidos. La restricción de "12 meses" era de Databento, no de ThetaData.
- Límites: 30 pedidos por minuto y **1 pedido a la vez** (`Max concurrent requests: 1`). Una cadena completa de una empresa y un día es un solo pedido. El cliente espera 3,1 s entre pedidos.
- El servidor del Terminal escucha por defecto en `0.0.0.0`; se cambió a `127.0.0.1` en `config.toml`.
- Las tasas (`interest_rate/history/eod`) son gratis solo desde 2024-01-01. Antes de esa fecha el servidor responde 400 y se usa la tasa de respaldo (4 %).
- El reporte de fin de día no trae la hora de cada cotización: no se puede medir su antigüedad.

## Decisión 11: piso de spread en dólares para las alas

Con el filtro de spread relativo de 0,5 solo, se descartaba la mayor parte de las opciones a más del 5 % del precio: son opciones de pocos centavos y un spread de $0,02 ya es el 100 %. Se agregó `min_abs_spread = 0,05` (dólares): un spread de hasta ese tamaño se acepta siempre, y el bid sigue debiendo ser positivo.

Efecto medido en AAPL (vencimiento más corto): la diferencia entre el strike sin y con extrapolación pasa de 3,2 % a 0,7 %, la volatilidad ATM no cambia y el rango de strikes cubierto sube de [-0,08; 0,04] a [-0,18; 0,06] en moneyness logarítmica. Es una regla de tamaño de tick, no un ajuste a una empresa; conviene revisarla con otros precios de acción.

## Decisión 12: tasa libre de riesgo

La tasa de cada día sale del bono del Tesoro a 1 mes (`TREASURY_M1`) de ThetaData, guardada en la meta del evento como `r_pre` y `r_post`. Si no está disponible se usa 4 %. Con las tasas reales (4,0 % a 4,1 %) el cambio frente al 4 % fijo es mínimo, como se esperaba.

## Prueba de punta a punta sobre un evento real

Datos: AAPL, cierre del 2026-11-02 y del 2026-11-03.

| Medida | Valor |
|---|---|
| Volatilidad ATM del vencimiento de 4 días, antes / después | 48,3 % / 27,9 %, razón observada CR = 0,577 |
| Varianza de evento del mercado (26), eps | 0,0022 (ATM) y 0,0030 (strike); movimiento implícito de 4,7 % a 5,5 % |
| Control en la cadena posterior (eps) | 0,0001, cercano a cero como debe ser |
| Movimiento realizado de la acción | -1,9 % (cierre a cierre) |

Calibración (3 arranques, θ fijo en la varianza ATM del vencimiento largo = 0,088):

| | Merton-evento | Bates-evento |
|---|---|---|
| RMSE_IV | 2,44 pts | 2,25 pts |
| BIC | -679,6 | -681,5 |
| Error de K_repl, vencimiento de 4 días | +1,2 % | +0,5 % |
| Error de K_repl, vencimiento de 5 días | -1,0 % | -0,8 % |
| CR del modelo (vencimientos de 4 y 5 días) | 0,48 y 0,52 | 0,53 y 0,56 |

- Ambos modelos quedan cerca del strike del mercado (menos de 1,5 %). Bates mejora poco el ajuste y apenas gana por BIC con un solo evento.
- Ambos **sobreestiman el crush** (0,48 a 0,56 contra 0,58 observado). Casi todo el crush del modelo viene de la salida del evento; la reversión de la varianza de Bates es ≈ 1, porque κ queda en su cota inferior (0,1).
- Cada calibración tardó unos 3 minutos con 3 arranques.
- Con un día no anunciado (2026-11-18) los parámetros del evento van a sus cotas (Λ_e = 0,2 en el mínimo, ξ = 3 en el máximo) y el RMSE_IV sube a 3 a 6 puntos: sin evento en la cadena, el modelo con evento no tiene nada que explicar.

Qué falta: más eventos, con fechas y horas verificadas; repetir con θ libre para ver cuánto cambian los parámetros; y revisar las cotas de κ y ξ.


## Decisión 13: movimiento realizado contra implícito

Para cada evento: `R_e = ln(S_post / S_pre)`, con los dos cierres del evento (los mismos que definen `pre_day` y `post_day`, según `bmo`/`amc`), contra la varianza de evento ε de la ecuación (26). Se reportan con ε de dos formas (ATM y strike de replicación):

- movimiento implícito `sqrt(ε)` y realizado `|R_e|`;
- `R_e² / ε` y `z = |R_e| / sqrt(ε)` (cuántos "movimientos implícitos" se movió la acción);
- `ε - R_e²`, prima de varianza del evento: positiva si el mercado cobró más de lo que se movió la acción;
- el valor del variance swap comprado antes del anuncio, justo después de que ocurre.

**Valor del swap.** Con la ecuación (35), `RV_{0,t} = R_e²/t`, `K_{t,T} = σ_b²` (el evento ya no está) y `K_var = σ_b² + ε/T` sale exacto `V = N (R_e² - ε - σ_b² t) / T`. La ecuación (36) del marco teórico es la misma sin el término `σ_b² t`, que es la varianza base de ese día. Con t = 1 día es chico, pero no cero; el código usa la exacta y se puede pasar `sigma_b = 0` para tener (36). La prueba `test_event_swap_value_matches_equation_35` comprueba la identidad contra `simulation.rv.swap_value`.

**Historia propia.** `add_history` calcula el promedio de `R_e²` de los anuncios anteriores de la misma empresa (contraparte bajo la medida física), solo con eventos de fecha previa y con al menos 2 previos.

**Contraste.** Wilcoxon sobre `ε - R_e²` agrupado por fecha del anuncio, igual que en la Pregunta 1. Se excluyen los eventos con `|R_e| > 0,5` (posible split sin ajustar). Con menos de 6 fechas no se reporta p-valor.

**Prueba con datos reales (no es resultado del estudio).** El evento de AAPL del 2026-11-02 (fecha no verificada): movimiento implícito de 4,7 % contra 1,9 % realizado, `R_e²/ε = 0,17`, `z = 0,41`; el comprador del swap habría perdido (`V ≈ -0,14` por unidad de nocional). Las otras tres fechas de prueba (2026-10-29, 11-03 y 11-18) no son el día del anuncio y sirven de contraste: el movimiento realizado no se parece al implícito (`z` de 0,31, 4,53 y 1,05), lo que muestra cuánto depende todo de tener la fecha y la hora correctas.

Se corre con `python scripts/07_realized.py` (los eventos de prueba, con `--root data/smoke`).

## Decisión 14: crush instantáneo o gradual con días posteriores al anuncio

**Descarga.** `scripts/08_extra_days.py` agrega, para los mismos contratos de cada evento, los cierres 1, 2 y 4 días hábiles después del cierre posterior (el cierre posterior es el día 0). Se guardan al lado del evento como `post_p<k>.parquet` y su día, precio de la acción y tasa quedan en `meta.json`, bajo `extra_days`. No toca `pre.parquet` ni `post.parquet`, es reanudable (lo ya pedido no se vuelve a pedir) y omite los días que aún no existen. Cuesta unos 3 pedidos por día extra y evento (≈ 14 minutos para 30 eventos).

**Análisis.** `analysis.crush.path_table` calcula, para cada vencimiento que sigue vivo (al menos 3 días hábiles), la razón de crush observada y la del modelo en cada día, y la **continuación** desde el día 0: `g = IV(k) / IV(0)`.

- Si el crush es instantáneo, `g_obs ≈ 1` después del día 0.
- Si la volatilidad sigue bajando, `g_obs < 1`.
- `g_gap = g_obs - g_modelo` es la caída extra sobre lo que el modelo ya espera. Merton-evento predice `g ≈ 1`; Bates-evento predice una caída gradual si `v0 > θ` (reversión hacia θ a velocidad κ).

Las pruebas con mercados simulados lo distinguen: si el mercado lo generó Merton-evento, `g_gap` es ≈ 0 para Merton; si lo generó Bates con `v0 ≫ θ`, la volatilidad baja después del día 0, Merton falla (`g_gap` negativo) y Bates acierta.

**Prueba con datos reales (no es resultado del estudio).** AAPL con el anuncio supuesto del 2026-11-02 (fecha no verificada), vencimiento del 2026-12-04, `g_obs` a +1, +2 y +4 días: 0,91, 1,09 y 0,90. Los dos modelos predicen ≈ 1,00. La volatilidad ATM del vencimiento largo se movió entre 22 % y 27 % en cuatro días, sin una tendencia de caída: ese ruido es de movimientos generales del mercado, no del anuncio.

**Límite principal.** El `g_obs` de un solo evento queda dominado por la volatilidad general del mercado, que cambia del orden de ±10 % en pocos días. Con 30 eventos hay que promediar entre ellos, y para aislar el efecto del anuncio conviene **descontar el movimiento del mercado** (por ejemplo, con la volatilidad implícita de una acción sin anuncio ese día, o de un ETF como SPY, cuyas opciones también entran en el plan gratis). Eso queda como la mejora siguiente; no está implementado.

Se corre con `python scripts/09_crush_path.py` después de la calibración y de `08_extra_days.py`.


## Decisión 15: control de mercado con SPY para la trayectoria del crush

**Por qué.** La volatilidad de una acción se mueve día a día por la del mercado entero, y la decisión 14 mostró que ese ruido (±10 % en pocos días) tapa cualquier efecto del anuncio. Se descuenta con la volatilidad implícita ATM de SPY sobre el vencimiento más cercano en plazo al de la acción (el mismo vencimiento de SPY en todos los días del evento).

**Descarga.** `scripts/10_control.py` guarda la cadena de SPY de cada día que usa un evento, en `data/control/theta/SPY/`, compartida entre eventos (un día ya guardado no se pide de nuevo). Cuesta unos 3 pedidos por día nuevo.

**Error encontrado y corregido.** Con `strike_range = 15`, el 2026-11-05 la cadena de SPY salió sin strikes cerca del dinero: ThetaData centra la ventana en el precio del día anterior, no en el cierre del día pedido, y SPY se había movido más de 15 puntos. Sin el control habría pasado sin avisar (casi todos los vencimientos quedaban sin datos). Ahora: ventana por defecto de 60 strikes y `ensure_control_day` rechaza, y no guarda, cualquier cadena que no cubra el precio de cierre (`covers_spot`). Ese mismo riesgo no afecta a las cadenas de los eventos, que se piden sin `strike_range`.

**Beta.** Suponer que la acción reacciona igual que el mercado (beta 1) sobrecorrige. Con el evento de prueba de AAPL, SPY cambió su volatilidad del vencimiento de diciembre entre 13 % y 22 % en cinco días (16,3 % el 3-nov, 13,1 % el 4, 21,9 % el 5, 15,2 % el 9), y la de AAPL se movió mucho menos. `estimate_control_beta` estima la sensibilidad con la regresión `ln g_obs = alpha + beta ln g_spy`; `alpha` es la caída promedio que queda después de descontar el mercado (negativa significa que el crush continúa tras el día 0). `path_table(..., control_beta=...)` la usa; `scripts/09_crush_path.py` la estima con todos los eventos si no se le pasa `--beta`.

**Prueba con datos reales (no es resultado del estudio).** AAPL con el anuncio supuesto del 2026-11-02: beta ≈ 0,23 y alpha ≈ -0,02 con solo 4 observaciones de un evento, así que solo dan el orden de magnitud. Tras descontar el mercado, la desviación de `g` respecto del modelo pasa de entre -0,10 y +0,09 a entre -0,09 y +0,03; el ruido baja pero sigue ahí con un evento. El 2026-11-03 fue día de elecciones de medio período en EE. UU., lo que podría explicar la caída de SPY al día siguiente; es una hipótesis sin verificar.

**Límites.** El beta se supone constante y común a todas las acciones; SPY no es sectorial (para tecnológicas, un ETF como QQQ podría ser un mejor control). Con 30 eventos el beta se estima con más datos, pero el contraste final (¿`alpha` distinto de cero?) hay que hacerlo agrupando por fecha de anuncio, igual que en las demás pruebas, y todavía no está implementado.
Se corre con `python scripts/09_crush_path.py` después de la calibración y de `08_extra_days.py`.
