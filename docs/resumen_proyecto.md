# Pricing the Crush

**Variance swaps, saltos y *volatility crush* alrededor de los anuncios de resultados de empresas (*earnings*): Merton frente a Bates.**

Proyecto de investigación en finanzas cuantitativas que compara dos modelos matemáticos de precios de opciones para ver cuál describe mejor lo que el mercado espera de una acción justo antes de que la empresa publique sus resultados trimestrales.

## 1. Pregunta de investigación

En lenguaje claro: antes de un anuncio de resultados, el precio de las opciones sube porque nadie sabe cuánto se moverá la acción ese día; apenas se publican los resultados, esa incertidumbre desaparece y los precios de las opciones caen de golpe (el *volatility crush*). El proyecto plantea dos preguntas:

1. **¿Cuál de los dos modelos (Merton o Bates, ambos con un componente específico para el evento) se acerca más a la medida de incertidumbre que el mercado fija alrededor de un anuncio?** Esa medida se llama *fair strike* de un variance swap, un contrato que paga según cuánto varió la acción.
2. **¿Cómo reacciona el modelo de Bates a la caída de volatilidad posterior al anuncio?** Es decir, si el ajuste ocurre de golpe, como indica la teoría del salto del evento, o de forma gradual en los días siguientes.

## 2. Cómo se hace

- **Valoración con dos modelos.** Se implementan los modelos de Merton (saltos súbitos del precio) y de Bates (saltos más volatilidad que cambia en el tiempo), con un factor adicional que representa el salto del día del anuncio.
- **Datos de mercado.** Se toman cadenas de opciones de acciones individuales de EE. UU. el día antes y el día después de cada anuncio, y se usa el punto medio entre el precio de compra (*bid*) y de venta (*ask*) como precio de referencia, solo con opciones fuera del dinero y con filtros de calidad.
- **Volatilidad implícita.** Cada precio se convierte en la volatilidad que lo explica, para poder comparar opciones distintas entre sí.
- **Calibración.** Se buscan los parámetros de cada modelo que mejor reproducen los precios observados, y se comparan los modelos penalizando que uno tenga más parámetros que el otro.
- ***Fair strike*.** Se calcula la medida de incertidumbre del mercado a partir de las opciones y se compara con la que implica cada modelo calibrado.
- **Simulación (Monte Carlo).** Se simulan miles de trayectorias del precio para verificar que las fórmulas están bien y para ver la distribución completa de resultados. Al final, los resultados de cada evento se comparan entre modelos con pruebas estadísticas agrupadas por fecha y corregidas por comparaciones múltiples (Holm).

## 3. Datos

- **Qué se usó:** cadenas históricas de opciones de acciones individuales de EE. UU. con precio de compra y venta (*bid* y *ask*) al cierre del día, de ThetaData (plan gratuito, que resultó suficiente), para el día antes, el día después y los días +1, +2 y +4 de cada anuncio. Más las cadenas de SPY y QQQ de los mismos días, para descontar el movimiento general del mercado.
- **Cuántos:** 79 anuncios de 2025 y 2026 de 10 empresas (AAPL, MSFT, NVDA, AMZN, META, NFLX, TSLA, JPM, WMT y HD), en 63 fechas distintas. Cada fecha y hora se verificó con SEC EDGAR (8-K) y una segunda fuente, la página de relaciones con inversionistas de la empresa. Un anuncio de JPM (15-ene-2025) se excluyó por no tener segunda fuente.
- **Por qué:** el día antes mide la incertidumbre que el mercado tiene incorporada; el día después, y los siguientes, permiten medir cuánto cae y si sigue cayendo.
- **Límites de los datos:** el reporte de fin de día no trae la hora de cada cotización, y el plan gratuito limita a 30 pedidos por minuto.

## 4. Estado actual

**Hecho y probado (208 pruebas automáticas):** motor de precios (Black-Scholes, Merton, Heston, Bates, factor de evento), calibración, descarga y limpieza de datos reales, simulación Monte Carlo, y todo el análisis de las dos preguntas.

**Resultados con los 79 eventos:**
- **Pregunta 1.** Bates se acerca más al strike del mercado en el 62 % (vencimiento 1) y el 66 % (vencimiento 2) de los eventos. La diferencia es significativa en el vencimiento 2 (p = 0,0001; con corrección por comparaciones múltiples, 0,0004) y no en el 1 (p = 0,073). Se mantiene por año, sin el choque arancelario de abril de 2025, por horario del anuncio y quitando cada empresa.
- **Pregunta 2.** El crush no es del todo instantáneo: la volatilidad sigue bajando unos 3 a 7 puntos porcentuales en los días siguientes, más de lo que explican el mercado y los modelos, con Bates y con Merton. Un placebo en días sin anuncio no muestra esa caída (+0,3 %, p = 0,65), así que no es decaimiento normal de la volatilidad.
- **Movimiento realizado contra implícito:** el mercado cobró en promedio 6,9 % y la acción se movió 4,8 % (p = 0,0004).
- **Monte Carlo:** reproduce las fórmulas cerradas dentro de 0,2 % en promedio.

**Límites:** 79 eventos de dos años y 10 empresas; los parámetros de reversión de Bates (κ y θ) no son identificables con vencimientos tan cortos, así que θ se fija; el beta del control es único para todas las empresas.

**Pendiente:** validación fuera de muestra (en curso), más empresas y los anuncios de 2024, y acordar con el equipo las correcciones al marco teórico. El informe completo está en Claude Docs y los números en `results/`.

## 5. Qué se le pide a la universidad

- Revisión metodológica de los resultados y de las pruebas estadísticas.
- Orientación sobre cómo citar y reportar datos de ThetaData en un trabajo académico (el plan gratuito es para uso no comercial).
- Opcional: acceso a una base como OptionMetrics a través de WRDS para ampliar la muestra a más años y empresas.

---

## Versión corta para correo

Estoy desarrollando el proyecto "Pricing the Crush", que compara los modelos de Merton y Bates para explicar cómo el mercado de opciones valora la incertidumbre alrededor de los anuncios de resultados de empresas y la caída de volatilidad posterior. Con 79 anuncios de 10 empresas de EE. UU. (2025 y 2026), Bates se acerca más que Merton a lo que cobra el mercado en el vencimiento más largo, y la volatilidad sigue bajando unos 3 a 7 puntos porcentuales después del día del anuncio, más de lo que explican el mercado y los modelos; un placebo en días sin anuncio descarta que sea un efecto normal de vencimiento. ¿Sería posible una revisión metodológica de los resultados o acceso a una base como OptionMetrics/WRDS para ampliar la muestra? Con gusto envío el resumen completo.
