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
- **Simulación (Monte Carlo).** Se simulan miles de trayectorias del precio para verificar que las fórmulas están bien y para ver la distribución completa de resultados. Al final, los resultados de cada evento se comparan entre modelos con pruebas estadísticas.

## 3. Qué datos necesita y por qué

- **Qué:** cadenas históricas de opciones de acciones individuales de EE. UU. con precio de compra y venta (*bid* y *ask*) al cierre del día (datos de fin de día son suficientes), para el **día antes y el día después** de cada anuncio de resultados.
- **Cuántos:** unas 30 empresas-evento aproximadamente, entre **noviembre de 2024 y noviembre de 2026**, con 2 o 3 vencimientos cercanos por evento.
- **Complementos:** precio de cierre de la acción y calendario de anuncios con la hora (antes de la apertura o después del cierre), porque define en qué día cae el movimiento.
- **Por qué:** el día antes mide la incertidumbre que el mercado tiene incorporada; el día después permite medir cuánto cae. Sin precios *bid* y *ask* no se puede estimar un precio justo de forma fiable.
- **Tipo de uso:** descarga única de datos históricos para investigación; no se necesita información en tiempo real.
- **Fuentes que se están considerando:** la API de Massive (plan Advanced, de pago), Databento, ThetaData y OptionMetrics a través de WRDS, si la universidad tiene acceso. Precios, cobertura y condiciones de licencia de cada una están por verificar.

## 4. Estado actual

**Hecho y probado con pruebas automáticas (unas 97 pruebas pasando):**
- Motor de precios: Black-Scholes, Merton, Heston, Bates y el factor de evento de resultados.
- Solver de volatilidad implícita.
- Simulación Monte Carlo de Merton y de Bates. Los precios de Bates por Monte Carlo coinciden con los obtenidos por el método de inversión de Fourier.

**En curso:**
- Calibración con datos sintéticos, que consiste en comprobar que el código recupera parámetros conocidos.

**Pendiente:**
- Obtener los datos reales de opciones.
- Calibrar con datos reales y comparar con el mercado.

Todavía **no hay resultados con datos reales**; lo avanzado hasta ahora valida la herramienta, no responde aún las preguntas de investigación.

## 5. Qué se le pide a la universidad

Según lo que sea posible, cualquiera de estas opciones ayudaría:

- Acceso a una base de datos de opciones, por ejemplo OptionMetrics a través de WRDS, si la universidad la tiene.
- Orientación sobre licencias académicas de datos financieros (qué se puede usar, para qué y cómo citarlo).
- Tutoría o acompañamiento académico para revisar la metodología y los resultados.

---

## Versión corta para correo

Estoy desarrollando el proyecto "Pricing the Crush", que compara los modelos de Merton y Bates para explicar cómo el mercado de opciones valora la incertidumbre alrededor de los anuncios de resultados de empresas y la caída de volatilidad posterior. El motor de precios, el cálculo de volatilidad implícita y la simulación ya están implementados y probados; la calibración con datos sintéticos está en curso. Me falta obtener datos históricos de fin de día (bid y ask) de opciones de acciones de EE. UU. para unos 30 anuncios entre noviembre de 2024 y noviembre de 2026. ¿Sería posible acceso a una base como OptionMetrics/WRDS, orientación sobre licencias académicas de datos o una tutoría? Con gusto envío el resumen completo.
