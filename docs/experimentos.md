# Experimentos de robustez y siguientes pasos

Lista de experimentos sobre los resultados principales (Preguntas 1 y 2), con qué prueba cada uno, cómo correrlo y su estado. Ninguno cambia los resultados principales ni los archivos del pipeline: cada uno escribe en archivos propios. Los contrastes de esta lista son descriptivos y no entran en la corrección por comparaciones múltiples de la Decisión 17 (los 13 contrastes principales).

## Hechos o en curso

| # | Experimento | Pregunta que contesta | Cómo correrlo | Salida | Estado |
|---|---|---|---|---|---|
| E1 | Robustez por subconjuntos y beta con validación cruzada | ¿La conclusión depende de un año, del choque arancelario de abril de 2025, del horario del anuncio o de una empresa? ¿El beta estimado con los mismos datos infla el resultado de la Pregunta 2? | `python scripts/13_robustness.py` | `results/robustness_q1.csv`, `results/robustness_q2.csv` | Código y pruebas listos; se corre con la muestra final |
| E2 | Bates con θ libre y con κ más amplio | ¿La ventaja (o falta de ella) de Bates sobre Merton cambia si θ no se fija en la varianza del vencimiento lejano? | `python scripts/14_bates_variants.py --workers 5` | `data/processed/bates_variants.csv`, `results/bates_variants_q1.csv` | Lanzado; es reanudable |
| E3 | Control QQQ en lugar de SPY | ¿El crush gradual sobrevive a otro control de mercado (más cercano a las tecnológicas)? | `python scripts/10_control.py --symbol QQQ --strike-range 40`, luego `python scripts/15_alt_control.py --control QQQ` | `results/q2_*_QQQ.csv` | Script listo; falta la descarga (Theta Terminal) |

## Ideas pendientes (no implementadas)

1. **Placebo de ventana.** Repetir la Pregunta 2 en días sin anuncio de las mismas empresas (misma distancia al vencimiento). Si también aparece una "caída extra", parte del efecto es decaimiento normal de la volatilidad implícita cerca del vencimiento y no crush gradual.
2. **Beta por empresa o por sector.** Hoy el beta del control es único. Estimarlo por empresa (con shrinkage hacia el global) o por sector.
3. **Sensibilidad al filtro de calidad.** Repetir con filtros de spread bid-ask más estrictos y con otras ventanas de strikes alrededor del dinero para la volatilidad ATM.
4. **Más años y más empresas.** La tasa libre de ThetaData empieza en 2024-01-01, así que se pueden agregar los anuncios de 2024 de las mismas empresas. Después, más empresas (por ejemplo GOOGL, AMD, COST, BAC) con la misma verificación de dos fuentes.
5. **Merton con saltos genéricos.** Comparar contra un Merton con λ_c libre, para ver cuánto del resultado depende de fijar λ_c = 0.
6. **Prueba de signo exacta y bootstrap por fecha.** Alternativas al Wilcoxon, sin depender de simetría.
7. **Corrección jerárquica (Holm con la familia ampliada).** Si se agregan E1 a E3 a los resultados reportados, decidir antes qué familia corrige a cuál.

## Reglas para agregar experimentos

- Un script nuevo por experimento, con la salida en archivos con nombre propio.
- No modificar los módulos del pipeline para un experimento: usar módulos nuevos o parámetros.
- Documentar aquí la hipótesis antes de ver el resultado, y reportar también los resultados que no confirman.
