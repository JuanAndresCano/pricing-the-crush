# Calendario de anuncios

**Estado:** `earnings_events.csv` tiene 80 filas (10 empresas, 2025 y 2026), 79 verificadas con dos fuentes (SEC EDGAR 8-K Item 2.02 y la página de relaciones con inversionistas). La de JPM del 2025-01-15 quedó con `verified=False` por no tener segunda fuente y el código la ignora. La plantilla de abajo es el punto de partida original (40 filas vacías); el texto que sigue describe cómo se llenó.

`earnings_events_template.csv` es la plantilla: 10 empresas × 4 trimestres = 40 filas, de las que se necesitan unas 30. Las fechas están **vacías a propósito**: no se inventan, se verifican. El archivo que lee el código es `earnings_events.csv`; copia ahí las filas que completes.

## Por qué estas empresas y estas ventanas

- **Ventana:** el plan gratis de ThetaData devolvió cadenas completas de AAPL con bid y ask de fechas desde agosto de 2023 (probado el 23-nov-2026 con 2023-08-01, 2025-06-03 y 2025-11-24), incluidos contratos ya vencidos. La documentación dice que el primer acceso es 2023-06-01. Entonces se pueden elegir anuncios desde mediados de 2023, no solo los últimos 12 meses (esa restricción es del crédito de Databento, que queda de respaldo). Más trimestres permiten más fechas distintas por empresa.
- **Opciones semanales:** el modelo necesita vencimientos de 5 a 10 días hábiles que crucen el anuncio. Todas estas empresas tienen vencimientos semanales y spreads angostos.
- **Mezcla `amc` / `bmo`:** 7 tecnológicas reportan después del cierre y 3 (JPM, WMT, HD) antes de la apertura, para no depender de un solo tipo de evento.
- **Fechas distintas:** el contraste de Wilcoxon se hace por fecha, así que importa tener muchas fechas diferentes. Evita llenar solo con los días en que reportan varias a la vez.

## Cómo completar cada fila

1. `announcement_date`: la fecha del comunicado de resultados, en formato `AAAA-MM-DD`. Debe ser un día hábil de la NYSE.
2. `timing`: `bmo` si se publicó antes de abrir el mercado, `amc` si fue después del cierre. La plantilla trae el valor habitual de cada empresa, pero hay que confirmarlo con la hora del comunicado.
3. `source_a` y `source_b`: dos fuentes **independientes** para la fecha y la hora. Por ejemplo, el comunicado en la página de relaciones con inversionistas de la empresa y el formulario 8-K (Item 2.02) en SEC EDGAR, que trae su propia fecha de envío.
4. `verified`: `true` solo cuando las dos fuentes coinciden.
5. Si una fecha no se puede verificar, borra la fila en vez de dejarla.

`scripts/01_download.py` ignora las filas con `verified` distinto de `true`, salvo que se pase `--include-unverified`.
