# Calendario de anuncios

`earnings_events_template.csv` es la plantilla: 10 empresas × 4 trimestres = 40 filas, de las que se necesitan unas 30. Las fechas están **vacías a propósito**: no se inventan, se verifican. El archivo que lee el código es `earnings_events.csv`; copia ahí las filas que completes.

## Por qué estas empresas y estas ventanas

- **Ventana:** solo anuncios de los últimos 12 meses (desde noviembre de 2025), porque tanto el plan gratis de ThetaData como el crédito de Databento limitan la historia de cotizaciones a ese plazo. Si resulta que alcanza más atrás, se amplía.
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
