# Plan de trabajo autónomo: validar el flujo con datos reales

Fecha: 2026-11-23. Contexto: ThetaData gratis ya responde con cadenas de opciones reales (bid y ask de cierre, desde 2023, contratos vencidos incluidos). Falta la lista de eventos verificada, que depende del equipo. Mientras tanto, este plan prueba el código contra datos reales sin necesitar esa lista.

## Reglas de este trabajo

- **No se gasta dinero.** Solo el plan gratis de ThetaData, con ritmo de unos 19 pedidos por minuto. No se crean cuentas nuevas, no se usa Databento ni Massive.
- **Sin push.** Todo queda en una rama local, `autonomo/datos-reales`, con un commit por tarea y sin línea de coautoría.
- **Cambios aditivos.** Código nuevo en archivos nuevos. Las ediciones a código existente son mínimas y están en su propio commit.
- **Los datos de prueba no se mezclan con los reales.** Van en `data/smoke/` y `data/raw_api/theta/`, ambos ignorados por git.
- **No se toca la teoría ni los modelos.** Solo el flujo de datos, la limpieza y su documentación.

## Cómo volver atrás

```powershell
git switch main                          # vuelve al estado que ya tenían
git branch -D autonomo/datos-reales      # borra todo lo hecho aquí (solo si no sirve)
```

Para descartar una tarea suelta: `git revert <commit>`. Los archivos descargados no son parte de git: se borran con `Remove-Item -Recurse data\smoke`.

## Tareas, con pros y contras

| # | Tarea | Pros | Contras | Decisión |
|---|---|---|---|---|
| 1 | Prueba de punta a punta con AAPL real: descargar, armar las cadenas pre y post, pasar por la fase de mercado | Valida formatos reales; descubre errores antes de gastar tiempo en eventos reales; no cuesta nada | No es un anuncio real: los resultados económicos no significan nada. Depende de que el Terminal esté abierto | **Hacer**, con un evento "de mentira" claramente marcado |
| 2 | Revisar con datos reales los umbrales de limpieza (spread, bid positivo, mínimo de puntos) | Con cotizaciones reales se ve cuántas se descartan | Riesgo de ajustar a una sola acción. Cambiar los valores por defecto sin base puede empeorar todo | **Hacer solo la medición**; cambiar un valor por defecto únicamente si algo está roto |
| 3 | Calibrar Merton-evento y Bates-evento sobre esa cadena real | Muestra el tiempo real de ejecución y si el optimizador converge con ruido de mercado | Lento (minutos). El "evento" es de mentira, así que los parámetros no se interpretan | **Hacer**, solo para medir ajuste, tiempo y estabilidad |
| 4 | Tasa libre de riesgo real (endpoint de tasas de ThetaData, gratis) | Reemplaza el 0,04 fijo por la tasa del día | Código extra para un efecto pequeño. Endpoint sin verificar | **Hacer**, como módulo opcional con el 0,04 como respaldo |
| 5 | Verificar fechas de anuncios (SEC, relaciones con inversionistas) | Quita trabajo manual al equipo | Es responsabilidad del equipo; fechas mal verificadas contaminan todo; el scraping de SEC pide identificación de contacto | **No hacer** |
| 6 | Crear cuenta en Databento y descargar | Respaldo de datos | Crear cuentas requiere permiso; gasta crédito de un solo uso | **No hacer** |
| 7 | Ampliar la plantilla a más trimestres | Más fechas para elegir | Sin riesgo; solo es un archivo | Se deja como sugerencia |

## Evaluación

El plan es sólido porque todo es reversible y solo lee datos gratis. El peor caso es perder tiempo de cómputo o descubrir que el formato real difiere de lo documentado, y eso es justo lo que queremos saber antes de depender de él. No hay ningún paso que toque dinero, claves, cuentas o la rama principal.

Límites que me impongo: máximo unos 40 pedidos en total a ThetaData, parar y documentar si algo falla dos veces, y no interpretar resultados económicos de un evento de mentira.

## Resultado (23-nov-2026)

Tareas 1 a 4 hechas, cada una en su commit sobre la rama `autonomo/datos-reales`. No se gastó dinero (solo plan gratis de ThetaData, unos 35 pedidos) y no se hizo push. Los hallazgos están en `docs/decisiones.md`, secciones "Hallazgos con datos reales", decisión 11 (piso de spread en dólares) y decisión 12 (tasa libre de riesgo). Se verificó con un evento real: el flujo completo, desde la descarga hasta el crush del modelo, funciona y da números coherentes. Quedan para el equipo las fechas verificadas de los anuncios.
