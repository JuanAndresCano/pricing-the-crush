# Pricing the Crush

Variance swaps, saltos y volatility crush alrededor de earnings: Merton frente a Bates.

Preguntas de investigación:

1. ¿Qué modelo (Merton o Bates, con evento) se acerca más al *fair strike* del mercado alrededor de un anuncio de resultados?
2. ¿Cómo reacciona la estructura temporal de Bates al *vol crush*: el ajuste es instantáneo o gradual?

El marco teórico está en [`docs/marco_teorico.md`](docs/marco_teorico.md) (copia de la teoría de Julián). Las decisiones del proyecto están en [`docs/decisiones.md`](docs/decisiones.md), los experimentos de robustez en [`docs/experimentos.md`](docs/experimentos.md) y un resumen para quien no conoce el proyecto en [`docs/resumen_proyecto.md`](docs/resumen_proyecto.md).

## Estado

Muestra: 79 anuncios de 2025 y 2026 de 10 empresas (AAPL, MSFT, NVDA, AMZN, META, NFLX, TSLA, JPM, WMT, HD), 63 fechas distintas. Cada fecha y hora se verificó con SEC EDGAR y una segunda fuente (`events/earnings_events.csv`). Los datos son cadenas de opciones de fin de día de ThetaData (plan gratis).

Resultados (todos en `results/`, con la corrección de Holm sobre los 13 contrastes principales):

- **Pregunta 1.** Bates queda más cerca del strike del mercado en el 62 % (vencimiento 1) y el 66 % (vencimiento 2) de los eventos. El vencimiento 2 es significativo (p = 0,0001; Holm 0,0004) y el 1 no (p = 0,073). Se mantiene por año, sin el choque arancelario de abril de 2025, por horario del anuncio y quitando cada empresa.
- **Pregunta 2.** El crush no es del todo instantáneo: la volatilidad sigue bajando 3 a 7 puntos porcentuales después del día 0, más de lo que explican el mercado (SPY o QQQ como control) y los modelos. Significativo con los dos modelos y en +1, +2 y +4 días, también con Holm. Un placebo en días sin anuncio no muestra caída extra (+0,3 %, p = 0,65), así que no es decaimiento normal de la volatilidad.
- **Movimiento realizado contra implícito.** El mercado cobró en promedio 6,9 % y la acción se movió 4,8 % (p = 0,0004).
- **Validación fuera de muestra.** Con strikes que no vio en la calibración, Bates ajusta mejor en el 83,5 % de los eventos (p < 0,001) y su strike queda más cerca en el vencimiento 2 (p = 0,001).
- **Monte Carlo.** La simulación reproduce las fórmulas cerradas dentro de 0,2 % en promedio.

Límites: 79 eventos de dos años y 10 empresas, con κ y θ de Bates no identificables con vencimientos tan cortos (θ se fija). Ver la sección de límites del informe y `docs/decisiones.md`.

## Instalación

Python 3.11 o superior (se usa 3.13).

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
pytest
```

Para descargar datos se necesita el Theta Terminal v3 abierto (`java -jar ThetaTerminalv3.jar`, con una cuenta gratuita de ThetaData). Las credenciales las maneja el Terminal: no se guardan en este repositorio. `.env.example` y `MASSIVE_API_KEY` quedaron de una fuente de datos que no se usó (Massive).

## Estructura

```text
src/crush/
  data/          fuentes de datos (ThetaData; Massive sin usar), calendario, selección de contratos, controles, días extra, placebo
  market/        cadenas, forward por paridad, volatilidad implícita, strike del mercado, tasas
  models/        black_scholes, merton, heston, bates, event, event_models, fourier
  calibration/   calibración por mínimos cuadrados ponderados por vega, selección, variantes de Bates
  simulation/    merton_mc, bates_qe, jumps, rv
  analysis/      compare, crush, gradual, realized, multiple (Holm), robustness, placebo, beta_groups, oos, montecarlo
scripts/         pipeline numerado, ver abajo
events/          calendario de anuncios verificado (earnings_events.csv) y su LEEME
docs/            teoría, decisiones, experimentos, resumen
results/         tablas que sí se versionan
data/            datos crudos e intermedios (fuera de git)
tests/           pruebas
```

## Reproducir los resultados

Con el Theta Terminal abierto (límite del plan gratis: 30 pedidos por minuto, así que la descarga completa tarda horas):

```powershell
python scripts/01_download.py                    # cadenas previa y posterior de cada anuncio verificado
python scripts/08_extra_days.py                  # días +1, +2 y +4
python scripts/10_control.py                     # SPY, para descontar el mercado
python scripts/03_calibrate.py --starts 3        # Merton-evento y Bates-evento (lento: horas)
python scripts/04_compare.py                     # Pregunta 1
python scripts/05_crush.py                       # crush del día 0 contra el modelo
python scripts/07_realized.py                    # movimiento realizado contra implícito
python scripts/09_crush_path.py                  # trayectoria posterior
python scripts/11_gradual_test.py                # Pregunta 2: contraste final
python scripts/12_multiple_testing.py            # Holm
```

Experimentos de robustez (cada uno escribe en archivos propios y no cambia los resultados principales):

| Script | Experimento |
|---|---|
| `13_robustness.py` | subconjuntos y beta con validación cruzada |
| `14_bates_variants.py` | Bates con θ libre y κ más amplio |
| `15_alt_control.py` | control QQQ (antes `10_control.py --symbol QQQ --strike-range 40`) |
| `16_placebo_download.py`, `17_placebo.py` | placebo de ventana |
| `18_beta_groups.py` | beta por empresa y por sector |
| `19_oos.py` | validación fuera de muestra |
| `20_monte_carlo.py` | Monte Carlo sobre los eventos calibrados |

Los scripts `00_smoke_massive.py` y `06_smoke_real.py` son pruebas de acceso a las fuentes de datos. Detalle y hallazgos de cada experimento en `docs/experimentos.md`.

## Flujo de trabajo

Se trabaja directo sobre `main`. Las pruebas deben pasar antes de cada commit. Los `git push` los hace cada persona.
