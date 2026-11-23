# Pricing the Crush

Variance swaps, saltos y volatility crush alrededor de earnings: Merton frente a Bates.

Preguntas de investigación:

1. ¿Qué modelo (Merton o Bates, con evento) se acerca más al *fair strike* del mercado alrededor de un anuncio de resultados?
2. ¿Cómo reacciona la estructura temporal de Bates al *vol crush*?

El marco teórico está en [`docs/marco_teorico.md`](docs/marco_teorico.md). Las decisiones del proyecto se registran en [`docs/decisiones.md`](docs/decisiones.md).

## Instalación

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
copy .env.example .env   # y completa MASSIVE_API_KEY
pytest
```

## Estructura

```text
src/crush/
  data/          cliente de la API, calendario, tasas, filtros de calidad
  models/        black_scholes, merton, heston, bates, event, fourier
  market/        forward, implied_vol, smile, strike_mkt
  calibration/   loss, calibrate, selection
  simulation/    merton_mc, bates_qe, rv
  analysis/      metrics, stats, figures
scripts/         pipeline numerado (01_download ... 05_report)
tests/           pruebas de validación de la teoría
results/         tablas y figuras que sí se versionan
data/            datos crudos e intermedios (fuera de git)
```

## Reproducir los resultados

Pendiente: se completa cuando existan los scripts de `scripts/`.
