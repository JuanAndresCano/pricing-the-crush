"""Variantes de la calibración de Bates-evento para probar si la conclusión de la Pregunta 1 depende de cómo se fija θ.

Decisión 10: con vencimientos de pocos días κ y θ no se identifican, así que la calibración principal fija θ en la
varianza ATM del vencimiento más lejano. Aquí se repite la calibración de Bates con otras elecciones, sobre la misma
cadena y con el mismo evento:

  - `theta_libre`:   θ se estima con los demás parámetros (cotas por defecto).
  - `kappa_ancha`:   θ fijo como en la principal, pero κ con cotas más amplias (0,01 a 60 en vez de 0,1 a 30).
  - `libre_ancha`:   θ libre y κ con cotas amplias.

Cada variante devuelve las mismas columnas que `calibration_row` para Bates (con el sufijo del nombre de la variante),
de modo que `compare.summarize_q1` pueda compararla con Merton cambiando solo los nombres de las columnas.
"""

from __future__ import annotations

import numpy as np

from crush.calibration.calibrate import calibrate
from crush.calibration.real import chain_from_slices, theta_from_far
from crush.market.chain import build_slices
from crush.market.rates import event_rates
from crush.market.strike_mkt import k_repl_smooth
from crush.models.event_models import BatesEvent

VARIANTS = ("theta_libre", "kappa_ancha", "libre_ancha")
WIDE_KAPPA = (0.01, 60.0)


def _model(wide_kappa: bool) -> BatesEvent:
    m = BatesEvent()
    if wide_kappa:
        b = list(m.bounds)
        b[1] = WIDE_KAPPA
        m = BatesEvent(bounds=tuple(b))
    return m


def bates_variant_row(data, r: float, variant: str, n_starts: int = 3, seed: int = 0) -> dict | None:
    """Calibra Bates con la variante `variant` y devuelve una fila con ajuste, parámetros y strikes. None si no hay datos."""
    if variant not in VARIANTS:
        raise ValueError(f"Variante desconocida: {variant}. Opciones: {VARIANTS}")
    m = data.meta
    slices = build_slices(data.pre, m["pre_day"], m["spot_pre"], event_rates(m, r)[0], None)
    if len(slices) < 2:
        return None
    chain = chain_from_slices(slices)
    model = _model(wide_kappa=variant in ("kappa_ancha", "libre_ancha"))
    fixed = None
    if variant == "kappa_ancha":
        th, _ = theta_from_far(slices)
        fixed = {"theta": float(np.clip(th, model.lower[2], model.upper[2]))}
    res = calibrate(model, chain, n_starts=n_starts, seed=seed, fixed=fixed)
    row = {"ticker": m["ticker"], "announcement_date": m["announcement_date"], "variant": variant,
           "rmse_iv": res.rmse_iv, "bic": res.bic, "eps_repl": res.model.eps_repl(res.theta)}
    row.update({f"p_{k}": v for k, v in res.params().items()})
    for i, s in enumerate(slices[:2], start=1):
        k_mod = res.model.k_repl(res.theta, s.T, True)
        k_mkt = k_repl_smooth(s.K, s.iv, s.T, s.r, s.F, extrapolate=True)
        row[f"k_{i}"], row[f"k_mkt_{i}"], row[f"T_{i}"], row[f"relerr_{i}"] = k_mod, k_mkt, s.T, k_mod / k_mkt - 1.0
    return row
