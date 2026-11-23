"""Calibración por mínimos cuadrados ponderados por vega, ecuación (41) con la corrección del piso de vega.

    L(theta) = (1/M) Sum_i ( (C_i^mod(theta) - C_i^mkt) / max(nu_i, nu_min) )^2

Dividir entre el vega hace que cada error valga aproximadamente un error en volatilidad implícita. Con
5 a 10 días de plazo, el vega de las opciones muy OTM es casi cero y la división amplifica el ruido del
bid-ask, así que se usa un piso nu_min = vega_floor * mediana(vega).

Optimización: búsqueda global con `differential_evolution` y refinamiento local con `least_squares`.
Con varios arranques se mide la estabilidad de los parámetros.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import differential_evolution, least_squares

from crush.calibration.quotes import OptionChain, model_prices
from crush.calibration.selection import bic, rmse_iv
from crush.models.event_models import EventModel

BAD = 1e6


@dataclass
class CalibrationResult:
    model: EventModel
    theta: np.ndarray
    loss: float
    rmse_iv: float
    bic: float
    starts: list = field(default_factory=list)  # (loss, theta) de cada arranque

    def params(self) -> dict[str, float]:
        return self.model.as_dict(self.theta)

    def spread(self) -> dict[str, float]:
        """Desviación estándar de cada parámetro entre arranques (0 si hubo uno solo)."""
        if len(self.starts) < 2:
            return {n: 0.0 for n in self.model.names}
        arr = np.array([t for _, t in self.starts])
        return dict(zip(self.model.names, map(float, arr.std(axis=0))))


def vega_weights(chain: OptionChain, vega_floor: float = 0.05) -> np.ndarray:
    v = chain.vega()
    floor = vega_floor * np.nanmedian(v)
    return 1.0 / np.maximum(v, floor)


def calibrate(
    model: EventModel,
    chain: OptionChain,
    n_starts: int = 1,
    seed: int = 0,
    popsize: int = 12,
    maxiter: int = 80,
    vega_floor: float = 0.05,
    refine: bool = True,
) -> CalibrationResult:
    ok = np.isfinite(chain.market_iv())
    chain = chain.subset(ok) if not ok.all() else chain
    w = vega_weights(chain, vega_floor)
    lo, hi = model.lower, model.upper

    def resid(theta):
        # Algunas combinaciones extremas de parámetros dan divisiones por cero en la función característica.
        # Esos precios quedan como nan y se castigan con BAD, así que se silencian las advertencias.
        with np.errstate(all="ignore"):
            r = (model_prices(model, theta, chain) - chain.price) * w
        return np.where(np.isfinite(r), r, BAD)

    def objective(theta):
        return float(np.mean(resid(theta) ** 2))

    starts = []
    for s in range(n_starts):
        de = differential_evolution(
            objective, list(zip(lo, hi)), seed=seed + s, popsize=popsize, maxiter=maxiter,
            tol=1e-10, atol=0, polish=False, updating="immediate", init="sobol",
        )
        theta = de.x
        if refine:
            ls = least_squares(resid, theta, bounds=(lo, hi), x_scale=(hi - lo), xtol=1e-14, ftol=1e-14, gtol=1e-14)
            if objective(ls.x) <= de.fun:
                theta = ls.x
        starts.append((objective(theta), theta))
    loss, theta = min(starts, key=lambda s: s[0])
    return CalibrationResult(model, theta, loss, rmse_iv(model, theta, chain), bic(model, theta, chain), starts)
