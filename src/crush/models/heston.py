"""Modelo de Heston (1993): varianza estocástica con reversión a la media.

Implementa la sección 3 del marco teórico (docs/marco_teorico.md):

    dv = kappa (theta - v) dt + xi sqrt(v) dW^v,   d<W^S, W^v> = rho dt              (14)
    E[v_t] = theta + (v0 - theta) e^{-kappa t}
    vbar(T) = (1/T) E Int_0^T v dt = theta + (v0 - theta)(1 - e^{-kappa T}) / (kappa T)   (15)

La función característica usa la versión estable de Albrecher et al. (2007), "Little Heston Trap",
ecuaciones (19) y (20). Se escribe con el forward F0, así que el término de tasa (r - q) ya va dentro de F0.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

XI_ZERO = 1e-10  # por debajo de este valor la varianza es determinista


@dataclass(frozen=True)
class HestonParams:
    v0: float  # varianza de hoy
    kappa: float  # velocidad de reversión
    theta: float  # varianza de largo plazo
    xi: float  # volatilidad de la varianza
    rho: float  # correlación precio-varianza

    @property
    def feller(self) -> float:
        """Condición de Feller: 2 kappa theta - xi^2 >= 0 mantiene v > 0."""
        return 2.0 * self.kappa * self.theta - self.xi**2


def v_bar(T: float, p: HestonParams) -> float:
    """Varianza esperada promedio sobre [0, T], ecuación (15)."""
    if p.kappa * T < 1e-12:
        return p.v0
    return p.theta + (p.v0 - p.theta) * (1.0 - np.exp(-p.kappa * T)) / (p.kappa * T)


def heston_cf(u, T: float, F0: float, p: HestonParams):
    """phi(u) = E[exp(i u ln S_T)] bajo Q, ecuaciones (19) a (21) sin saltos. Acepta u complejo."""
    u = np.asarray(u, dtype=complex)
    if p.xi < XI_ZERO:
        total = v_bar(T, p) * T  # varianza integrada determinista
        return np.exp(1j * u * (np.log(F0) - 0.5 * total) - 0.5 * u**2 * total)
    xi2 = p.xi**2
    b = p.kappa - p.rho * p.xi * 1j * u
    d = np.sqrt(b**2 + xi2 * (1j * u + u**2))
    g = (b - d) / (b + d)
    e = np.exp(-d * T)
    D = (b - d) / xi2 * (1.0 - e) / (1.0 - g * e)
    C = p.kappa * p.theta / xi2 * ((b - d) * T - 2.0 * np.log((1.0 - g * e) / (1.0 - g)))
    return np.exp(1j * u * np.log(F0) + C + D * p.v0)
