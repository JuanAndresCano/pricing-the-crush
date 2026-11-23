"""Modelo de Bates (1996): Heston más saltos log-normales de Merton.

Implementa la sección 3 del marco teórico (docs/marco_teorico.md):

  - dinámica bajo Q (16) y (17), con 8 parámetros: v0, kappa, theta, xi, rho, lam, mu_j, sigma_j
  - función característica (21): producto de la parte de Heston y la de saltos
  - strikes (22) y (23), y el sesgo de replicación (24)

Los precios se calculan solo por Fourier; no hay serie como en Merton.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import exp

import numpy as np

from crush.models.fourier import gil_pelaez_call
from crush.models.heston import HestonParams, heston_cf, v_bar


@dataclass(frozen=True)
class BatesParams:
    v0: float
    kappa: float
    theta: float
    xi: float
    rho: float
    lam: float = 0.0
    mu_j: float = 0.0
    sigma_j: float = 0.0

    @property
    def heston(self) -> HestonParams:
        return HestonParams(self.v0, self.kappa, self.theta, self.xi, self.rho)

    @property
    def k(self) -> float:
        """k = E[e^J - 1]."""
        return exp(self.mu_j + 0.5 * self.sigma_j**2) - 1.0


def bates_cf(u, T: float, F0: float, p: BatesParams):
    """Función característica de ln S_T, ecuación (21). Con lam = 0 es la de Heston."""
    u = np.asarray(u, dtype=complex)
    jumps = p.lam * T * (np.exp(1j * u * p.mu_j - 0.5 * p.sigma_j**2 * u**2) - 1.0 - 1j * u * p.k)
    return heston_cf(u, T, F0, p.heston) * np.exp(jumps)


def bates_call(F0: float, K, T: float, r: float, p: BatesParams, **kw):
    """Call europeo por inversión de Fourier (Gil-Pelaez)."""
    return gil_pelaez_call(lambda u: bates_cf(u, T, F0, p), F0, K, T, r, **kw)


def bates_put(F0: float, K, T: float, r: float, p: BatesParams, **kw):
    """Put por paridad put-call."""
    c = bates_call(F0, K, T, r, p, **kw)
    return c - np.exp(-r * T) * (F0 - np.asarray(K, dtype=float))


def k_qv(T: float, p: BatesParams) -> float:
    """Lo que paga el contrato, ecuación (22): vbar(T) + lam (mu_j^2 + sigma_j^2)."""
    return v_bar(T, p.heston) + p.lam * (p.mu_j**2 + p.sigma_j**2)


def k_repl(T: float, p: BatesParams) -> float:
    """Lo que arma la tira de opciones, ecuación (23): vbar(T) + 2 lam (k - mu_j)."""
    return v_bar(T, p.heston) + 2.0 * p.lam * (p.k - p.mu_j)


def jump_share(T: float, p: BatesParams) -> float:
    """Participación de los saltos en K_repl."""
    return 2.0 * p.lam * (p.k - p.mu_j) / k_repl(T, p)
