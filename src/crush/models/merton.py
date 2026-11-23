"""Modelo de Merton (1976): difusión con saltos log-normales.

Implementa la sección 2 del marco teórico (docs/marco_teorico.md):

  - dinámica bajo Q:   d ln S = (r - lam k - sigma^2/2) dt + sigma dW + J dN,   J ~ N(mu_j, sigma_j^2)
  - función característica (8)
  - serie de Merton (9) para opciones europeas
  - strikes de un variance swap: K_QV (11) y K_repl (13)

Se usa el forward F0 = S0 e^{(r-q)T} como dato, así que los dividendos no se estiman por separado.
El plazo T está en años de negociación (días hábiles / 252).
"""

from __future__ import annotations

from dataclasses import dataclass
from math import exp, log1p

import numpy as np
from scipy.stats import poisson

from crush.models.black_scholes import bs_price


@dataclass(frozen=True)
class MertonParams:
    sigma: float  # volatilidad de la difusión (anual)
    lam: float  # intensidad de saltos (por año)
    mu_j: float  # media del log-salto
    sigma_j: float  # desviación del log-salto

    @property
    def k(self) -> float:
        """k = E[e^J - 1], el compensador de los saltos."""
        return exp(self.mu_j + 0.5 * self.sigma_j**2) - 1.0


def merton_cf(u, T: float, F0: float, p: MertonParams):
    """Función característica de ln S_T bajo Q, ecuación (8), escrita con el forward.

    phi(u) = E[exp(i u ln S_T)]. Acepta u complejo, y phi(-i) = F0 (condición de martingala).
    """
    u = np.asarray(u, dtype=complex)
    diffusion = 1j * u * (np.log(F0) - p.lam * p.k * T - 0.5 * p.sigma**2 * T) - 0.5 * p.sigma**2 * u**2 * T
    jumps = p.lam * T * (np.exp(1j * u * p.mu_j - 0.5 * p.sigma_j**2 * u**2) - 1.0)
    return np.exp(diffusion + jumps)


def merton_call(F0: float, K, T: float, r: float, p: MertonParams, tail: float = 1e-15):
    """Precio de un call europeo con la serie de Merton, ecuación (9).

    Condicionado a n saltos, ln S_T es normal: media del forward F_n = F0 e^{-lam k T + n ln(1+k)}
    y varianza sigma^2 T + n sigma_j^2. El precio es el promedio de Black-Scholes con pesos de Poisson(lam T).
    """
    Ks = np.atleast_1d(np.asarray(K, dtype=float))
    lamT = p.lam * T
    nmax = int(poisson.isf(tail, lamT)) + 1 if lamT > 0 else 0
    n = np.arange(nmax + 1)
    w = poisson.pmf(n, lamT)
    Fn = F0 * np.exp(-p.lam * p.k * T + n * log1p(p.k))
    sig_n = np.sqrt(p.sigma**2 + n * p.sigma_j**2 / T)
    prices = bs_price(Fn[None, :], Ks[:, None], T, sig_n[None, :], r, True)
    out = prices @ w
    return float(out[0]) if np.ndim(K) == 0 else out


def merton_put(F0: float, K, T: float, r: float, p: MertonParams, tail: float = 1e-15):
    """Put por paridad put-call: P = C - e^{-rT} (F0 - K)."""
    c = merton_call(F0, K, T, r, p, tail)
    return c - exp(-r * T) * (F0 - np.asarray(K, dtype=float)) if np.ndim(K) else c - exp(-r * T) * (F0 - K)


def k_qv(p: MertonParams) -> float:
    """Lo que paga el contrato (varianza cuadrática esperada, anualizada), ecuación (11)."""
    return p.sigma**2 + p.lam * (p.mu_j**2 + p.sigma_j**2)


def k_repl(p: MertonParams) -> float:
    """Lo que arma la tira de opciones (anualizado), ecuación (13)."""
    return p.sigma**2 + 2.0 * p.lam * (p.k - p.mu_j)


def jump_share(p: MertonParams) -> float:
    """Participación de los saltos en K_repl: 2 lam (k - mu_j) / K_repl."""
    return 2.0 * p.lam * (p.k - p.mu_j) / k_repl(p)
