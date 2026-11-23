"""Strike de un variance swap a partir de precios de opciones OTM.

Ecuación (4) del marco teórico, en forma numérica sobre una malla de strikes:

    K_repl = (2/T) e^{rT} [ Int_0^F P(K)/K^2 dK + Int_F^inf C(K)/K^2 dK ]

Tres versiones:

  - `k_repl_from_otm_prices`: trapecio sobre la malla dada (control de código con malla muy fina).
  - `k_repl_discrete`: la suma (44), la del VIX, sobre los strikes que cotizan.
  - `k_repl_smooth`: interpola la volatilidad implícita en la moneyness logarítmica, la convierte a precios sobre
    una malla fina con Black-Scholes e integra. Con `extrapolate=True` extiende la volatilidad de forma plana
    hasta |ln(K/F)| = `half_width`; con `False` integra solo el rango observado. Comparar ambas mide cuánto
    pesa la cola que el mercado no cotiza (truncamiento, Jiang y Tian).
"""

from __future__ import annotations

import numpy as np
from scipy.interpolate import PchipInterpolator

from crush.models.black_scholes import bs_price


def k_repl_from_otm_prices(K, otm_prices, T: float, r: float) -> float:
    """Integra Q(K)/K^2 con el trapecio. `otm_prices[i]` es el put si K[i] < F0 y el call si K[i] >= F0."""
    K = np.asarray(K, dtype=float)
    q = np.asarray(otm_prices, dtype=float)
    return float((2.0 / T) * np.exp(r * T) * np.trapezoid(q / K**2, K))


def k_repl_discrete(K, iv, T: float, r: float, F: float) -> float:
    """Ecuación (44). `K` ordenados y `iv` su volatilidad implícita.

    Q(K_i) es el put si K_i < K0, el call si K_i > K0 y el promedio de ambos en K0 (el mayor strike <= F).
    Los dos precios en K0 salen de la misma volatilidad implícita, que es consistente con la paridad con F.
    """
    K = np.asarray(K, dtype=float)
    iv = np.asarray(iv, dtype=float)
    if K.size < 3 or K.min() > F or K.max() < F:
        return float("nan")
    i0 = int(np.flatnonzero(K <= F)[-1])
    put = bs_price(F, K, T, iv, r, False)
    call = bs_price(F, K, T, iv, r, True)
    q = np.where(np.arange(K.size) < i0, put, np.where(np.arange(K.size) > i0, call, 0.5 * (put + call)))
    dk = np.empty_like(K)
    dk[1:-1] = (K[2:] - K[:-2]) / 2.0
    dk[0] = K[1] - K[0]
    dk[-1] = K[-1] - K[-2]
    return float((2.0 / T) * np.exp(r * T) * np.sum(dk / K**2 * q) - (F / K[i0] - 1.0) ** 2 / T)


def k_repl_smooth(K, iv, T: float, r: float, F: float, extrapolate: bool = True, half_width: float = 1.0, n: int = 4001) -> float:
    """Strike de replicación con el smile interpolado (PCHIP, monótono) y, opcionalmente, extendido de forma plana."""
    K = np.asarray(K, dtype=float)
    iv = np.asarray(iv, dtype=float)
    if K.size < 3:
        return float("nan")
    x = np.log(K / F)
    spline = PchipInterpolator(x, iv)
    lo, hi = (min(-half_width, x[0]), max(half_width, x[-1])) if extrapolate else (x[0], x[-1])
    grid = np.linspace(lo, hi, n)
    vol = spline(np.clip(grid, x[0], x[-1]))
    Kg = F * np.exp(grid)
    q = np.where(grid < 0, bs_price(F, Kg, T, vol, r, False), bs_price(F, Kg, T, vol, r, True))
    # Int Q/K^2 dK = Int Q/K dx
    return float((2.0 / T) * np.exp(r * T) * np.trapezoid(q / Kg, grid))
