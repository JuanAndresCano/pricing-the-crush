"""Black-Scholes en forma de forward y solver de volatilidad implícita.

Implementa las ecuaciones (37) y (38) del marco teórico (docs/marco_teorico.md, sección 5):

    C(sigma) = e^{-rT} [ F N(d1) - K N(d2) ],   P(sigma) = e^{-rT} [ K N(-d2) - F N(-d1) ]
    d1 = (ln(F/K) + sigma^2 T / 2) / (sigma sqrt(T)),   d2 = d1 - sigma sqrt(T)

F es el forward, r la tasa libre de riesgo (para descontar) y T el plazo en años de
negociación (días hábiles / 252). La volatilidad implícita es la raíz de C(sigma) = precio.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import brentq
from scipy.special import ndtr

SIGMA_LO = 1e-6
SIGMA_HI = 10.0


def _d1_d2(F, K, T, sigma):
    F, K, T, sigma = np.broadcast_arrays(*(np.asarray(x, dtype=float) for x in (F, K, T, sigma)))
    sd = sigma * np.sqrt(T)
    with np.errstate(divide="ignore", invalid="ignore"):
        d1 = (np.log(F / K) + 0.5 * sd**2) / sd
    return d1, d1 - sd, sd


def bs_price(F, K, T, sigma, r, is_call=True):
    """Precio Black-Scholes. Admite arreglos. Con T o sigma iguales a 0 devuelve el valor intrínseco."""
    F, K, T, sigma, r = np.broadcast_arrays(*(np.asarray(x, dtype=float) for x in (F, K, T, sigma, r)))
    disc = np.exp(-r * T)
    sign = 1.0 if is_call else -1.0
    intrinsic = disc * np.maximum(sign * (F - K), 0.0)
    d1, d2, sd = _d1_d2(F, K, T, sigma)
    with np.errstate(invalid="ignore"):
        priced = disc * sign * (F * ndtr(sign * d1) - K * ndtr(sign * d2))
    return np.where(sd > 0, priced, intrinsic)


def bs_vega(F, K, T, sigma, r):
    """Derivada del precio respecto de sigma (igual para call y put)."""
    d1, _, sd = _d1_d2(F, K, T, sigma)
    F, T, r = np.broadcast_arrays(*(np.asarray(x, dtype=float) for x in (F, T, r)))
    pdf = np.exp(-0.5 * d1**2) / np.sqrt(2.0 * np.pi)
    with np.errstate(invalid="ignore"):
        v = np.exp(-r * T) * F * pdf * np.sqrt(T)
    return np.where(sd > 0, v, 0.0)


def price_bounds(F, K, T, r, is_call=True):
    """Cotas de no arbitraje del precio: (valor intrínseco descontado, techo)."""
    disc = np.exp(-r * T)
    if is_call:
        return disc * max(F - K, 0.0), disc * F
    return disc * max(K - F, 0.0), disc * K


def implied_vol(price, F, K, T, r, is_call=True, tol=1e-12):
    """Volatilidad implícita de una opción (escalar). Devuelve nan si no hay solución.

    No hay solución si el precio viola las cotas de no arbitraje o si T <= 0. La función es
    estrictamente creciente en sigma (vega > 0), así que la raíz es única y Brent la encuentra.
    """
    if not (T > 0 and F > 0 and K > 0 and np.isfinite(price)):
        return float("nan")
    lo, hi = price_bounds(F, K, T, r, is_call)
    if not (lo < price < hi):
        return float("nan")

    def f(s):
        return float(bs_price(F, K, T, s, r, is_call)) - price

    if f(SIGMA_LO) > 0 or f(SIGMA_HI) < 0:
        return float("nan")
    return brentq(f, SIGMA_LO, SIGMA_HI, xtol=tol, rtol=1e-14, maxiter=200)


def implied_vol_array(price, F, K, T, r, is_call):
    """Versión para arreglos: aplica `implied_vol` elemento a elemento."""
    price, F, K, T, r, is_call = np.broadcast_arrays(
        *(np.asarray(x) for x in (price, F, K, T, r, is_call))
    )
    out = np.empty(price.shape, dtype=float)
    for idx in np.ndindex(price.shape):
        out[idx] = implied_vol(
            float(price[idx]), float(F[idx]), float(K[idx]), float(T[idx]), float(r[idx]), bool(is_call[idx])
        )
    return out
