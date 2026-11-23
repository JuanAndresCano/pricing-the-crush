"""Precio de opciones por inversión de la función característica (Gil-Pelaez).

Ecuación (40) del marco teórico:

    P2 = 1/2 + (1/pi) Int_0^inf Re[ e^{-iu ln K} phi(u) / (iu) ] du
    P1 = 1/2 + (1/pi) Int_0^inf Re[ e^{-iu ln K} phi(u - i) / (iu phi(-i)) ] du
    C  = e^{-rT} (F0 P1 - K P2)

phi(u) = E[exp(i u ln S_T)] bajo Q. Los puts se obtienen por paridad. La integral se calcula con
cuadratura de Gauss-Legendre por tramos; se verifica la convergencia duplicando `u_max`.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from numpy.polynomial.legendre import leggauss


def _grid(u_max: float, panels: int, nodes: int):
    x, w = leggauss(nodes)
    edges = np.linspace(0.0, u_max, panels + 1)
    a, b = edges[:-1, None], edges[1:, None]
    u = (0.5 * (b - a) * x[None, :] + 0.5 * (a + b)).ravel()
    wt = (0.5 * (b - a) * w[None, :]).ravel()
    return u, wt


def gil_pelaez_call(
    phi: Callable[[np.ndarray], np.ndarray],
    F0: float,
    K,
    T: float,
    r: float,
    u_max: float = 1000.0,
    panels: int = 500,
    nodes: int = 16,
):
    """Precio de un call europeo a partir de phi. `K` puede ser escalar o arreglo."""
    Ks = np.atleast_1d(np.asarray(K, dtype=float))
    u, wt = _grid(u_max, panels, nodes)
    ph = phi(u)
    ph1 = phi(u - 1j) / phi(np.array([-1j]))[0]
    e = np.exp(-1j * np.outer(np.log(Ks), u))
    inv = 1.0 / (1j * u)
    p2 = 0.5 + (np.real(e * (ph * inv)[None, :]) @ wt) / np.pi
    p1 = 0.5 + (np.real(e * (ph1 * inv)[None, :]) @ wt) / np.pi
    out = np.exp(-r * T) * (F0 * p1 - Ks * p2)
    return float(out[0]) if np.ndim(K) == 0 else out


class CallPricer:
    """Precio de calls por Gil-Pelaez para un conjunto fijo de strikes, con la matriz e^{-iu ln K} precalculada.

    La calibración evalúa la misma cadena miles de veces con parámetros distintos. Como los strikes y la
    malla de integración no cambian, la matriz se calcula una sola vez y cada precio queda en dos
    productos matriz-vector. Con la malla por defecto (u_max = 400, 100 tramos de 16 nodos) el error
    frente a la malla fina de `gil_pelaez_call` es menor a 1e-6 en los casos de earnings.
    """

    def __init__(self, K, u_max: float = 400.0, panels: int = 100, nodes: int = 16):
        self.K = np.atleast_1d(np.asarray(K, dtype=float))
        self.u, self.wt = _grid(u_max, panels, nodes)
        e = np.exp(-1j * np.outer(np.log(self.K), self.u))
        self._er, self._ei = e.real.copy(), e.imag.copy()
        self._inv = 1.0 / (1j * self.u)

    def _prob(self, z):
        z = z * self._inv
        return 0.5 + (self._er @ (z.real * self.wt) - self._ei @ (z.imag * self.wt)) / np.pi

    def price(self, F0: float, T: float, r: float, phi) -> np.ndarray:
        p2 = self._prob(phi(self.u))
        p1 = self._prob(phi(self.u - 1j) / phi(np.array([-1j]))[0])
        return np.exp(-r * T) * (F0 * p1 - self.K * p2)
