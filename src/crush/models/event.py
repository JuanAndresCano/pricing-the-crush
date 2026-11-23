"""Varianza de evento a partir de dos vencimientos que cruzan el anuncio.

Implementa las ecuaciones (25), (26) y (34) del marco teórico (sección 4):

    w(T)  = sigma_imp^2 T                     varianza total implícita
    sigma_b^2 = (w(T2) - w(T1)) / (T2 - T1)   varianza base
    eps       = w(T1) - sigma_b^2 T1          varianza del evento
    CR        = sigma_b / sigma_imp(T1)       razón de crush (aproximada)
"""

from __future__ import annotations

from dataclasses import dataclass
from math import exp, sqrt

import numpy as np


@dataclass(frozen=True)
class EventVariance:
    sigma_b: float  # volatilidad base anualizada
    eps: float  # varianza del evento (total, no anualizada)
    crush_ratio: float  # sigma_b / sigma_imp(T1)

    @property
    def implied_move(self) -> float:
        """Movimiento implícito del anuncio, sqrt(eps)."""
        return sqrt(self.eps)


def split_event_variance(vol1: float, T1: float, vol2: float, T2: float) -> EventVariance:
    """Separa la volatilidad base y la varianza del evento con dos vencimientos T1 < T2 que cruzan el anuncio.

    Lanza ValueError si T2 <= T1 o si la varianza base resulta no positiva.
    """
    if not 0 < T1 < T2:
        raise ValueError("Se requiere 0 < T1 < T2.")
    w1 = vol1**2 * T1
    w2 = vol2**2 * T2
    sb2 = (w2 - w1) / (T2 - T1)
    if sb2 <= 0:
        raise ValueError("La varianza base no es positiva: w(T2) <= w(T1).")
    eps = w1 - sb2 * T1
    return EventVariance(sigma_b=sqrt(sb2), eps=eps, crush_ratio=sqrt(sb2) / vol1)


# ---------------------------------------------------------------------------
# Factor de evento: salto de earnings en una fecha conocida (sección 4, ecuaciones (29) a (31))
# ---------------------------------------------------------------------------


def event_k(mu_e: float, sigma_e: float) -> float:
    """k_e = E[e^{J_EA} - 1], el compensador del salto del evento."""
    return exp(mu_e + 0.5 * sigma_e**2) - 1.0


def event_cf(u, Lambda_e: float, mu_e: float, sigma_e: float):
    """Función característica del factor de evento, ecuación (30), con su compensador.

    phi_evento(u) = exp{ Lambda_e ( e^{i u mu_e - sigma_e^2 u^2 / 2} - 1 - i u k_e ) }.
    El producto con la función característica del modelo base conserva phi(-i) = F0.
    """
    u = np.asarray(u, dtype=complex)
    k = event_k(mu_e, sigma_e)
    return np.exp(Lambda_e * (np.exp(1j * u * mu_e - 0.5 * sigma_e**2 * u**2) - 1.0 - 1j * u * k))


def eps_qv(Lambda_e: float, mu_e: float, sigma_e: float) -> float:
    """Varianza de evento que paga el contrato, ecuación (31): Lambda_e (mu_e^2 + sigma_e^2)."""
    return Lambda_e * (mu_e**2 + sigma_e**2)


def eps_repl(Lambda_e: float, mu_e: float, sigma_e: float) -> float:
    """Varianza de evento que arma la tira de opciones, ecuación (31): 2 Lambda_e (k_e - mu_e)."""
    return 2.0 * Lambda_e * (event_k(mu_e, sigma_e) - mu_e)
