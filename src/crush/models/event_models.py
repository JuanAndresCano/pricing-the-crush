"""Modelos con evento: Merton y Bates con un salto de earnings de intensidad fija Lambda_e.

Implementan la propuesta del documento de correcciones (Corrección 1): sin saltos genéricos (lambda_c = 0),
el salto del anuncio aporta una bolsa de varianza fija, sin importar el plazo mientras T >= tau:

    phi(u; T) = phi_base(u; T) * phi_evento(u)      si el vencimiento cruza el anuncio
    phi(u; T) = phi_base(u; T)                      si no lo cruza

    base de Merton-evento: Black-Scholes con volatilidad sigma     (4 parámetros: sigma, Lambda_e, mu_e, sigma_e)
    base de Bates-evento:  Heston                                  (8 parámetros: v0, kappa, theta, xi, rho + evento)

Strikes (ecuación (32) con lambda_c = 0, anualizados):
    K_QV(T)   = base(T) + Lambda_e (mu_e^2 + sigma_e^2) / T
    K_repl(T) = base(T) + 2 Lambda_e (k_e - mu_e) / T
con base = sigma^2 en Merton-evento y base = vbar(T) en Bates-evento.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from crush.models.event import eps_qv, eps_repl, event_cf
from crush.models.fourier import CallPricer
from crush.models.heston import HestonParams, heston_cf, v_bar

_EVENT_BOUNDS = {"Lambda_e": (0.2, 6.0), "mu_e": (-0.15, 0.15), "sigma_e": (0.005, 0.30)}


class EventModel:
    """Interfaz común: parámetros, cotas, precios de calls y strikes."""

    name: str
    names: tuple[str, ...]
    bounds: tuple[tuple[float, float], ...]

    @property
    def n_params(self) -> int:
        return len(self.names)

    @property
    def lower(self) -> np.ndarray:
        return np.array([b[0] for b in self.bounds])

    @property
    def upper(self) -> np.ndarray:
        return np.array([b[1] for b in self.bounds])

    def as_dict(self, theta) -> dict[str, float]:
        return dict(zip(self.names, map(float, theta)))

    def _event(self, theta) -> tuple[float, float, float]:
        return float(theta[-3]), float(theta[-2]), float(theta[-1])

    # --- strikes -----------------------------------------------------------------------------
    def base(self, theta, T: float) -> float:
        raise NotImplementedError

    def k_qv(self, theta, T: float, crosses: bool = True) -> float:
        extra = eps_qv(*self._event(theta)) / T if crosses else 0.0
        return self.base(theta, T) + extra

    def k_repl(self, theta, T: float, crosses: bool = True) -> float:
        extra = eps_repl(*self._event(theta)) / T if crosses else 0.0
        return self.base(theta, T) + extra

    def eps_repl(self, theta) -> float:
        return eps_repl(*self._event(theta))

    def eps_qv(self, theta) -> float:
        return eps_qv(*self._event(theta))

    # --- precios -----------------------------------------------------------------------------
    def call_prices(self, theta, F: float, T: float, r: float, crosses: bool, pricer: CallPricer) -> np.ndarray:
        raise NotImplementedError


@dataclass
class MertonEvent(EventModel):
    name: str = "Merton-evento"
    names: tuple[str, ...] = ("sigma", "Lambda_e", "mu_e", "sigma_e")
    bounds: tuple[tuple[float, float], ...] = ((0.05, 1.5), _EVENT_BOUNDS["Lambda_e"], _EVENT_BOUNDS["mu_e"], _EVENT_BOUNDS["sigma_e"])

    def base(self, theta, T):
        return float(theta[0]) ** 2

    def call_prices(self, theta, F, T, r, crosses, pricer):
        sigma = float(theta[0])
        Lam, mu, sg = self._event(theta)

        def phi(u):
            u = np.asarray(u, dtype=complex)
            base = np.exp(1j * u * (np.log(F) - 0.5 * sigma**2 * T) - 0.5 * sigma**2 * u**2 * T)
            return base * event_cf(u, Lam, mu, sg) if crosses else base

        return pricer.price(F, T, r, phi)


@dataclass
class BatesEvent(EventModel):
    name: str = "Bates-evento"
    names: tuple[str, ...] = ("v0", "kappa", "theta", "xi", "rho", "Lambda_e", "mu_e", "sigma_e")
    bounds: tuple[tuple[float, float], ...] = (
        (0.0025, 2.0),
        (0.1, 30.0),
        (0.0025, 2.0),
        (0.05, 3.0),
        (-0.99, 0.99),
        _EVENT_BOUNDS["Lambda_e"],
        _EVENT_BOUNDS["mu_e"],
        _EVENT_BOUNDS["sigma_e"],
    )

    @staticmethod
    def heston(theta) -> HestonParams:
        return HestonParams(*(float(x) for x in theta[:5]))

    def base(self, theta, T):
        return v_bar(T, self.heston(theta))

    def call_prices(self, theta, F, T, r, crosses, pricer):
        hp = self.heston(theta)
        Lam, mu, sg = self._event(theta)

        def phi(u):
            base = heston_cf(u, T, F, hp)
            return base * event_cf(u, Lam, mu, sg) if crosses else base

        return pricer.price(F, T, r, phi)
