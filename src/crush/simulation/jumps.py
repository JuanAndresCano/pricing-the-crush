"""Calendario de saltos por paso de la simulación.

Cada paso diario i lleva tres números: el número esperado de saltos `lam_dt[i]` (lambda por Delta, o
Lambda_e el día del anuncio), la media `mu[i]` y la desviación `sigma[i]` del log-salto. Así el salto de
earnings se concentra en un solo paso (marco teórico, sección 6, "Concentrar el salto en el día del anuncio").

El paso i es el retorno del cierre i al cierre i+1. Si el anuncio es antes de la apertura del día tau, el
salto cae en el paso que termina en el cierre de tau. Si es después del cierre de tau, cae en el paso siguiente.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class JumpSchedule:
    lam_dt: np.ndarray  # número esperado de saltos en cada paso
    mu: np.ndarray  # media del log-salto en cada paso
    sigma: np.ndarray  # desviación del log-salto en cada paso

    def __post_init__(self):
        if not (self.lam_dt.shape == self.mu.shape == self.sigma.shape and self.lam_dt.ndim == 1):
            raise ValueError("lam_dt, mu y sigma deben ser vectores del mismo largo.")
        if np.any(self.lam_dt < 0) or np.any(self.sigma < 0):
            raise ValueError("lam_dt y sigma deben ser no negativos.")

    @property
    def n(self) -> int:
        return int(self.lam_dt.size)

    @property
    def k(self) -> np.ndarray:
        """Compensador por paso: k_i = E[e^J - 1] = exp(mu + sigma^2 / 2) - 1."""
        return np.exp(self.mu + 0.5 * self.sigma**2) - 1.0

    @classmethod
    def constant(cls, n: int, dt: float, lam: float, mu: float, sigma: float) -> "JumpSchedule":
        """Saltos de Merton repartidos de forma uniforme en los n pasos."""
        return cls(np.full(n, lam * dt), np.full(n, mu), np.full(n, sigma))

    @classmethod
    def with_event(
        cls,
        n: int,
        dt: float,
        event_step: int,
        Lambda_e: float,
        mu_e: float,
        sigma_e: float,
        lam_bg: float = 0.0,
        mu_bg: float = 0.0,
        sigma_bg: float = 0.0,
    ) -> "JumpSchedule":
        """Saltos de fondo (lam_bg, normalmente 0) y un evento concentrado en el paso `event_step`."""
        if not 0 <= event_step < n:
            raise ValueError("event_step debe estar entre 0 y n - 1.")
        lam_dt = np.full(n, lam_bg * dt)
        mu = np.full(n, mu_bg)
        sigma = np.full(n, sigma_bg)
        lam_dt[event_step], mu[event_step], sigma[event_step] = Lambda_e, mu_e, sigma_e
        return cls(lam_dt, mu, sigma)


def draw_jumps(rng: np.random.Generator, schedule: JumpSchedule, n_paths: int, z: np.ndarray) -> np.ndarray:
    """Suma de saltos por paso y trayectoria. `z` son normales estándar de forma (n_paths, n)."""
    counts = rng.poisson(schedule.lam_dt[None, :], size=(n_paths, schedule.n))
    return counts * schedule.mu[None, :] + schedule.sigma[None, :] * np.sqrt(counts) * z


def normals(rng: np.random.Generator, n_paths: int, n: int, antithetic: bool) -> np.ndarray:
    """Normales estándar (n_paths, n). Con antitéticas, la segunda mitad es el espejo de la primera."""
    if not antithetic:
        return rng.standard_normal((n_paths, n))
    if n_paths % 2:
        raise ValueError("Con variables antitéticas n_paths debe ser par.")
    half = rng.standard_normal((n_paths // 2, n))
    return np.vstack([half, -half])
