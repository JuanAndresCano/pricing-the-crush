"""Simulación exacta de Merton por paso, ecuación (46) del marco teórico (sección 6).

    ln S_{t+D} = ln S_t + (b - sigma^2/2) D - Lambda k + sigma sqrt(D) Z1 + N mu + sigma_j sqrt(N) Z2,
    N ~ Poisson(Lambda)

b = r - q es el costo de acarreo, y Lambda, mu, sigma_j y k salen del calendario de saltos de cada paso.
No hay error de discretización: la suma de N saltos normales es una normal.
"""

from __future__ import annotations

import numpy as np

from crush.config import TRADING_DAYS_PER_YEAR
from crush.simulation.jumps import JumpSchedule, draw_jumps, normals


def simulate_merton_logreturns(
    sigma: float,
    schedule: JumpSchedule,
    n_paths: int,
    b: float = 0.0,
    dt: float = 1.0 / TRADING_DAYS_PER_YEAR,
    rng: np.random.Generator | None = None,
    antithetic: bool = True,
) -> np.ndarray:
    """Log-retornos por paso, matriz (n_paths, n). El compensador mantiene E[S_T] = S_0 e^{bT}."""
    rng = rng or np.random.default_rng()
    n = schedule.n
    z1 = normals(rng, n_paths, n, antithetic)
    z2 = normals(rng, n_paths, n, antithetic)
    drift = (b - 0.5 * sigma**2) * dt - schedule.lam_dt * schedule.k
    return drift[None, :] + sigma * np.sqrt(dt) * z1 + draw_jumps(rng, schedule, n_paths, z2)
