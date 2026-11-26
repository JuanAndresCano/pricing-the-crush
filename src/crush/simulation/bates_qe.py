"""Simulación de Bates con el esquema QE de Andersen (2008), ecuaciones (47) y (48) del marco teórico.

La varianza se simula con el esquema cuadrático-exponencial, que respeta la positividad y reproduce los
dos primeros momentos condicionales. El log-precio usa la descomposición del ruido con rho (K0 a K4,
con gamma1 = gamma2 = 1/2) más los saltos de Merton del calendario de cada paso.
"""

from __future__ import annotations

import numpy as np

from crush.config import TRADING_DAYS_PER_YEAR
from crush.models.heston import HestonParams
from crush.simulation.jumps import JumpSchedule, draw_jumps, normals

PSI_C = 1.5  # umbral entre la rama cuadrática y la exponencial


def qe_variance_step(v, kappa, theta, xi, dt, zv, u):
    """Un paso de varianza QE, ecuación (47). `zv` ~ N(0,1) y `u` ~ U(0,1), uno por trayectoria."""
    e = np.exp(-kappa * dt)
    m = theta + (v - theta) * e
    s2 = v * xi**2 * e * (1.0 - e) / kappa + theta * xi**2 * (1.0 - e) ** 2 / (2.0 * kappa)
    psi = s2 / m**2
    out = np.empty_like(v)

    low = psi <= PSI_C
    inv = 2.0 / np.maximum(psi[low], 1e-12)
    b2 = inv - 1.0 + np.sqrt(inv) * np.sqrt(inv - 1.0)
    a = m[low] / (1.0 + b2)
    out[low] = a * (np.sqrt(b2) + zv[low]) ** 2

    hi = ~low
    p = (psi[hi] - 1.0) / (psi[hi] + 1.0)
    beta = (1.0 - p) / m[hi]
    uu = u[hi]
    out[hi] = np.where(uu <= p, 0.0, np.log((1.0 - p) / (1.0 - np.minimum(uu, 1.0 - 1e-15))) / beta)
    return out


def simulate_bates_logreturns(
    heston: HestonParams,
    schedule: JumpSchedule,
    n_paths: int,
    b: float = 0.0,
    dt: float = 1.0 / TRADING_DAYS_PER_YEAR,
    rng: np.random.Generator | None = None,
    antithetic: bool = True,
    return_variance: bool = False,
    crush: tuple[int, float] | None = None,
):
    """Log-retornos por paso (n_paths, n). Con `return_variance` entrega también v en cada nodo (n_paths, n+1).

    `crush = (paso, c)` multiplica la varianza por `c` al final de ese paso (marco teórico, sección 6, "el vol crush en
    la simulación"): el retorno de ese paso usa la varianza sin reajustar y los siguientes, la reajustada.
    """
    rng = rng or np.random.default_rng()
    p = heston
    n = schedule.n
    rxi = p.rho / p.xi if p.xi > 0 else 0.0
    k0 = -rxi * p.kappa * p.theta * dt
    k1 = 0.5 * dt * (p.kappa * rxi - 0.5) - rxi
    k2 = 0.5 * dt * (p.kappa * rxi - 0.5) + rxi
    k34 = 0.5 * dt * (1.0 - p.rho**2)

    z = normals(rng, n_paths, n, antithetic)  # ruido del precio
    zv = normals(rng, n_paths, n, antithetic)  # ruido de la varianza
    zj = normals(rng, n_paths, n, antithetic)  # tamaño de los saltos
    uq = rng.random((n_paths, n))
    if antithetic:
        uq[n_paths // 2 :] = 1.0 - uq[: n_paths // 2]
    jumps = draw_jumps(rng, schedule, n_paths, zj)
    comp = schedule.lam_dt * schedule.k

    v = np.full(n_paths, p.v0)
    vs = [v]
    out = np.empty((n_paths, n))
    for i in range(n):
        vn = qe_variance_step(v, p.kappa, p.theta, p.xi, dt, zv[:, i], uq[:, i])
        out[:, i] = (
            b * dt - comp[i] + k0 + k1 * v + k2 * vn + np.sqrt(np.maximum(k34 * (v + vn), 0.0)) * z[:, i] + jumps[:, i]
        )
        if crush is not None and i == crush[0]:
            vn = crush[1] * vn
        v = vn
        vs.append(v)
    return (out, np.column_stack(vs)) if return_variance else out
