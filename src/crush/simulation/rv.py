"""Varianza realizada y valor del swap sobre trayectorias simuladas.

Ecuaciones (50), (51) y (35) del marco teórico:

    RV = (252 / n) Sum_i (ln S_{t_i} / S_{t_{i-1}})^2        retornos diarios, sin restar la media
    K_hat = (1/M) Sum_m RV^(m),   EE = s / sqrt(M)
    V_t = e^{-r(T-t)} N [ (t/T) RV_{0,t} + ((T-t)/T) K_{t,T} - K_var ]
"""

from __future__ import annotations

import numpy as np

from crush.config import TRADING_DAYS_PER_YEAR


def realized_variance(logret: np.ndarray, trading_days: int = TRADING_DAYS_PER_YEAR) -> np.ndarray:
    """RV anualizada por trayectoria, ecuación (50). `logret` tiene forma (n_paths, n)."""
    n = logret.shape[1]
    return trading_days / n * np.sum(logret**2, axis=1)


def mean_and_stderr(x: np.ndarray) -> tuple[float, float]:
    """Media muestral y error estándar, ecuación (51)."""
    x = np.asarray(x, dtype=float)
    return float(x.mean()), float(x.std(ddof=1) / np.sqrt(x.size))


def event_share(logret: np.ndarray, event_step: int) -> np.ndarray:
    """Participación del día del anuncio en la suma de cuadrados: R_e^2 / Sum r_i^2, por trayectoria."""
    sq = logret**2
    return sq[:, event_step] / sq.sum(axis=1)


def rv_ex_event(logret: np.ndarray, event_step: int, trading_days: int = TRADING_DAYS_PER_YEAR) -> np.ndarray:
    """RV sin el día del anuncio, normalizada por los n - 1 pasos restantes."""
    keep = np.delete(logret, event_step, axis=1)
    return realized_variance(keep, trading_days)


def swap_value(rv_realized, t: float, T: float, k_forward, k_var: float, notional: float = 1.0, r: float = 0.0):
    """Valor de un variance swap en t, ecuación (35). `rv_realized` es RV_{0,t} y `k_forward` el strike K_{t,T}."""
    return np.exp(-r * (T - t)) * notional * ((t / T) * rv_realized + ((T - t) / T) * k_forward - k_var)
