"""Lo que faltaba del Monte Carlo de la sección 6 del marco teórico, con los parámetros calibrados de cada evento.

  - Valor del variance swap justo después del anuncio, ecuación (35), trayectoria por trayectoria: RV del primer paso
    (el que contiene el anuncio) más el strike forward vigente K_{t,T} sin el evento, menos el strike de mercado. En
    Bates K_{t,T} usa la varianza simulada de cada trayectoria. `corr2_value_event` es la parte de la variación de ese
    valor que explica R_e^2, como dice la aproximación (36).
  - El vol crush en la simulación: se repite con v_{tau+} = c v_{tau} (en Merton, sigma^2 -> c sigma^2 después del
    primer paso), con el c calibrado con la cadena posterior (`analysis.forward_c`).
  - Pago del comprador: media, desviación, asimetría y cuantiles 1 % y 5 %.
  - Variable de control para Bates: la integral de la varianza (trapecio diario), cuya esperanza se conoce por (15).
"""

from __future__ import annotations

import numpy as np
from scipy.stats import skew

from crush.analysis.compare import MODELS, theta_from_row
from crush.config import TRADING_DAYS_PER_YEAR
from crush.models.heston import HestonParams
from crush.simulation.bates_qe import simulate_bates_logreturns
from crush.simulation.jumps import JumpSchedule
from crush.simulation.merton_mc import simulate_merton_logreturns
from crush.simulation.rv import mean_and_stderr, realized_variance, swap_value

DT = 1.0 / TRADING_DAYS_PER_YEAR


def _params(row, tag: str) -> dict:
    return dict(zip(MODELS[tag].names, map(float, theta_from_row(row, tag))))


def _simulate(p: dict, tag: str, n: int, n_paths: int, seed: int, c: float = 1.0):
    """(log-retornos, varianza en cada nodo o None)."""
    sched = JumpSchedule.with_event(n, DT, 0, p["Lambda_e"], p["mu_e"], p["sigma_e"])
    rng = np.random.default_rng(seed)
    if tag == "merton":
        sigma = np.full(n, p["sigma"])
        sigma[1:] *= np.sqrt(c)
        return simulate_merton_logreturns(sigma, sched, n_paths, b=0.0, dt=DT, rng=rng), None
    hp = HestonParams(p["v0"], p["kappa"], p["theta"], p["xi"], p["rho"])
    return simulate_bates_logreturns(hp, sched, n_paths, b=0.0, dt=DT, rng=rng, return_variance=True, crush=(0, c))


def value_after_event(lr: np.ndarray, p: dict, tag: str, variance, k_var: float, c: float = 1.0) -> np.ndarray:
    """Ecuación (35) en t = un día (justo después del anuncio), una por trayectoria."""
    T = lr.shape[1] * DT
    tau = T - DT
    if tag == "merton":
        k_fwd = c * p["sigma"] ** 2
    else:  # la varianza simulada al cierre del primer paso ya incluye el reajuste c
        kt = p["kappa"] * tau
        k_fwd = p["theta"] + (variance[:, 1] - p["theta"]) * (1.0 - np.exp(-kt)) / kt
    return swap_value(lr[:, 0] ** 2 / DT, DT, T, k_fwd, k_var)


def control_variate(rv: np.ndarray, variance: np.ndarray, p: dict) -> tuple[float, float]:
    """(media de RV corregida, error estándar) usando la integral de v como variable de control."""
    n = variance.shape[1] - 1
    integral = DT * (variance[:, :-1] + variance[:, 1:]).sum(axis=1) / 2.0
    ev = p["theta"] + (p["v0"] - p["theta"]) * np.exp(-p["kappa"] * DT * np.arange(n + 1))
    expected = DT * (ev[:-1] + ev[1:]).sum() / 2.0
    cov = np.cov(rv, integral)
    beta = cov[0, 1] / cov[1, 1] if cov[1, 1] > 0 else 0.0
    return mean_and_stderr(rv - beta * (integral - expected))


def _tail(x: np.ndarray, prefix: str) -> dict:
    return {f"{prefix}_mean": float(x.mean()), f"{prefix}_sd": float(x.std()), f"{prefix}_skew": float(skew(x)),
            f"{prefix}_q01": float(np.quantile(x, 0.01)), f"{prefix}_q05": float(np.quantile(x, 0.05))}


def mc_extra(row, tag: str, i: int, c: float, n_paths: int = 100_000, seed: int = 0) -> dict:
    p = _params(row, tag)
    n = max(int(round(float(row[f"T_{i}"]) * TRADING_DAYS_PER_YEAR)), 2)
    k_mkt = float(row[f"k_mkt_{i}"])
    lr, var = _simulate(p, tag, n, n_paths, seed)
    rv = realized_variance(lr)
    value = value_after_event(lr, p, tag, var, k_mkt)
    out = {"ticker": row["ticker"], "announcement_date": row["announcement_date"], "model": tag, "maturity": i, "n_steps": n, "k_mkt": k_mkt, "c": c}
    out.update(_tail(rv - k_mkt, "pay"))
    out.update(_tail(value, "value"))
    out["corr2_value_event"] = float(np.corrcoef(value, lr[:, 0] ** 2)[0, 1] ** 2)
    out["k_hat"], out["se"] = mean_and_stderr(rv)
    if tag == "bates":
        out["k_hat_cv"], out["se_cv"] = control_variate(rv, var, p)
    if np.isfinite(c):
        lr_c, var_c = _simulate(p, tag, n, n_paths, seed, c)
        rv_c = realized_variance(lr_c)
        out["k_hat_c"] = float(rv_c.mean())
        out["pay_mean_c"] = float((rv_c - k_mkt).mean())
        out["value_mean_c"] = float(value_after_event(lr_c, p, tag, var_c, k_mkt, c).mean())
    return out
