"""Monte Carlo sobre los parámetros calibrados de cada evento (sección 6 del marco teórico).

Para cada evento y modelo se simulan trayectorias diarias hasta cada uno de los dos vencimientos con el salto de
evento concentrado en el primer paso (el retorno del cierre previo al cierre posterior, el que contiene el anuncio) y
se calcula:

  - K_hat = promedio de la varianza realizada (50) y su error estándar (51), contra la fórmula cerrada K_QV;
  - control de martingala (52): E[S_T] e^{-bT} ≈ 1;
  - el pago del comprador de un variance swap con strike de mercado, RV - K_mkt: media, desviación y cuantiles;
  - participación del evento η = R_e² / Σ R_i² y RV sin el día del anuncio.

La fórmula cerrada es de tiempo continuo y la simulación es diaria, así que se compara K_hat con K_QV dentro de dos
errores estándar más la diferencia de discretización, que se reporta aparte (`rel_diff`).
"""

from __future__ import annotations

import numpy as np

from crush.analysis.compare import MODELS, theta_from_row
from crush.config import TRADING_DAYS_PER_YEAR
from crush.models.heston import HestonParams
from crush.simulation.bates_qe import simulate_bates_logreturns
from crush.simulation.jumps import JumpSchedule
from crush.simulation.merton_mc import simulate_merton_logreturns
from crush.simulation.rv import event_share, mean_and_stderr, realized_variance, rv_ex_event


def simulate_event(row, tag: str, T: float, n_paths: int = 100_000, seed: int = 0, b: float = 0.0) -> np.ndarray:
    """Log-retornos diarios (n_paths, n) hasta el vencimiento a T años, con el salto de evento en el paso 0."""
    n = max(int(round(T * TRADING_DAYS_PER_YEAR)), 2)
    dt = 1.0 / TRADING_DAYS_PER_YEAR
    theta = theta_from_row(row, tag)
    names = MODELS[tag].names
    p = dict(zip(names, map(float, theta)))
    sched = JumpSchedule.with_event(n, dt, 0, p["Lambda_e"], p["mu_e"], p["sigma_e"])
    rng = np.random.default_rng(seed)
    if tag == "merton":
        return simulate_merton_logreturns(p["sigma"], sched, n_paths, b=b, dt=dt, rng=rng)
    hp = HestonParams(p["v0"], p["kappa"], p["theta"], p["xi"], p["rho"])
    return simulate_bates_logreturns(hp, sched, n_paths, b=b, dt=dt, rng=rng)


def mc_summary(row, tag: str, i: int, n_paths: int = 100_000, seed: int = 0, b: float = 0.0) -> dict:
    T = float(row[f"T_{i}"])
    lr = simulate_event(row, tag, T, n_paths, seed, b)
    n = lr.shape[1]
    rv = realized_variance(lr)
    k_hat, ee = mean_and_stderr(rv)
    theta = theta_from_row(row, tag)
    k_qv = MODELS[tag].k_qv(theta, n / TRADING_DAYS_PER_YEAR, True)  # mismo horizonte que la malla simulada
    k_mkt = float(row[f"k_mkt_{i}"])
    pay = rv - k_mkt
    martingale = float(np.mean(np.exp(lr.sum(axis=1) - b * n / TRADING_DAYS_PER_YEAR)))
    return {
        "ticker": row["ticker"], "announcement_date": row["announcement_date"], "model": tag, "maturity": i, "T": n / TRADING_DAYS_PER_YEAR,
        "k_hat": k_hat, "se": ee, "k_qv": k_qv, "z": (k_hat - k_qv) / ee, "rel_diff": k_hat / k_qv - 1.0, "k_mkt": k_mkt,
        "k_repl": float(row[f"k_{tag}_{i}"]), "martingale": martingale,
        "pay_mean": float(pay.mean()), "pay_sd": float(pay.std()), "pay_q01": float(np.quantile(pay, 0.01)), "pay_q05": float(np.quantile(pay, 0.05)),
        "pay_q95": float(np.quantile(pay, 0.95)), "prob_loss": float((pay < 0).mean()),
        "event_share": float(event_share(lr, 0).mean()), "rv_ex_event": float(rv_ex_event(lr, 0).mean()),
    }
