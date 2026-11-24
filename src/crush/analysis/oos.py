"""Validación fuera de muestra de la calibración (sección 7 del marco teórico, Pregunta 1).

La calibración principal usa todos los strikes del día previo. Aquí cada modelo se calibra solo con strikes alternos
de cada vencimiento y se evalúa con los strikes que no vio: RMSE de volatilidad implícita fuera de muestra y error
relativo del strike K_repl frente al del mercado (que sale de todas las cotizaciones y no depende del modelo).
Para Bates, θ se fija como en la calibración principal (varianza ATM del vencimiento más lejano, de todas las
cotizaciones); es el único dato que ve de los strikes de prueba y solo el nivel ATM de largo plazo.
"""

from __future__ import annotations

import numpy as np

from crush.calibration.calibrate import calibrate
from crush.calibration.real import chain_from_slices, theta_from_far
from crush.calibration.selection import rmse_iv
from crush.analysis.compare import split_chain
from crush.market.chain import build_slices
from crush.market.rates import event_rates
from crush.market.strike_mkt import k_repl_smooth
from crush.models.event_models import BatesEvent, MertonEvent


def oos_row(data, r: float, n_starts: int = 2, seed: int = 0) -> dict | None:
    m = data.meta
    slices = build_slices(data.pre, m["pre_day"], m["spot_pre"], event_rates(m, r)[0], None)
    if len(slices) < 2:
        return None
    chain = chain_from_slices(slices)
    train, test = split_chain(chain)
    bates = BatesEvent()
    th, _ = theta_from_far(slices)
    th = float(np.clip(th, bates.lower[2], bates.upper[2]))
    row = {"ticker": m["ticker"], "announcement_date": m["announcement_date"], "n_train": len(train), "n_test": len(test)}
    for tag, model, fixed in (("merton", MertonEvent(), None), ("bates", bates, {"theta": th})):
        res = calibrate(model, train, n_starts=n_starts, seed=seed, fixed=fixed)
        row[f"rmse_train_{tag}"] = res.rmse_iv
        row[f"rmse_test_{tag}"] = float(rmse_iv(model, res.theta, test))
        for i, s in enumerate(slices[:2], start=1):
            k_mod = model.k_repl(res.theta, s.T, True)
            k_mkt = k_repl_smooth(s.K, s.iv, s.T, s.r, s.F, extrapolate=True)
            row[f"relerr_{tag}_{i}"] = k_mod / k_mkt - 1.0
    return row
