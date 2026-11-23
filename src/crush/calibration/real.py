"""Calibración de un evento real: de las cadenas del día previo a Merton-evento y Bates-evento.

Se usa la cadena del cierre previo (todos los vencimientos seleccionados cruzan el anuncio). Decisión 10: con
vencimientos de pocos días kappa y theta no se identifican. Si hay un vencimiento lejano (>= `far_min_days`
hábiles) se calibra con él y, además, theta se fija en su varianza ATM (el nivel al que revierte la varianza a
ese plazo); si no lo hay, también se fija theta con el vencimiento más largo disponible y se marca `theta_weak`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from crush.calibration.calibrate import CalibrationResult, calibrate
from crush.calibration.quotes import OptionChain
from crush.config import TRADING_DAYS_PER_YEAR
from crush.market.chain import CleanParams, Slice, build_slices
from crush.market.strike_mkt import k_repl_smooth
from crush.models.event_models import BatesEvent, EventModel, MertonEvent


def chain_from_slices(slices: list[Slice], crosses: bool = True) -> OptionChain:
    T, F, r, K, ic, px, cr = [], [], [], [], [], [], []
    for s in slices:
        n = s.K.size
        T += [s.T] * n; F += [s.F] * n; r += [s.r] * n; K += list(s.K); ic += list(s.is_call); px += list(s.price); cr += [crosses] * n
    return OptionChain(np.array(T), np.array(F), np.array(r), np.array(K), np.array(ic), np.array(px), np.array(cr))


def theta_from_far(slices: list[Slice], far_min_days: int = 21) -> tuple[float, bool]:
    """(theta, fuerte). theta = varianza ATM del vencimiento más largo; fuerte si ese plazo es >= far_min_days hábiles."""
    s = max(slices, key=lambda x: x.T)
    return float(s.atm_iv**2), bool(s.T * TRADING_DAYS_PER_YEAR >= far_min_days)


@dataclass
class EventCalibration:
    merton: CalibrationResult
    bates: CalibrationResult
    theta_fixed: float
    theta_strong: bool
    slices: list[Slice]


def calibrate_event(data, r: float, params: CleanParams | None = None, n_starts: int = 2, seed: int = 0, **kw) -> EventCalibration | None:
    m = data.meta
    slices = build_slices(data.pre, m["pre_day"], m["spot_pre"], r, params)
    if len(slices) < 2:
        return None
    chain = chain_from_slices(slices)
    th, strong = theta_from_far(slices)
    bates_model = BatesEvent()
    lo, hi = bates_model.lower[2], bates_model.upper[2]
    th = float(np.clip(th, lo, hi))
    merton = calibrate(MertonEvent(), chain, n_starts=n_starts, seed=seed, **kw)
    bates = calibrate(bates_model, chain, n_starts=n_starts, seed=seed, fixed={"theta": th}, **kw)
    return EventCalibration(merton, bates, th, strong, slices)


def calibration_row(data, cal: EventCalibration) -> dict:
    """Fila por evento: ajuste, parámetros, strikes de cada modelo y del mercado en los dos vencimientos más cercanos."""
    m = data.meta
    row = {"ticker": m["ticker"], "announcement_date": m["announcement_date"], "theta_fixed": cal.theta_fixed, "theta_strong": cal.theta_strong}
    for tag, res in (("merton", cal.merton), ("bates", cal.bates)):
        row[f"rmse_iv_{tag}"] = res.rmse_iv
        row[f"bic_{tag}"] = res.bic
        row[f"eps_repl_{tag}"] = res.model.eps_repl(res.theta)
        row[f"eps_qv_{tag}"] = res.model.eps_qv(res.theta)
        for k, v in res.params().items():
            row[f"{tag}_{k}"] = v
        for i, s in enumerate(cal.slices[:2], start=1):
            row[f"k_{tag}_{i}"] = res.model.k_repl(res.theta, s.T, True)
    for i, s in enumerate(cal.slices[:2], start=1):
        row[f"k_mkt_{i}"] = k_repl_smooth(s.K, s.iv, s.T, s.r, s.F, extrapolate=True)
        row[f"T_{i}"] = s.T
        for tag in ("merton", "bates"):
            row[f"relerr_{tag}_{i}"] = row[f"k_{tag}_{i}"] / row[f"k_mkt_{i}"] - 1.0
    return row
