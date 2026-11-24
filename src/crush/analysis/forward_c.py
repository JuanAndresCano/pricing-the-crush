"""Varianza forward (escalón del anuncio) y parámetro c de reajuste de la varianza (marco teórico, secciones 4 y 7).

Varianza forward. Con los strikes de dos vencimientos, K_{T1,T2} = (T2 K_{T2} - T1 K_{T1}) / (T2 - T1). Los dos vencimientos
descargados cruzan el anuncio, así que la teoría (33), con T1 < tau, no se puede observar tal cual: no hay un vencimiento
anterior al anuncio. Lo que sí se observa es el mismo escalón visto de otro modo. Antes del anuncio el primer tramo [0, T1]
contiene el evento y el tramo [T1, T2] no:

    escalon_pre = K(T1) - K_{T1,T2}  (con el evento en el primer tramo: ~ 2 Lambda_e (k_e - mu_e) / T1)

Después del anuncio el evento ya pasó y los dos tramos solo tienen varianza base, así que `escalon_post` debe ser ~ 0.
Se mide en el mercado y se compara con lo que dan Merton-evento y Bates-evento.

Parámetro c. Después del anuncio la varianza instantánea se reajusta, v_{tau+} = c E[v_tau] (en Merton, sigma_post^2 = c
sigma^2). Con c = 1 el crush viene solo de que el evento desaparece; c < 1, de que además bajó la varianza base; c > 1,
de que bajó menos de lo que dice el modelo. Se calibra, por vencimiento y día k, el c que iguala la volatilidad ATM del
modelo (sin evento) a la observada; `c_adj` usa la observada descontando el mercado (SPY, beta).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import brentq

from crush.analysis.compare import MODELS, theta_from_row
from crush.analysis.crush import _control_ratios, evolve_state, model_atm_iv
from crush.config import TRADING_DAYS_PER_YEAR
from crush.market.chain import CleanParams, build_slices
from crush.market.rates import event_rates
from crush.market.strike_mkt import k_repl_smooth
from crush.models.event_models import BatesEvent


def forward_var(T1: float, K1: float, T2: float, K2: float) -> float:
    return (T2 * K2 - T1 * K1) / (T2 - T1)


def _k_mkt(s) -> float:
    return k_repl_smooth(s.K, s.iv, s.T, s.r, s.F, extrapolate=True)


def forward_step_row(data, cal_row, r: float, params: CleanParams | None = None) -> dict | None:
    """Escalón de la varianza forward antes y después del anuncio, en el mercado y en cada modelo."""
    m = data.meta
    r_pre, r_post = event_rates(m, r)
    pre = build_slices(data.pre, m["pre_day"], m["spot_pre"], r_pre, params)[:2]
    if len(pre) < 2:
        return None
    post_all = {s.expiry: s for s in build_slices(data.post, m["post_day"], m["spot_post"], r_post, params)}
    if any(s.expiry not in post_all for s in pre):
        return None
    post = [post_all[s.expiry] for s in pre]
    (a, b), (ap, bp) = pre, post
    out = {"ticker": m["ticker"], "announcement_date": m["announcement_date"], "T1": a.T, "T2": b.T}
    k1, k2, k1p, k2p = _k_mkt(a), _k_mkt(b), _k_mkt(ap), _k_mkt(bp)
    f_pre, f_post = forward_var(a.T, k1, b.T, k2), forward_var(ap.T, k1p, bp.T, k2p)
    out.update(mkt_k1_pre=k1, mkt_k2_pre=k2, mkt_fwd_pre=f_pre, mkt_step_pre=k1 - f_pre,
               mkt_k1_post=k1p, mkt_k2_post=k2p, mkt_fwd_post=f_post, mkt_step_post=k1p - f_post)
    for tag, model in MODELS.items():
        theta = theta_from_row(cal_row, tag)
        m1, m2 = model.k_repl(theta, a.T, True), model.k_repl(theta, b.T, True)
        th_post = evolve_state(model, theta, a.T - ap.T)
        m1p, m2p = model.k_repl(th_post, ap.T, False), model.k_repl(th_post, bp.T, False)
        fp, fq = forward_var(a.T, m1, b.T, m2), forward_var(ap.T, m1p, bp.T, m2p)
        out.update({f"{tag}_k1_pre": m1, f"{tag}_k2_pre": m2, f"{tag}_fwd_pre": fp, f"{tag}_step_pre": m1 - fp,
                    f"{tag}_k1_post": m1p, f"{tag}_k2_post": m2p, f"{tag}_fwd_post": fq, f"{tag}_step_post": m1p - fq})
    return out


def theta_with_c(model, theta, dt: float, c: float) -> np.ndarray:
    """Parámetros después de `dt` años con la varianza instantánea reajustada por `c` (Bates: v0; Merton: sigma^2)."""
    th = evolve_state(model, theta, dt)
    th[0] = th[0] * (c if isinstance(model, BatesEvent) else np.sqrt(c))
    return th


def solve_c(model, theta, slice_post, dt: float, iv_target: float, lo: float = 0.05, hi: float = 5.0) -> float:
    """c tal que la volatilidad ATM del modelo sin evento iguala a `iv_target`. nan si no hay solución en [lo, hi]."""
    if not np.isfinite(iv_target):
        return float("nan")
    s = slice_post

    def f(c):
        return model_atm_iv(model, theta_with_c(model, theta, dt, c), s.F, s.T, s.r, False) - iv_target

    try:
        return float(brentq(f, lo, hi))
    except (ValueError, RuntimeError):
        return float("nan")


def c_table(data, extra: dict, cal_row, r: float, params: CleanParams | None = None, min_days_left: int = 3,
            control: dict | None = None, control_beta: float = 0.0) -> pd.DataFrame:
    """Una fila por vencimiento, día k (0 = cierre posterior) y modelo, con `c_raw` y `c_adj` (descontando el mercado)."""
    m = data.meta
    r_pre, r_post = event_rates(m, r)
    pre = {s.expiry: s for s in build_slices(data.pre, m["pre_day"], m["spot_pre"], r_pre, params)}
    days = {0: (data.post, m["post_day"], m["spot_post"], r_post)}
    for k, (df, day, spot, rd) in extra.items():
        days[k] = (df, day, spot, r_post if rd is None else rd)
    rows = []
    day0 = str(m["post_day"])
    for k, (df, day, spot, rd) in sorted(days.items()):
        for exp, s in {s.expiry: s for s in build_slices(df, day, spot, rd, params)}.items():
            left = round(s.T * TRADING_DAYS_PER_YEAR)
            if exp not in pre or left < min_days_left:
                continue
            dt = pre[exp].T - s.T
            _, spy_vs_pre = _control_ratios(control, str(m["pre_day"]), day0, str(day), pre[exp].T)
            adj = s.atm_iv / spy_vs_pre**control_beta if np.isfinite(spy_vs_pre) and spy_vs_pre > 0 else float("nan")
            for tag, model in MODELS.items():
                theta = theta_from_row(cal_row, tag)
                rows.append({"ticker": m["ticker"], "announcement_date": m["announcement_date"], "expiry": exp, "k": k,
                             "days_left": left, "model": tag, "c_raw": solve_c(model, theta, s, dt, s.atm_iv),
                             "c_adj": solve_c(model, theta, s, dt, adj)})
    return pd.DataFrame(rows)
