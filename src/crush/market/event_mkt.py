"""Lo que el mercado dice del anuncio: varianza de evento, volatilidad base y razón de crush.

Con los dos vencimientos más cercanos antes del anuncio (que lo cruzan) se aplica (26). Hay dos formas de medir la
varianza total w(T):

  - "atm":    w = sigma_ATM^2 T                    (la del marco teórico)
  - "strike": w = K_repl_mkt T                     (model-free; incluye todo el smile)

Con la cadena del día posterior (vencimientos sin evento adentro) el mismo cálculo debe dar eps ~ 0: sirve de control.
La razón de crush observada (55) es sigma_ATM(post) / sigma_ATM(pre) del mismo vencimiento.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from crush.market.chain import CleanParams, Slice, build_slices
from crush.market.rates import event_rates
from crush.market.strike_mkt import k_repl_discrete, k_repl_smooth
from crush.models.event import EventVariance, split_event_variance


def total_variance(s: Slice, how: str = "atm") -> float:
    if how == "atm":
        return s.atm_iv**2 * s.T
    if how == "strike":
        return k_repl_smooth(s.K, s.iv, s.T, s.r, s.F, extrapolate=True) * s.T
    raise ValueError("how debe ser 'atm' o 'strike'")


def market_event(s1: Slice, s2: Slice, how: str = "atm") -> EventVariance | None:
    """(26) con dos vencimientos. None si los datos no alcanzan o w(T2) <= w(T1)."""
    w1, w2 = total_variance(s1, how), total_variance(s2, how)
    if not (np.isfinite(w1) and np.isfinite(w2)):
        return None
    try:
        return split_event_variance(np.sqrt(w1 / s1.T), s1.T, np.sqrt(w2 / s2.T), s2.T)
    except ValueError:
        return None


def slice_row(s: Slice) -> dict:
    return {
        "expiry": s.expiry, "T": s.T, "F": s.F, "n_pairs": s.n_pairs, "n_strikes": int(s.K.size),
        "logm_min": float(s.logm[0]), "logm_max": float(s.logm[-1]), "atm_iv": s.atm_iv,
        "k_repl_discrete": k_repl_discrete(s.K, s.iv, s.T, s.r, s.F),
        "k_repl_trunc": k_repl_smooth(s.K, s.iv, s.T, s.r, s.F, extrapolate=False),
        "k_repl_extrap": k_repl_smooth(s.K, s.iv, s.T, s.r, s.F, extrapolate=True),
    }


def slice_table(data, r: float, params: CleanParams | None = None) -> pd.DataFrame:
    """Una fila por (día, vencimiento) de un evento descargado (`EventData`)."""
    m = data.meta
    r_pre, r_post = event_rates(m, r)
    rows = []
    for label, df, day, spot, rd in (("pre", data.pre, m["pre_day"], m["spot_pre"], r_pre), ("post", data.post, m["post_day"], m["spot_post"], r_post)):
        for s in build_slices(df, day, spot, rd, params):
            rows.append({"ticker": m["ticker"], "announcement_date": m["announcement_date"], "when": label, **slice_row(s)})
    return pd.DataFrame(rows)


def event_row(data, r: float, params: CleanParams | None = None) -> dict:
    """Fila de resultados del mercado para un evento: eps, sigma_b y CR (26)/(34), CR observado (55), control post."""
    m = data.meta
    r_pre, r_post = event_rates(m, r)
    pre = build_slices(data.pre, m["pre_day"], m["spot_pre"], r_pre, params)
    post = build_slices(data.post, m["post_day"], m["spot_post"], r_post, params)
    row = {"ticker": m["ticker"], "announcement_date": m["announcement_date"], "timing": m["timing"], "n_slices_pre": len(pre), "n_slices_post": len(post)}
    if len(pre) >= 2:
        for how in ("atm", "strike"):
            ev = market_event(pre[0], pre[1], how)
            row[f"eps_{how}"] = ev.eps if ev else np.nan
            row[f"sigma_b_{how}"] = ev.sigma_b if ev else np.nan
            row[f"cr_split_{how}"] = ev.crush_ratio if ev else np.nan
        row["T_1"] = pre[0].T  # plazo del vencimiento más cercano (el del swap de referencia)
        row["atm_iv_pre"] = pre[0].atm_iv
        row["k_repl_pre"] = k_repl_smooth(pre[0].K, pre[0].iv, pre[0].T, r_pre, pre[0].F)
    if len(post) >= 2:
        ev = market_event(post[0], post[1], "atm")
        row["eps_post_control"] = ev.eps if ev else np.nan
        row["atm_iv_post"] = post[0].atm_iv
        row["k_repl_post"] = k_repl_smooth(post[0].K, post[0].iv, post[0].T, r_post, post[0].F)
    if len(pre) >= 1 and len(post) >= 1 and pre[0].expiry == post[0].expiry:
        row["cr_obs"] = post[0].atm_iv / pre[0].atm_iv  # ecuación (55)
    return row
