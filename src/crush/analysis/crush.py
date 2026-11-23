"""Pregunta 2: ¿cómo reacciona la estructura temporal al vol crush?

Para cada vencimiento que cruza el anuncio compara la razón de crush observada (55),

    CR_obs = sigma_ATM(post) / sigma_ATM(pre),

con la que implica el modelo calibrado el día previo. Con los parámetros del día previo el modelo da el paso a
paso, y se separa en dos pedazos:

  1. Salida del evento (instantáneo): el factor de evento deja de estar en el futuro (crosses True -> False), con
     la misma varianza instantánea.
  2. Reversión de la varianza (gradual): en Bates la varianza pasa de v0 a E[v_dt] = theta + (v0 - theta) e^{-kappa dt}
     en el día que transcurre. En Merton no hay pedazo gradual.

    CR_modelo = CR_evento * CR_reversión

Si CR_obs es mucho menor que CR_modelo, el mercado quitó más volatilidad de la que el modelo dice que el evento
contenía (o revisó la varianza base). Con un solo día posterior no se puede ver si la caída continúa los días
siguientes; para eso hay que descargar más días (ver docs/decisiones.md).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from crush.analysis.compare import MODELS, theta_from_row
from crush.config import TRADING_DAYS_PER_YEAR
from crush.market.chain import CleanParams, build_slices
from crush.market.rates import event_rates
from crush.models.black_scholes import implied_vol
from crush.models.event_models import BatesEvent, EventModel
from crush.models.fourier import CallPricer


def evolve_state(model: EventModel, theta, dt: float) -> np.ndarray:
    """Parámetros después de `dt` años: en Bates, v0 pasa a su esperanza; en Merton no cambia nada."""
    out = np.array(theta, dtype=float)
    if isinstance(model, BatesEvent):
        v0, kappa, th = out[0], out[1], out[2]
        out[0] = th + (v0 - th) * np.exp(-kappa * dt)
    return out


def model_atm_iv(model: EventModel, theta, F: float, T: float, r: float, crosses: bool) -> float:
    c = model.call_prices(theta, F, T, r, crosses, CallPricer(np.array([F])))[0]
    return float(implied_vol(c, F, F, T, r, True))


def model_crush(model: EventModel, theta, pre, post) -> dict:
    """CR del modelo para un vencimiento, dados el `Slice` previo y el posterior (mismo vencimiento)."""
    dt = (pre.T - post.T)
    iv_pre = model_atm_iv(model, theta, pre.F, pre.T, pre.r, True)
    iv_noevent = model_atm_iv(model, theta, post.F, post.T, post.r, False)
    iv_post = model_atm_iv(model, evolve_state(model, theta, dt), post.F, post.T, post.r, False)
    return {
        "iv_pre_model": iv_pre, "iv_post_model": iv_post,
        "cr_event": iv_noevent / iv_pre, "cr_revert": iv_post / iv_noevent, "cr_model": iv_post / iv_pre,
    }


def crush_table(data, cal_row, r: float, params: CleanParams | None = None) -> pd.DataFrame:
    """Una fila por vencimiento y modelo: CR observado, CR del modelo y sus dos pedazos."""
    m = data.meta
    r_pre, r_post = event_rates(m, r)
    pre = {s.expiry: s for s in build_slices(data.pre, m["pre_day"], m["spot_pre"], r_pre, params)}
    post = {s.expiry: s for s in build_slices(data.post, m["post_day"], m["spot_post"], r_post, params)}
    rows = []
    for exp in sorted(set(pre) & set(post)):
        cr_obs = post[exp].atm_iv / pre[exp].atm_iv
        for tag, model in MODELS.items():
            res = model_crush(model, theta_from_row(cal_row, tag), pre[exp], post[exp])
            rows.append({
                "ticker": m["ticker"], "announcement_date": m["announcement_date"], "expiry": exp,
                "days": round(pre[exp].T * TRADING_DAYS_PER_YEAR), "model": tag, "cr_obs": cr_obs,
                "iv_pre_obs": pre[exp].atm_iv, "iv_post_obs": post[exp].atm_iv, **res, "gap": cr_obs - res["cr_model"],
            })
    return pd.DataFrame(rows)


def summarize_crush(table: pd.DataFrame) -> pd.DataFrame:
    """Por modelo y plazo: CR observado medio, CR del modelo medio, error medio y cuánto aporta la reversión."""
    g = table.groupby(["model", "days"])
    return g.agg(
        n=("cr_obs", "size"), cr_obs=("cr_obs", "mean"), cr_model=("cr_model", "mean"), cr_event=("cr_event", "mean"),
        cr_revert=("cr_revert", "mean"), gap=("gap", "mean"), mae=("gap", lambda x: float(np.mean(np.abs(x)))),
    ).reset_index()
