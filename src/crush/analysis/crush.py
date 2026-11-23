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


# --- trayectoria en los días posteriores (instantáneo contra gradual) -------------------------------------------


def control_slices(loaded: dict, r_default: float, params: CleanParams | None = None) -> dict:
    """Convierte lo que devuelve `data.control.load_control` en `{día: {vencimiento: Slice}}` para `path_table`."""
    return {
        day: {s.expiry: s for s in build_slices(df, day, spot, r_default if r is None else r, params)}
        for day, (df, spot, r) in loaded.items()
    }


def _control_ratios(control, pre_day: str, day0: str, day_k: str, T0: float) -> tuple[float, float]:
    """(IV_ctrl(k)/IV_ctrl(0), IV_ctrl(k)/IV_ctrl(previo)) sobre el vencimiento del control más cercano en plazo al del
    día 0. nan si falta el control. Se usa el mismo vencimiento de control en los tres días."""
    nan = (float("nan"), float("nan"))
    if not control or any(d not in control for d in (pre_day, day0, day_k)):
        return nan
    slices0 = control[day0]
    if not slices0:
        return nan
    exp = min(slices0, key=lambda e: abs(slices0[e].T - T0))
    try:
        iv0, ivk, ivp = (control[d][exp].atm_iv for d in (day0, day_k, pre_day))
    except KeyError:
        return nan
    return ivk / iv0, ivk / ivp


def path_table(
    data, extra: dict, cal_row, r: float, params: CleanParams | None = None, min_days_left: int = 3, control: dict | None = None,
    control_beta: float = 1.0,
) -> pd.DataFrame:
    """Una fila por vencimiento, día posterior y modelo.

    `extra` viene de `data.extra_days.load_extra_days`. El día 0 es `post`; los demás son los cierres 1, 2, 4 días
    hábiles después. Para cada vencimiento que sigue vivo (al menos `min_days_left` días hábiles) compara:

      cr_obs, cr_model   volatilidad ATM del día k / la del día previo al anuncio, observada y del modelo
      g_obs, g_model     lo que cambia respecto al día 0: IV(k) / IV(0). Si el crush fuera instantáneo, g ~ 1; si sigue
                         bajando, g_obs < 1. `g_gap = g_obs - g_model` mide cuánta caída extra hay sobre lo que el
                         modelo ya espera (en Bates, por la reversión de la varianza hacia theta).

    Con `control` (`{día: {vencimiento: Slice}}` de un activo como SPY) se descuenta el movimiento general del
    mercado: `g_spy = IV_spy(k) / IV_spy(0)` sobre el vencimiento de SPY más parecido, `g_obs_adj = g_obs / g_spy^beta`,
    `g_gap_adj = g_obs_adj - g_model` y `cr_obs_adj = cr_obs / (IV_spy(k) / IV_spy(previo))^beta`. Con beta = 1 (por
    defecto) se supone que la acción reacciona igual que el mercado; en datos reales suele reaccionar menos, así que
    conviene pasar el beta que da `estimate_control_beta`. Sin control, o si falta algún día, esas columnas quedan en nan.
    """
    m = data.meta
    r_pre, r_post = event_rates(m, r)
    pre = {s.expiry: s for s in build_slices(data.pre, m["pre_day"], m["spot_pre"], r_pre, params)}
    days = {0: (data.post, m["post_day"], m["spot_post"], r_post)}
    for k, (df, day, spot, rd) in extra.items():
        days[k] = (df, day, spot, r_post if rd is None else rd)
    by_day = {k: {s.expiry: s for s in build_slices(df, day, spot, rd, params)} for k, (df, day, spot, rd) in days.items()}
    thetas = {tag: theta_from_row(cal_row, tag) for tag in MODELS}
    day_of = {0: str(m["post_day"]), **{k: str(day) for k, (_, day, _, _) in extra.items()}}
    rows = []
    for k in sorted(by_day):
        for exp, s in by_day[k].items():
            left = round(s.T * TRADING_DAYS_PER_YEAR)
            if exp not in pre or left < min_days_left or exp not in by_day[0]:
                continue
            s0 = by_day[0][exp]
            g_spy, spy_vs_pre = _control_ratios(control, str(m["pre_day"]), day_of[0], day_of[k], s0.T)
            for tag, model in MODELS.items():
                res = model_crush(model, thetas[tag], pre[exp], s)
                res0 = model_crush(model, thetas[tag], pre[exp], s0)
                g_obs = s.atm_iv / s0.atm_iv
                g_model = res["iv_post_model"] / res0["iv_post_model"]
                rows.append({
                    "ticker": m["ticker"], "announcement_date": m["announcement_date"], "expiry": exp, "k": k, "days_left": left, "model": tag,
                    "cr_obs": s.atm_iv / pre[exp].atm_iv, "cr_model": res["cr_model"], "g_obs": g_obs, "g_model": g_model,
                    "g_gap": g_obs - g_model, "iv_obs": s.atm_iv, "iv_model": res["iv_post_model"],
                    "g_spy": g_spy, "g_obs_adj": g_obs / g_spy**control_beta, "g_gap_adj": g_obs / g_spy**control_beta - g_model,
                    "cr_obs_adj": s.atm_iv / pre[exp].atm_iv / spy_vs_pre**control_beta,
                })
    return pd.DataFrame(rows)


def summarize_path(table: pd.DataFrame) -> pd.DataFrame:
    """Promedios por modelo y día posterior: crush observado y del modelo, y la caída extra sobre el día 0."""
    g = table.groupby(["model", "k"])
    return g.agg(
        n=("g_obs", "size"), cr_obs=("cr_obs", "mean"), cr_model=("cr_model", "mean"),
        g_obs=("g_obs", "mean"), g_model=("g_model", "mean"), g_gap=("g_gap", "mean"),
        g_spy=("g_spy", "mean"), g_obs_adj=("g_obs_adj", "mean"), g_gap_adj=("g_gap_adj", "mean"),
    ).reset_index()


def estimate_control_beta(table: pd.DataFrame, model: str = "merton") -> dict:
    """Sensibilidad de la volatilidad de la acción a la del mercado, de los días posteriores al día 0.

    Regresión por mínimos cuadrados `ln g_obs = alpha + beta * ln g_spy` sobre las filas con k > 0 (una sola vez por
    fila: se usa un solo modelo, que da las mismas g_obs). `beta` es la sensibilidad. `alpha` es la caída (o subida)
    promedio que queda después de descontar el mercado: negativa significa que el crush sigue después del día 0.
    Necesita varias observaciones; con las de un solo evento solo sirve para ver el orden de magnitud.
    """
    t = table[(table["model"] == model) & (table["k"] > 0)]
    t = t[np.isfinite(t["g_spy"]) & np.isfinite(t["g_obs"]) & (t["g_spy"] > 0) & (t["g_obs"] > 0)]
    if len(t) < 3:
        return {"beta": float("nan"), "alpha": float("nan"), "n": int(len(t))}
    x, y = np.log(t["g_spy"].to_numpy(float)), np.log(t["g_obs"].to_numpy(float))
    beta, alpha = np.polyfit(x, y, 1)
    return {"beta": float(beta), "alpha": float(alpha), "n": int(len(t))}
