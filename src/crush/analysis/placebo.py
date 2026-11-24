"""Placebo de ventana para la Pregunta 2: ¿la "caída extra" después del día 0 es crush gradual o decaimiento normal?

La Pregunta 2 mide g = IV(día k) / IV(día 0) en los días siguientes al anuncio y encuentra que baja más de lo que
explican el mercado (SPY) y el modelo. Una objeción posible: la volatilidad implícita de cualquier opción de pocos
días tiende a bajar a medida que se acerca el vencimiento, con o sin anuncio. El placebo repite exactamente la misma
medición en una ventana tranquila de las mismas empresas, sin anuncio:

  - día 0 del placebo = `PLACEBO_OFFSET` días hábiles después del día 0 real (unas 4 semanas después del anuncio y
    unas 5 antes del siguiente), con los mismos días k = 1, 2, 4 y la misma regla de selección de vencimientos
    (dos cercanos y uno lejano);
  - mismo control (SPY) y mismo filtro de días restantes que en el análisis real.

El modelo de evento no se usa (no hay evento que modelar), así que la comparación justa es con el análisis real de
Merton, donde `g_model = 1`: e = ln g_obs - beta ln g_spy. Aquí:

  - `placebo_table`: una fila por evento, vencimiento y día k, con g_obs y g_spy.
  - `stacked_effect`: regresión conjunta ln g = a + b ln g_spy + c D_real, con errores robustos por fecha de anuncio.
    `c` es la caída extra atribuible al anuncio, por encima de la del placebo.
  - `paired_by_event`: promedio de e real menos promedio de e placebo, por evento; Wilcoxon sobre el promedio por fecha.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from crush.analysis.compare import clustered_wilcoxon
from crush.analysis.crush import _control_ratios
from crush.analysis.gradual import cluster_ols
from crush.config import TRADING_DAYS_PER_YEAR
from crush.market.chain import CleanParams, build_slices

PLACEBO_OFFSET = 20  # días hábiles entre el día 0 real y el día 0 del placebo
OFFSETS = (1, 2, 4)


def placebo_table(meta: dict, days: dict, control: dict | None, r_default: float, params: CleanParams | None = None,
                  min_days_left: int = 3) -> pd.DataFrame:
    """Filas (evento, vencimiento, k) del placebo.

    `meta`: ticker y fecha del anuncio. `days`: {k: (cadena, día, precio, tasa o None)} con k = 0, 1, 2, 4.
    `control`: lo que devuelve `analysis.crush.control_slices` para esos mismos días (o None).
    """
    if 0 not in days:
        return pd.DataFrame()
    by_day = {}
    for k, (df, day, spot, rd) in days.items():
        by_day[k] = (str(day), {s.expiry: s for s in build_slices(df, str(day), spot, r_default if rd is None else rd, params)})
    day0, slices0 = by_day[0]
    rows = []
    for k in sorted(by_day):
        if k == 0:
            continue
        dayk, slicesk = by_day[k]
        for exp, s in slicesk.items():
            left = round(s.T * TRADING_DAYS_PER_YEAR)
            if exp not in slices0 or left < min_days_left:
                continue
            s0 = slices0[exp]
            g_spy, _ = _control_ratios(control, day0, day0, dayk, s0.T)  # pre_day = day0: solo importa IV(k)/IV(0)
            rows.append({
                "ticker": meta["ticker"], "announcement_date": str(meta["announcement_date"]), "expiry": exp, "k": k,
                "days_left": left, "g_obs": s.atm_iv / s0.atm_iv, "g_spy": g_spy,
            })
    return pd.DataFrame(rows)


def _clean(t: pd.DataFrame) -> pd.DataFrame:
    t = t[np.isfinite(t["g_obs"]) & np.isfinite(t["g_spy"]) & (t["g_obs"] > 0) & (t["g_spy"] > 0)].copy()
    t["ln_g"], t["ln_spy"] = np.log(t["g_obs"]), np.log(t["g_spy"])
    t["announcement_date"] = t["announcement_date"].astype(str)
    return t


def real_table(path: pd.DataFrame) -> pd.DataFrame:
    """Filas reales comparables: q2_path.csv con el modelo Merton (g_model = 1), solo k > 0."""
    t = path[(path["model"] == "merton") & (path["k"] > 0)]
    return t[["ticker", "announcement_date", "expiry", "k", "days_left", "g_obs", "g_spy"]].reset_index(drop=True)


def cluster_multi(X: np.ndarray, y: np.ndarray, clusters) -> dict:
    """MCO con varias columnas (la primera es la constante) y errores robustos por conglomerado (CR1, G - 1 grados)."""
    X, y = np.asarray(X, float), np.asarray(y, float)
    codes, uniq = pd.factorize(pd.Series(clusters))
    G, (n, p) = len(uniq), X.shape
    if G < 3 or n <= p:
        return {"coef": np.full(p, np.nan), "se": np.full(p, np.nan), "p": np.full(p, np.nan), "n": n, "n_clusters": G}
    XtX_inv = np.linalg.inv(X.T @ X)
    coef = XtX_inv @ X.T @ y
    resid = y - X @ coef
    meat = np.zeros((p, p))
    for g in range(G):
        s = X[codes == g].T @ resid[codes == g]
        meat += np.outer(s, s)
    V = XtX_inv @ meat @ XtX_inv * (G / (G - 1)) * ((n - 1) / (n - p))
    se = np.sqrt(np.diag(V))
    return {"coef": coef, "se": se, "p": 2 * stats.t.sf(np.abs(coef / se), df=G - 1), "n": int(n), "n_clusters": int(G)}


def stacked_effect(real: pd.DataFrame, placebo: pd.DataFrame, controls_days_left: bool = False) -> dict:
    """ln g = a + b ln g_spy + c D_real (+ d ln días_restantes). `c` es la caída extra del anuncio sobre el placebo."""
    r, p = _clean(real), _clean(placebo)
    r["real"], p["real"] = 1.0, 0.0
    t = pd.concat([r, p], ignore_index=True)
    cols = [np.ones(len(t)), t["ln_spy"].to_numpy(), t["real"].to_numpy()]
    names = ["alpha_placebo", "beta", "c_real_minus_placebo"]
    if controls_days_left:
        cols.append(np.log(t["days_left"].to_numpy(float)))
        names.append("ln_days_left")
    res = cluster_multi(np.column_stack(cols), t["ln_g"].to_numpy(), t["announcement_date"])
    out = {"n": res["n"], "n_clusters": res["n_clusters"]}
    for i, nm in enumerate(names):
        out[nm], out[f"se_{nm}"], out[f"p_{nm}"] = float(res["coef"][i]), float(res["se"][i]), float(res["p"][i])
    return out


def group_report(t: pd.DataFrame, label: str) -> list[dict]:
    """Caída extra (alpha de ln g = alpha + beta ln g_spy) para todos los horizontes y para cada k, con Wilcoxon por fecha."""
    t = _clean(t)
    reg = cluster_ols(t["ln_spy"], t["ln_g"], t["announcement_date"])
    beta = reg["beta"] if np.isfinite(reg["beta"]) else 0.0
    t["e"] = t["ln_g"] - beta * t["ln_spy"]
    rows = []
    for hz, sub in [("all", t)] + [(int(k), t[t["k"] == k]) for k in sorted(t["k"].unique())]:
        w = clustered_wilcoxon(sub["e"], sub["announcement_date"])
        rows.append({"group": label, "horizon": hz, "n_obs": int(len(sub)), "n_dates": w["n_dates"], "beta": beta,
                     "mean_e": w["mean_d"], "median_e": w["median_d"], "share_negative": float((sub["e"] < 0).mean()),
                     "mean_days_left": float(sub["days_left"].mean()), "pvalue_wilcoxon": w["pvalue"],
                     **({"alpha_reg": reg["alpha"], "p_alpha_reg": reg["p_alpha"]} if hz == "all" else {})})
    return rows


def paired_by_event(real: pd.DataFrame, placebo: pd.DataFrame, beta: float) -> dict:
    """Por evento: promedio de e real menos promedio de e placebo (e = ln g - beta ln g_spy). Wilcoxon por fecha."""
    r, p = _clean(real), _clean(placebo)
    for t in (r, p):
        t["e"] = t["ln_g"] - beta * t["ln_spy"]
    key = ["ticker", "announcement_date"]
    m = r.groupby(key)["e"].mean().rename("e_real").to_frame().join(p.groupby(key)["e"].mean().rename("e_placebo"), how="inner").reset_index()
    d = m["e_real"] - m["e_placebo"]
    w = clustered_wilcoxon(d, m["announcement_date"])
    return {"n_events": w["n_events"], "n_dates": w["n_dates"], "mean_real": float(m["e_real"].mean()),
            "mean_placebo": float(m["e_placebo"].mean()), "mean_diff": w["mean_d"], "median_diff": w["median_d"],
            "share_real_lower": float((d < 0).mean()), "pvalue_wilcoxon": w["pvalue"]}
