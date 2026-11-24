"""Contraste final de la Pregunta 2: ¿el crush sigue después del día 0, descontado el mercado y el modelo?

Entrada: la tabla de `analysis.crush.path_table` (una fila por evento, vencimiento, día posterior k y modelo), con las
columnas `g_obs` (volatilidad ATM del día k sobre la del día 0), `g_spy` (lo mismo para el control) y `g_model`.

Para cada modelo se define el residuo en logaritmos

    e = ln g_obs - beta * ln g_spy - ln g_model

Si el crush fuera instantáneo y el modelo lo describiera bien, e ~ 0. e < 0 significa que la volatilidad bajó más de
lo que explican el mercado y el modelo (crush gradual); e > 0, que bajó menos. Dos contrastes, ambos agrupados por
fecha de anuncio porque los eventos de un mismo día comparten el mercado:

  1. Regresión `ln g_obs - ln g_model = alpha + beta ln g_spy` con errores robustos por conglomerado (fecha). `alpha`
     es la caída extra promedio, ya sin el movimiento del mercado.
  2. Wilcoxon de signos y rangos sobre el promedio de e por fecha, con el beta de la regresión (por eso es algo
     optimista: el beta sale de los mismos datos). Se reporta para todos los horizontes y por separado para k = 1, 2, 4.

Los dos necesitan varias fechas: con menos de 3 (regresión) o de 6 (Wilcoxon) no reportan p-valor.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from crush.analysis.compare import clustered_wilcoxon


def cluster_ols(x, y, clusters) -> dict:
    """Regresión `y = alpha + beta x` con errores estándar robustos por conglomerado (CR1) y t de G - 1 grados."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    nan = {"alpha": np.nan, "beta": np.nan, "se_alpha": np.nan, "se_beta": np.nan, "p_alpha": np.nan, "p_beta": np.nan,
           "n": int(len(y)), "n_clusters": 0}
    codes, uniq = pd.factorize(pd.Series(clusters))
    G = len(uniq)
    nan["n_clusters"] = int(G)
    X = np.column_stack([np.ones_like(x), x])
    n, p = X.shape
    if G < 3 or n <= p or np.ptp(x) == 0:
        return nan
    XtX_inv = np.linalg.inv(X.T @ X)
    coef = XtX_inv @ X.T @ y
    resid = y - X @ coef
    meat = np.zeros((p, p))
    for g in range(G):
        s = X[codes == g].T @ resid[codes == g]
        meat += np.outer(s, s)
    V = XtX_inv @ meat @ XtX_inv * (G / (G - 1)) * ((n - 1) / (n - p))
    se = np.sqrt(np.diag(V))
    pv = 2 * stats.t.sf(np.abs(coef / se), df=G - 1)
    return {"alpha": float(coef[0]), "beta": float(coef[1]), "se_alpha": float(se[0]), "se_beta": float(se[1]),
            "p_alpha": float(pv[0]), "p_beta": float(pv[1]), "n": int(n), "n_clusters": int(G)}


def _prepare(table: pd.DataFrame, model: str) -> pd.DataFrame:
    t = table[(table["model"] == model) & (table["k"] > 0)].copy()
    ok = np.isfinite(t["g_obs"]) & np.isfinite(t["g_spy"]) & np.isfinite(t["g_model"]) & (t["g_obs"] > 0) & (t["g_spy"] > 0) & (t["g_model"] > 0)
    t = t[ok]
    t["ln_g"] = np.log(t["g_obs"])
    t["ln_spy"] = np.log(t["g_spy"])
    t["ln_model"] = np.log(t["g_model"])
    t["announcement_date"] = t["announcement_date"].astype(str)
    return t


def gradual_report(table: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(regresión por modelo, contrastes de Wilcoxon por modelo y horizonte). Ver el docstring del módulo."""
    reg_rows, test_rows = [], []
    for model in sorted(table["model"].unique()):
        t = _prepare(table, model)
        reg = cluster_ols(t["ln_spy"], t["ln_g"] - t["ln_model"], t["announcement_date"])
        reg_rows.append({"model": model, **reg})
        beta = reg["beta"]
        if not np.isfinite(beta):
            beta = 0.0  # sin beta estimable: se contrasta sin descontar el mercado
        t["e"] = t["ln_g"] - beta * t["ln_spy"] - t["ln_model"]
        for label, sub in [("all", t)] + [(int(k), t[t["k"] == k]) for k in sorted(t["k"].unique())]:
            w = clustered_wilcoxon(sub["e"], sub["announcement_date"])
            test_rows.append({
                "model": model, "horizon": label, "beta_used": beta, "n_obs": w["n_events"], "n_dates": w["n_dates"],
                "mean_e": w["mean_d"], "median_e": w["median_d"], "share_negative": float((sub["e"] < 0).mean()) if len(sub) else np.nan,
                "pvalue_wilcoxon": w["pvalue"],
            })
    return pd.DataFrame(reg_rows), pd.DataFrame(test_rows)
