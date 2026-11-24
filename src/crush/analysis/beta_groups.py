"""¿El beta del control (SPY) es el mismo para todas las empresas o varía por sector?

Hoy se usa un beta único en `ln g_obs = alpha + beta ln g_spy`. Aquí se estima por empresa y por sector (con la tabla
de Merton, donde `g_model = 1`), con errores robustos por fecha de anuncio, y se contrasta que los betas sean iguales
con una prueba de Wald conjunta. También se mide cuánto cambia la caída extra `alpha` si cada grupo usa su propio beta.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

SECTOR = {  # GICS, un nivel
    "AAPL": "Tecnología", "MSFT": "Tecnología", "NVDA": "Tecnología",
    "META": "Comunicaciones", "NFLX": "Comunicaciones",
    "AMZN": "Consumo discrecional", "TSLA": "Consumo discrecional", "HD": "Consumo discrecional",
    "WMT": "Consumo básico", "JPM": "Financiero",
}


def _prep(path: pd.DataFrame) -> pd.DataFrame:
    t = path[(path["model"] == "merton") & (path["k"] > 0)].copy()
    t = t[np.isfinite(t["g_obs"]) & np.isfinite(t["g_spy"]) & (t["g_obs"] > 0) & (t["g_spy"] > 0)]
    t["y"], t["x"] = np.log(t["g_obs"]), np.log(t["g_spy"])
    t["announcement_date"] = t["announcement_date"].astype(str)
    t["sector"] = t["ticker"].map(SECTOR)
    return t.reset_index(drop=True)


def _fit(X, y, clusters):
    codes, uniq = pd.factorize(pd.Series(clusters))
    G, (n, p) = len(uniq), X.shape
    XtX_inv = np.linalg.pinv(X.T @ X)
    coef = XtX_inv @ X.T @ y
    resid = y - X @ coef
    meat = np.zeros((p, p))
    for g in range(G):
        s = X[codes == g].T @ resid[codes == g]
        meat += np.outer(s, s)
    return coef, XtX_inv @ meat @ XtX_inv * (G / (G - 1)) * ((n - 1) / (n - p)), G


def beta_by_group(path: pd.DataFrame, by: str) -> tuple[pd.DataFrame, dict]:
    """Beta y constante por grupo (`by` = "ticker" o "sector") y prueba de Wald de que todos los betas son iguales.

    La prueba usa un solo modelo con constante y pendiente propias de cada grupo y errores por fecha de anuncio; el
    estadístico Wald / q se compara con una F(q, fechas - 1).
    """
    t = _prep(path)
    groups = sorted(t[by].unique())
    X = np.column_stack([np.column_stack([(t[by] == g).astype(float), (t[by] == g) * t["x"]]) for g in groups])
    coef, V, C = _fit(X, t["y"].to_numpy(), t["announcement_date"])
    rows = []
    for i, g in enumerate(groups):
        s = t[t[by] == g]
        a, b = coef[2 * i], coef[2 * i + 1]
        se_b = float(np.sqrt(V[2 * i + 1, 2 * i + 1]))
        rows.append({by: g, "n_obs": int(len(s)), "n_events": int(s[["ticker", "announcement_date"]].drop_duplicates().shape[0]),
                     "beta": float(b), "se_beta": se_b, "ci_low": float(b - stats.t.ppf(0.975, C - 1) * se_b),
                     "ci_high": float(b + stats.t.ppf(0.975, C - 1) * se_b), "alpha": float(a)})
    q = len(groups) - 1
    R = np.zeros((q, X.shape[1]))
    for j in range(q):
        R[j, 2 * j + 1], R[j, 2 * (j + 1) + 1] = 1.0, -1.0
    d = R @ coef
    wald = float(d @ np.linalg.pinv(R @ V @ R.T) @ d)
    # F con C - 1 grados de libertad de los errores; con pocas fechas por grupo esta prueba tiene poca potencia
    test = {"by": by, "groups": len(groups), "wald": wald, "df": q, "pvalue": float(stats.f.sf(wald / q, q, C - 1)), "n_dates": int(C)}
    return pd.DataFrame(rows), test


def alpha_with_group_beta(path: pd.DataFrame, by: str) -> dict:
    """Caída extra media (y su error por fecha) usando el beta de cada grupo en lugar de un beta único."""
    t = _prep(path)
    tab, _ = beta_by_group(path, by)
    b = t[by].map(dict(zip(tab[by], tab["beta"])))
    t["e_group"] = t["y"] - b * t["x"]
    bg = float(np.polyfit(t["x"], t["y"], 1)[0])
    t["e_single"] = t["y"] - bg * t["x"]
    by_date = t.groupby("announcement_date")[["e_group", "e_single"]].mean()
    return {"by": by, "alpha_group_beta": float(t["e_group"].mean()), "alpha_single_beta": float(t["e_single"].mean()),
            "p_wilcoxon_group_beta": float(stats.wilcoxon(by_date["e_group"]).pvalue),
            "p_wilcoxon_single_beta": float(stats.wilcoxon(by_date["e_single"]).pvalue)}
