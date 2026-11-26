"""Prima de riesgo de varianza del evento hasta el vencimiento, ecuación (56) del marco teórico (sección 7).

    VRP_i = K_repl_mkt,i - RV_i

`K_repl_mkt` es el strike del mercado al cierre previo al anuncio y `RV_i` la varianza realizada anualizada de la acción
desde ese cierre hasta el vencimiento, con la convención del contrato (50): RV = 252/n * suma de los retornos
logarítmicos diarios al cuadrado, sin restar la media. Positiva: el mercado cobró más de lo que se realizó.

No es la medida de `analysis.realized` (varianza de evento menos R_e^2 del día del anuncio): esta incluye todos los
días hasta el vencimiento.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from crush.analysis.compare import clustered_wilcoxon
from crush.config import TRADING_DAYS_PER_YEAR
from crush.data.calendar import trading_days_between

SPLIT_LIMIT = 0.5  # |retorno diario| mayor: posible split sin ajustar, la ventana se excluye


def closes_from_eod(frame: pd.DataFrame) -> pd.Series:
    """Cierres diarios indexados por fecha (texto AAAA-MM-DD), a partir de la respuesta `stock/history/eod`."""
    if frame.empty:
        return pd.Series(dtype=float)
    day = pd.to_datetime(frame["created"], format="ISO8601").dt.strftime("%Y-%m-%d")
    s = pd.Series(frame["close"].to_numpy(float), index=day.to_numpy())
    s = s[s > 0]
    return s[~s.index.duplicated(keep="last")].sort_index()


def realized_to_expiry(closes: pd.Series, pre_day, expiry) -> dict | None:
    """RV desde el cierre de `pre_day` hasta el de `expiry`. None si falta algún cierre de la ventana."""
    a, b = str(pd.Timestamp(pre_day).date()), str(pd.Timestamp(expiry).date())
    window = closes[(closes.index >= a) & (closes.index <= b)]
    n = trading_days_between(a, b)
    if n < 1 or len(window) != n + 1 or window.index[0] != a or window.index[-1] != b:
        return None
    r = np.diff(np.log(window.to_numpy(float)))
    return {"n_days": int(n), "rv": float(TRADING_DAYS_PER_YEAR / n * np.sum(r**2)), "event_share": float(r[0] ** 2 / np.sum(r**2)) if np.sum(r**2) > 0 else np.nan,
            "max_abs_return": float(np.abs(r).max()), "split_suspect": bool(np.abs(r).max() > SPLIT_LIMIT)}


def summarize_vrp(table: pd.DataFrame) -> pd.DataFrame:
    """Por vencimiento: prima media y mediana (en varianza y en puntos de volatilidad) y Wilcoxon por fecha."""
    t = table[~table["split_suspect"]].copy()
    t["announcement_date"] = t["announcement_date"].astype(str)
    rows = []
    for i, g in t.groupby("maturity"):
        volpts = (np.sqrt(g["k_mkt"]) - np.sqrt(g["rv"])) * 100.0
        w = clustered_wilcoxon(g["vrp"], g["announcement_date"])
        rows.append({"maturity": i, "n_events": w["n_events"], "n_dates": w["n_dates"], "mean_k_mkt": float(g["k_mkt"].mean()), "mean_rv": float(g["rv"].mean()),
                     "mean_vrp": w["mean_d"], "median_vrp": w["median_d"], "share_positive": float((g["vrp"] > 0).mean()),
                     "mean_volpts": float(volpts.mean()), "median_volpts": float(volpts.median()), "median_event_share": float(g["event_share"].median()),
                     "pvalue_wilcoxon": w["pvalue"]})
    return pd.DataFrame(rows)
