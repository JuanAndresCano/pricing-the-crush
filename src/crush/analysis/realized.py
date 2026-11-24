"""Movimiento realizado contra movimiento implícito del anuncio.

Para cada evento se compara lo que el mercado cobró por el anuncio, la varianza de evento eps de la ecuación (26),
con lo que la acción se movió de verdad:

    R_e = ln(S_post / S_pre)           retorno logarítmico entre los dos cierres del evento
    R_e^2  contra  eps                  movimiento realizado al cuadrado contra el implícito

Con eps medido en el vencimiento más cercano (ver `market.event_mkt.event_row`), una diferencia eps - R_e^2 positiva
significa que el mercado cobró más de lo que se movió la acción (prima de varianza del evento, como en Gao, Xing y
Zhang 2018); negativa, que se movió más de lo cobrado.

Valor de un variance swap comprado antes del anuncio, justo después de que ocurre (ecuaciones (35) y (36)):

    V = N [ R_e^2 - eps - sigma_b^2 t ] / T          (exacta, sin descuento)
    V ~ N (R_e^2 - eps) / T                          (36), si se ignora la varianza base del día, sigma_b^2 t

con t = 1 día, T el plazo del swap y N el nocional de varianza. El crush por sí solo no genera pérdida: lo que
importa es si R_e^2 supera o no a eps.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from crush.analysis.compare import clustered_wilcoxon
from crush.config import TRADING_DAYS_PER_YEAR
from crush.market.chain import CleanParams
from crush.market.event_mkt import event_row

SPLIT_GUARD = 0.5  # un retorno logarítmico mayor a esto entre dos cierres seguidos sugiere un split sin ajustar


def realized_move(meta: dict) -> float:
    """Retorno logarítmico entre el cierre previo y el posterior del evento."""
    return float(np.log(meta["spot_post"] / meta["spot_pre"]))


def event_swap_value(R_e: float, eps: float, T: float, sigma_b: float = 0.0, t: float = 1.0 / TRADING_DAYS_PER_YEAR, notional: float = 1.0) -> float:
    """Valor de un variance swap largo tras el anuncio. Con `sigma_b = 0` es la aproximación (36)."""
    return notional * (R_e**2 - eps - sigma_b**2 * t) / T


def realized_row(data, r: float, params: CleanParams | None = None) -> dict:
    """Fila por evento: movimiento realizado, implícito (ATM y strike) y su diferencia."""
    m = data.meta
    mk = event_row(data, r, params)
    R = realized_move(m)
    row = {
        "ticker": m["ticker"], "announcement_date": m["announcement_date"], "timing": m["timing"],
        "R_e": R, "R2": R**2, "abs_move": abs(R), "split_suspect": abs(R) > SPLIT_GUARD,
    }
    for how in ("atm", "strike"):
        eps = mk.get(f"eps_{how}", np.nan)
        sb = mk.get(f"sigma_b_{how}", np.nan)
        row[f"eps_{how}"] = eps
        row[f"sigma_b_{how}"] = sb
        ok = np.isfinite(eps) and eps > 0
        row[f"implied_move_{how}"] = float(np.sqrt(eps)) if ok else np.nan
        row[f"r2_over_eps_{how}"] = R**2 / eps if ok else np.nan
        row[f"z_{how}"] = abs(R) / np.sqrt(eps) if ok else np.nan  # movimiento realizado en unidades del implícito
        row[f"vrp_{how}"] = eps - R**2 if np.isfinite(eps) else np.nan  # positiva: el mercado cobró más de lo que se movió
        T1 = mk.get("T_1")
        row[f"swap_value_{how}"] = (
            event_swap_value(R, eps, T1, sb if np.isfinite(sb) else 0.0) if (np.isfinite(eps) and T1) else np.nan
        )
    row["cr_obs"] = mk.get("cr_obs", np.nan)
    return row


def add_history(df: pd.DataFrame, min_prior: int = 2) -> pd.DataFrame:
    """Agrega el promedio de R^2 de los anuncios anteriores de la misma empresa (contraparte bajo la medida física).

    Con menos de `min_prior` anuncios previos queda en nan. Solo usa eventos con fecha anterior, nunca el propio.
    """
    out = df.sort_values(["ticker", "announcement_date"]).copy()
    prior_mean = out.groupby("ticker")["R2"].transform(lambda s: s.shift().expanding().mean())
    n_prior = out.groupby("ticker")["R2"].transform(lambda s: s.shift().expanding().count())
    out["hist_mean_r2"] = prior_mean.where(n_prior >= min_prior)
    out["eps_over_hist_atm"] = out["eps_atm"] / out["hist_mean_r2"]
    return out.sort_index()


def merge_models(df: pd.DataFrame, cal: pd.DataFrame) -> pd.DataFrame:
    """Une `calibration.csv` (eps de cada modelo) y agrega R^2 / eps_modelo."""
    cols = ["ticker", "announcement_date"] + [c for c in cal.columns if c.startswith(("eps_repl_", "eps_qv_"))]
    left = df.assign(announcement_date=df["announcement_date"].astype(str))
    right = cal[cols].assign(announcement_date=cal["announcement_date"].astype(str))
    out = left.merge(right, on=["ticker", "announcement_date"], how="left")
    for c in [c for c in out.columns if c.startswith("eps_repl_")]:
        out[f"r2_over_{c}"] = out["R2"] / out[c]
    return out


def summarize_realized(df: pd.DataFrame, how: str = "atm") -> dict:
    """Resumen sobre todos los eventos con `how` = 'atm' o 'strike'. El contraste es Wilcoxon por fecha de anuncio."""
    ok = df[~df["split_suspect"] & df[f"eps_{how}"].gt(0)]
    d = ok[f"vrp_{how}"]
    res = clustered_wilcoxon(d, ok["announcement_date"].astype(str))
    return {
        "n_events": int(len(ok)),
        "n_excluded": int(len(df) - len(ok)),
        "mean_implied_move": float(ok[f"implied_move_{how}"].mean()),
        "mean_abs_realized": float(ok["abs_move"].mean()),
        "median_r2_over_eps": float(ok[f"r2_over_eps_{how}"].median()),
        "mean_vrp": res["mean_d"],
        "share_implied_gt_realized": res["share_bates_closer"],  # fracción con d = eps - R^2 > 0
        "n_dates": res["n_dates"],
        "pvalue_wilcoxon": res["pvalue"],
    }
