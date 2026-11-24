"""Pruebas de robustez de los resultados de las Preguntas 1 y 2 (sin descargar ni recalibrar nada).

Cada escenario es un subconjunto de eventos (por año, sin la ventana del choque arancelario de abril de 2025, por
horario del anuncio, sin una empresa a la vez, ...). Para cada uno se repite el contraste principal:

  - Pregunta 1: `compare.summarize_q1` sobre la tabla de calibración del subconjunto.
  - Pregunta 2: regresión agrupada por fecha (`gradual.cluster_ols`), y el mismo contraste con el beta del control
    estimado dejando fuera la fecha (validación cruzada por fecha), que quita el optimismo de estimar el beta con
    los mismos datos que se contrastan.

Son contrastes descriptivos para ver si la conclusión depende de un subconjunto; no se corrigen por comparaciones
múltiples y no sustituyen a los contrastes principales.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from crush.analysis.compare import clustered_wilcoxon, summarize_q1
from crush.analysis.gradual import _prepare, cluster_ols

# Ventana del choque arancelario de abril de 2025 (aranceles del 2 de abril y recuperación posterior).
TARIFF_WINDOW = ("2025-04-01", "2025-05-09")


def scenarios(events: pd.DataFrame) -> dict[str, pd.Series]:
    """{nombre: máscara booleana sobre `events`} con columnas ticker, announcement_date (texto) y timing."""
    d = events["announcement_date"].astype(str)
    out = {
        "todos": pd.Series(True, index=events.index),
        "2025": d.str.startswith("2025"),
        "2026": d.str.startswith("2026"),
        "sin_choque_abril_2025": ~((d >= TARIFF_WINDOW[0]) & (d <= TARIFF_WINDOW[1])),
    }
    if "timing" in events:
        out["solo_amc"] = events["timing"] == "amc"
        out["solo_bmo"] = events["timing"] == "bmo"
    for t in sorted(events["ticker"].unique()):
        out[f"sin_{t}"] = events["ticker"] != t
    return out


def _with_event_info(table: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    ev = events[["ticker", "announcement_date", "timing"]].copy()
    ev["announcement_date"] = ev["announcement_date"].astype(str)
    t = table.copy()
    t["announcement_date"] = t["announcement_date"].astype(str)
    return t.merge(ev, on=["ticker", "announcement_date"], how="left")


def q1_robustness(calibration: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    """Contraste de la Pregunta 1 en cada escenario, por vencimiento (más el promedio de los dos vencimientos)."""
    cal = _with_event_info(calibration, events)
    rows = []
    for name, mask in scenarios(cal).items():
        sub = cal[mask]
        if len(sub) < 6:
            continue
        s = summarize_q1(sub)
        for mat, r in s.iterrows():
            rows.append({"scenario": name, "test": f"vencimiento {mat}", "n_events": int(r["n_events"]), "n_dates": int(r["n_dates"]),
                         "share_bates_closer": r["share_bates_closer"], "mean_d": r["mean_d"], "pvalue": r["pvalue"],
                         "mae_merton": r["mae_merton"], "mae_bates": r["mae_bates"]})
        # un solo contraste por evento: promedio de d en los dos vencimientos (evita contar dos veces el mismo evento)
        d = ((sub["relerr_merton_1"].abs() - sub["relerr_bates_1"].abs()) + (sub["relerr_merton_2"].abs() - sub["relerr_bates_2"].abs())) / 2
        w = clustered_wilcoxon(d, sub["announcement_date"].astype(str))
        rows.append({"scenario": name, "test": "promedio de vencimientos", "n_events": w["n_events"], "n_dates": w["n_dates"],
                     "share_bates_closer": w["share_bates_closer"], "mean_d": w["mean_d"], "pvalue": w["pvalue"],
                     "mae_merton": np.nan, "mae_bates": np.nan})
    return pd.DataFrame(rows)


def crossfit_alpha(t: pd.DataFrame) -> dict:
    """Caída extra media con beta estimado sin la fecha evaluada. `t` sale de `gradual._prepare`.

    Para cada fecha d se estima beta (MCO simple de ln g - ln g_modelo sobre ln g_spy) con las demás fechas y se
    calcula e = ln g - beta_{-d} ln g_spy - ln g_modelo. Se promedia e por fecha y se hace una t de una muestra
    sobre esos promedios (G - 1 grados de libertad) y un Wilcoxon.
    """
    dates = t["announcement_date"].to_numpy()
    uniq = pd.unique(dates)
    y = (t["ln_g"] - t["ln_model"]).to_numpy()
    x = t["ln_spy"].to_numpy()
    e = np.empty(len(t))
    for d in uniq:
        out = dates == d
        xin, yin = x[~out], y[~out]
        beta = np.polyfit(xin, yin, 1)[0] if np.ptp(xin) > 0 and len(xin) > 2 else 0.0
        e[out] = y[out] - beta * x[out]
    by = pd.Series(e).groupby(dates).mean()
    res = {"alpha_cv": float(by.mean()), "n_dates": int(by.size), "p_t_cv": np.nan, "p_wilcoxon_cv": np.nan}
    if by.size >= 3 and by.std(ddof=1) > 0:
        res["p_t_cv"] = float(stats.ttest_1samp(by, 0.0).pvalue)
    if by.size >= 6 and not (by == 0).all():
        res["p_wilcoxon_cv"] = float(stats.wilcoxon(by, zero_method="wilcox").pvalue)
    return res


def q2_robustness(path: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    """Contraste de crush gradual (Pregunta 2) en cada escenario y modelo."""
    tab = _with_event_info(path, events)
    rows = []
    for name, mask in scenarios(tab).items():
        sub = tab[mask]
        for model in sorted(sub["model"].unique()):
            t = _prepare(sub, model)
            if t["announcement_date"].nunique() < 3:
                continue
            reg = cluster_ols(t["ln_spy"], t["ln_g"] - t["ln_model"], t["announcement_date"])
            cv = crossfit_alpha(t)
            rows.append({"scenario": name, "model": model, "n_obs": reg["n"], "n_dates": reg["n_clusters"], "beta": reg["beta"],
                         "alpha": reg["alpha"], "p_alpha": reg["p_alpha"], "alpha_cv": cv["alpha_cv"], "p_t_cv": cv["p_t_cv"],
                         "p_wilcoxon_cv": cv["p_wilcoxon_cv"]})
    return pd.DataFrame(rows)
