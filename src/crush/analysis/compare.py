"""Métricas de la Pregunta 1: ¿qué modelo se acerca más al strike de replicación del mercado?

Entrada: la tabla `calibration.csv` (una fila por evento, ver `calibration.real.calibration_row`).

  - Error absoluto relativo del strike en cada vencimiento y modelo: |K_mod / K_mkt - 1|.
  - Diferencia pareada d = err_Merton - err_Bates (positiva si Bates se acerca más).
  - Contraste de Wilcoxon sobre d. Los eventos de una misma fecha de anuncio comparten el mercado de ese día, así
    que no son independientes: se promedia d por fecha y el contraste se hace sobre esos promedios (n = fechas).
  - Descomposición de (45): K_B - K_M = parte continua (v_bar - sigma^2) + parte del evento (eps_B - eps_M)/T.
  - BIC: cuántas veces gana cada modelo (menor es mejor).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from crush.calibration.calibrate import calibrate
from crush.calibration.quotes import OptionChain
from crush.calibration.selection import rmse_iv
from crush.models.event_models import BatesEvent, EventModel, MertonEvent

MODELS = {"merton": MertonEvent(), "bates": BatesEvent()}


def theta_from_row(row, tag: str) -> np.ndarray:
    m = MODELS[tag]
    return np.array([row[f"{tag}_{n}"] for n in m.names], dtype=float)


def strike_errors(df: pd.DataFrame, maturities=(1, 2)) -> pd.DataFrame:
    """Agrega columnas `abs_err_<modelo>_<i>` y `d_<i>` (positiva si Bates gana)."""
    out = df.copy()
    for i in maturities:
        for tag in ("merton", "bates"):
            out[f"abs_err_{tag}_{i}"] = out[f"relerr_{tag}_{i}"].abs()
        out[f"d_{i}"] = out[f"abs_err_merton_{i}"] - out[f"abs_err_bates_{i}"]
    return out


def decompose(df: pd.DataFrame, maturities=(1, 2)) -> pd.DataFrame:
    """Descomposición (45) por evento y vencimiento: K_B - K_M = continua + evento."""
    rows = []
    for _, r in df.iterrows():
        thM, thB = theta_from_row(r, "merton"), theta_from_row(r, "bates")
        for i in maturities:
            T = r[f"T_{i}"]
            cont = MODELS["bates"].base(thB, T) - MODELS["merton"].base(thM, T)
            ev = (MODELS["bates"].eps_repl(thB) - MODELS["merton"].eps_repl(thM)) / T
            rows.append({
                "ticker": r["ticker"], "announcement_date": r["announcement_date"], "maturity": i, "T": T,
                "K_merton": r[f"k_merton_{i}"], "K_bates": r[f"k_bates_{i}"], "K_mkt": r[f"k_mkt_{i}"],
                "diff_continuous": cont, "diff_event": ev, "diff_total": cont + ev,
                "event_share_merton": MODELS["merton"].eps_repl(thM) / T / r[f"k_merton_{i}"],
                "event_share_bates": MODELS["bates"].eps_repl(thB) / T / r[f"k_bates_{i}"],
            })
    return pd.DataFrame(rows)


def clustered_wilcoxon(d: pd.Series, dates: pd.Series) -> dict:
    """Wilcoxon de signos y rangos sobre el promedio de `d` por fecha. Alternativa: mediana(d) distinta de 0."""
    by_date = d.groupby(dates).mean().dropna()
    res = {
        "n_events": int(d.notna().sum()), "n_dates": int(by_date.size), "mean_d": float(d.mean()), "median_d": float(d.median()),
        "share_bates_closer": float((d.dropna() > 0).mean()) if d.notna().any() else float("nan"),
    }
    if by_date.size < 6 or (by_date == 0).all():  # con menos de 6 fechas el contraste exacto no puede dar p < 0,05
        res["pvalue"] = float("nan")
    else:
        res["pvalue"] = float(wilcoxon(by_date, zero_method="wilcox").pvalue)
    return res


def summarize_q1(df: pd.DataFrame, maturities=(1, 2)) -> pd.DataFrame:
    e = strike_errors(df, maturities)
    rows = []
    for i in maturities:
        r = clustered_wilcoxon(e[f"d_{i}"], e["announcement_date"])
        r.update(maturity=i, mae_merton=float(e[f"abs_err_merton_{i}"].mean()), mae_bates=float(e[f"abs_err_bates_{i}"].mean()))
        rows.append(r)
    out = pd.DataFrame(rows).set_index("maturity")
    out.attrs["bic_bates_wins"] = float((e["bic_bates"] < e["bic_merton"]).mean())
    out.attrs["rmse_iv_merton"] = float(e["rmse_iv_merton"].mean())
    out.attrs["rmse_iv_bates"] = float(e["rmse_iv_bates"].mean())
    return out


# --- validación fuera de muestra -------------------------------------------------------------------


def split_chain(chain: OptionChain, step: int = 2) -> tuple[OptionChain, OptionChain]:
    """Entrena con strikes alternos y evalúa con los demás, dentro de cada vencimiento."""
    train = np.zeros(len(chain), dtype=bool)
    for _, _, _, _, idx in chain.groups:
        order = idx[np.argsort(chain.K[idx])]
        train[order[::step]] = True
    return chain.subset(train), chain.subset(~train)


def holdout_rmse(model: EventModel, chain: OptionChain, fixed=None, seed: int = 0, **kw) -> dict:
    """Calibra con la mitad de los strikes y mide RMSE_IV con la otra mitad (strikes que el modelo no vio)."""
    train, test = split_chain(chain)
    res = calibrate(model, train, seed=seed, fixed=fixed, **kw)
    return {"rmse_train": res.rmse_iv, "rmse_test": rmse_iv(model, res.theta, test)}
