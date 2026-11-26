"""Pregunta 1 comparando la misma cantidad en el mercado y en el modelo.

La comparación principal pone el K_repl del mercado (smile interpolado en los strikes que cotizan) contra la fórmula
cerrada del modelo, que integra de 0 a infinito: parte de la diferencia es recorte de strikes y no error del modelo
(corrección 2 propuesta al marco teórico). Aquí los precios del modelo se calculan en los mismos strikes del mercado y
se les aplica exactamente la misma receta, en tres versiones:

  - `extrap`: smile interpolado y extendido de forma plana (la del análisis principal);
  - `trunc`: smile interpolado solo en el rango que cotiza (sin extrapolar);
  - `disc`: la suma (44) sobre los strikes que cotizan.

Para cada receta, `error_modelo = K_mod(receta) / K_mkt(receta) - 1` y `cerrada = K_mod(cerrada) / K_mkt(receta) - 1`.
También se reportan las diferencias en puntos de volatilidad, sqrt(K_mod) - sqrt(K_mkt) (sección 7 del marco teórico).
Los strikes donde el precio del modelo no tiene volatilidad implícita (precio numéricamente nulo en las alas) se quitan
de los dos lados.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from crush.analysis.compare import MODELS, clustered_wilcoxon, theta_from_row
from crush.market.chain import CleanParams, build_slices
from crush.market.rates import event_rates
from crush.market.strike_mkt import k_repl_discrete, k_repl_smooth
from crush.models.black_scholes import implied_vol_array
from crush.models.fourier import CallPricer

RECIPES = {
    "extrap": lambda K, iv, s: k_repl_smooth(K, iv, s.T, s.r, s.F, extrapolate=True),
    "trunc": lambda K, iv, s: k_repl_smooth(K, iv, s.T, s.r, s.F, extrapolate=False),
    "disc": lambda K, iv, s: k_repl_discrete(K, iv, s.T, s.r, s.F),
}


def model_smile(model, theta, K, F: float, T: float, r: float) -> np.ndarray:
    """Volatilidad implícita del modelo (con el evento) en los strikes `K`, usando la opción fuera del dinero."""
    K = np.asarray(K, float)
    calls = model.call_prices(theta, F, T, r, True, CallPricer(K))
    is_call = K >= F
    price = np.where(is_call, calls, calls - np.exp(-r * T) * (F - K))
    return implied_vol_array(price, F, K, T, r, is_call)


def same_recipe_rows(data, cal_row, r: float, params: CleanParams | None = None) -> list[dict]:
    """Una fila por vencimiento (los dos más cercanos), receta y modelo."""
    m = data.meta
    slices = build_slices(data.pre, m["pre_day"], m["spot_pre"], event_rates(m, r)[0], params)[:2]
    rows = []
    for i, s in enumerate(slices, start=1):
        smiles = {tag: model_smile(model, theta_from_row(cal_row, tag), s.K, s.F, s.T, s.r) for tag, model in MODELS.items()}
        ok = np.isfinite(s.iv)
        for iv in smiles.values():
            ok &= np.isfinite(iv) & (iv > 0)
        for recipe, fn in RECIPES.items():
            k_mkt = fn(s.K[ok], s.iv[ok], s)
            for tag in MODELS:
                rows.append({
                    "ticker": m["ticker"], "announcement_date": m["announcement_date"], "maturity": i, "recipe": recipe, "model": tag,
                    "n_strikes": int(ok.sum()), "n_dropped": int((~ok).sum()), "k_mkt": k_mkt,
                    "k_mod_same": fn(s.K[ok], smiles[tag][ok], s), "k_mod_closed": float(cal_row[f"k_{tag}_{i}"]),
                })
    return rows


def summarize_same_recipe(table: pd.DataFrame) -> pd.DataFrame:
    """Por receta, forma de comparar y vencimiento (y el promedio de ambos): errores de cada modelo y Wilcoxon por fecha."""
    t = table.copy()
    t["announcement_date"] = t["announcement_date"].astype(str)
    rows = []
    for comparison, col in (("misma receta", "k_mod_same"), ("fórmula cerrada", "k_mod_closed")):
        t["rel"] = (t[col] / t["k_mkt"] - 1.0).abs()
        t["vol"] = (np.sqrt(t[col]) - np.sqrt(t["k_mkt"])).abs() * 100.0
        for recipe in RECIPES:
            wide = t[t["recipe"] == recipe].pivot_table(index=["ticker", "announcement_date", "maturity"], columns="model", values=["rel", "vol"]).dropna()
            for label in (1, 2, "promedio"):
                w_ = wide.groupby(level=["ticker", "announcement_date"]).mean() if label == "promedio" else wide.xs(label, level="maturity")
                dates = pd.Series(w_.index.get_level_values("announcement_date"), index=w_.index)
                w = clustered_wilcoxon(w_[("rel", "merton")] - w_[("rel", "bates")], dates)
                row = {"comparison": comparison, "recipe": recipe, "maturity": label, "n_events": w["n_events"], "n_dates": w["n_dates"],
                       "share_bates_closer": w["share_bates_closer"], "pvalue": w["pvalue"]}
                for tag in MODELS:
                    row[f"mean_abs_rel_{tag}"] = float(w_[("rel", tag)].mean())
                    row[f"median_abs_rel_{tag}"] = float(w_[("rel", tag)].median())
                    row[f"mean_abs_volpts_{tag}"] = float(w_[("vol", tag)].mean())
                    row[f"median_abs_volpts_{tag}"] = float(w_[("vol", tag)].median())
                rows.append(row)
    return pd.DataFrame(rows)


def range_effect(table: pd.DataFrame) -> pd.DataFrame:
    """Cuánto de la fórmula cerrada queda fuera de cada receta: K_cerrada / K_mod(receta) - 1, por modelo."""
    t = table.assign(effect=table["k_mod_closed"] / table["k_mod_same"] - 1.0)
    return t.groupby(["recipe", "model", "maturity"])["effect"].agg(["mean", "median", "size"]).reset_index()
