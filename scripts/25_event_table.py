"""Tabla de resultados por evento de la sección 7 del marco teórico (una fila por evento).

Uso (después de 05, 20, 23 y 24): python scripts/25_event_table.py
Junta, para los dos vencimientos más cercanos (sufijos _1 y _2): K_repl del mercado y de cada modelo (13) y (23), K_QV
de cada modelo (11) y (22) y el sesgo K_repl - K_QV, la participación de saltos s_M y s_B (45), el K_hat de Monte
Carlo con su error estándar (51), el crush observado y el de cada modelo (55), y la prima de varianza (56); por evento,
el RMSE de volatilidad implícita (53) de cada modelo. Escribe results/event_table.csv y results/model_bias_summary.csv.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

from crush.analysis.compare import MODELS, theta_from_row
from crush.config import PROCESSED_DIR, RAW_DIR, RESULTS_DIR
from crush.data.source import event_dir, load_event
from crush.market.chain import build_slices
from crush.market.rates import DEFAULT_RATE, event_rates

KEY = ["ticker", "announcement_date"]


def main() -> int:
    cal = pd.read_csv(PROCESSED_DIR / "calibration.csv")
    crush = pd.read_csv(RESULTS_DIR / "q2_crush.csv")
    mc = pd.read_csv(RESULTS_DIR / "mc_events.csv").set_index(KEY + ["model", "maturity"])
    vrp = pd.read_csv(RESULTS_DIR / "vrp_events.csv").set_index(KEY + ["maturity"])
    crush["expiry"] = pd.to_datetime(crush["expiry"]).dt.strftime("%Y-%m-%d")
    crush = crush.set_index(KEY + ["expiry", "model"])
    rows, bias = [], []
    for _, r in cal.iterrows():
        folder = event_dir(Path(RAW_DIR), "theta", r["ticker"], r["announcement_date"])
        data = load_event(folder)
        m = data.meta
        slices = build_slices(data.pre, m["pre_day"], m["spot_pre"], event_rates(m, DEFAULT_RATE)[0])[:2]
        key = (r["ticker"], r["announcement_date"])
        out = {"ticker": key[0], "announcement_date": key[1], "rmse_iv_merton": r["rmse_iv_merton"], "rmse_iv_bates": r["rmse_iv_bates"]}
        for i, s in enumerate(slices, start=1):
            expiry = str(pd.Timestamp(s.expiry).date())
            out[f"expiry_{i}"], out[f"k_mkt_{i}"] = expiry, r[f"k_mkt_{i}"]
            for tag, model in MODELS.items():
                theta = theta_from_row(r, tag)
                k_repl, k_qv = model.k_repl(theta, s.T, True), model.k_qv(theta, s.T, True)
                out[f"k_repl_{tag}_{i}"], out[f"k_qv_{tag}_{i}"], out[f"bias_{tag}_{i}"] = k_repl, k_qv, k_repl - k_qv
                out[f"jump_share_{tag}_{i}"] = model.eps_repl(theta) / s.T / k_repl
                sim = mc.loc[key + (tag, i)]
                out[f"mc_k_hat_{tag}_{i}"], out[f"mc_se_{tag}_{i}"] = sim["k_hat"], sim["se"]
                c = crush.loc[key + (expiry, tag)] if key + (expiry, tag) in crush.index else None
                out[f"cr_mkt_{i}"] = c["cr_obs"] if c is not None else out.get(f"cr_mkt_{i}")
                out[f"cr_{tag}_{i}"] = c["cr_model"] if c is not None else float("nan")
                bias.append({"model": tag, "maturity": i, "bias": k_repl - k_qv, "rel_bias": k_repl / k_qv - 1.0,
                             "jump_share": out[f"jump_share_{tag}_{i}"]})
            v = vrp.loc[key + (i,)] if key + (i,) in vrp.index else None
            out[f"rv_{i}"] = v["rv"] if v is not None else float("nan")
            out[f"vrp_{i}"] = v["vrp"] if v is not None else float("nan")
        rows.append(out)
    table = pd.DataFrame(rows)
    table.to_csv(RESULTS_DIR / "event_table.csv", index=False)
    s = pd.DataFrame(bias).groupby(["model", "maturity"]).agg(
        n=("bias", "size"), mean_bias=("bias", "mean"), median_bias=("bias", "median"), median_rel_bias=("rel_bias", "median"),
        mean_jump_share=("jump_share", "mean"), median_jump_share=("jump_share", "median")).reset_index()
    s.to_csv(RESULTS_DIR / "model_bias_summary.csv", index=False)
    pd.set_option("display.width", 200)
    print(f"{len(table)} eventos, {table.shape[1]} columnas")
    print(s.round(5).to_string(index=False))
    k_check = max((table[f"k_repl_{t}_{i}"] / cal[f"k_{t}_{i}"] - 1).abs().max() for t in MODELS for i in (1, 2))
    print(f"diferencia máxima con los strikes de la calibración: {k_check:.2e}")
    print("celdas vacías por columna:", table.isna().sum()[lambda x: x > 0].to_dict())
    return 0


if __name__ == "__main__":
    sys.exit(main())
