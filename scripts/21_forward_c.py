"""Varianza forward (escalón del anuncio) y parámetro c de reajuste (experimento E8). No descarga nada.

Uso: python scripts/21_forward_c.py
Lee data/raw/theta, data/processed/calibration.csv y las cadenas de SPY. Escribe results/forward_step.csv,
results/forward_step_summary.csv, results/c_table.csv y results/c_summary.csv. Ver analysis/forward_c.py.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from crush.analysis.compare import clustered_wilcoxon
from crush.analysis.crush import control_slices
from crush.analysis.forward_c import c_table, forward_step_row
from crush.config import CONTROL_DIR, PROCESSED_DIR, RAW_DIR, RESULTS_DIR
from crush.data.control import event_days, load_control
from crush.data.extra_days import load_extra_days
from crush.data.source import event_dir, load_event
from crush.market.rates import DEFAULT_RATE

BETA = 0.4519  # beta del control con Merton en la muestra completa (results/q2_gradual_regression.csv)


def summarize_forward(df: pd.DataFrame):
    d = df.copy()
    d["announcement_date"] = d["announcement_date"].astype(str)
    rows = []
    for tag in ("mkt", "merton", "bates"):
        rows.append({"source": tag, "mean_step_pre": d[f"{tag}_step_pre"].mean(), "median_step_pre": d[f"{tag}_step_pre"].median(),
                     "mean_step_post": d[f"{tag}_step_post"].mean(), "median_step_post": d[f"{tag}_step_post"].median(),
                     "share_step_pre_positive": float((d[f"{tag}_step_pre"] > 0).mean())})
    out = pd.DataFrame(rows)
    extra = []
    w = clustered_wilcoxon(d["mkt_step_pre"] - d["mkt_step_post"], d["announcement_date"])
    extra.append({"test": "mercado: escalón antes menos después (> 0: el escalón desaparece)", "n_events": w["n_events"], "n_dates": w["n_dates"],
                  "mean_d": w["mean_d"], "share": w["share_bates_closer"], "pvalue": w["pvalue"]})
    dd = (d["merton_step_pre"] - d["mkt_step_pre"]).abs() - (d["bates_step_pre"] - d["mkt_step_pre"]).abs()
    w = clustered_wilcoxon(dd, d["announcement_date"])
    extra.append({"test": "escalón antes del anuncio: error de Merton menos error de Bates (> 0: Bates más cerca)", "n_events": w["n_events"],
                  "n_dates": w["n_dates"], "mean_d": w["mean_d"], "share": w["share_bates_closer"], "pvalue": w["pvalue"]})
    return out, pd.DataFrame(extra)


def summarize_c(t: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model, k), g in t.groupby(["model", "k"]):
        for col in ("c_raw", "c_adj"):
            v = g[g[col].notna() & (g[col] > 0)]
            w = clustered_wilcoxon(np.log(v[col]), v["announcement_date"].astype(str)) if len(v) else {"n_dates": 0, "pvalue": np.nan}
            rows.append({"model": model, "k": k, "measure": col, "n": int(len(v)), "n_dates": w["n_dates"],
                         "median_c": float(v[col].median()) if len(v) else np.nan, "mean_c": float(v[col].mean()) if len(v) else np.nan,
                         "p_wilcoxon_ln_c_neq_0": w["pvalue"]})
    return pd.DataFrame(rows)


def main() -> int:
    cal = pd.read_csv(PROCESSED_DIR / "calibration.csv")
    fwd, ctabs = [], []
    for _, row in cal.iterrows():
        folder = event_dir(Path(RAW_DIR), "theta", row["ticker"], row["announcement_date"])
        if not (folder / "meta.json").exists():
            continue
        data = load_event(folder)
        f = forward_step_row(data, row, DEFAULT_RATE)
        if f:
            fwd.append(f)
        ctrl = control_slices(load_control(CONTROL_DIR, "theta", "SPY", event_days(data.meta)), DEFAULT_RATE)
        ctabs.append(c_table(data, load_extra_days(folder), row, DEFAULT_RATE, control=ctrl, control_beta=BETA))
        print(f"[listo] {folder.name}", flush=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    fdf = pd.DataFrame(fwd)
    fdf.to_csv(RESULTS_DIR / "forward_step.csv", index=False)
    s1, s2 = summarize_forward(fdf)
    pd.concat([s1, s2], ignore_index=True).to_csv(RESULTS_DIR / "forward_step_summary.csv", index=False)
    ct = pd.concat(ctabs, ignore_index=True)
    ct.to_csv(RESULTS_DIR / "c_table.csv", index=False)
    sc = summarize_c(ct)
    sc.to_csv(RESULTS_DIR / "c_summary.csv", index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 20)
    print(s1.round(4).to_string(index=False))
    print(s2.round(4).to_string(index=False))
    print(sc.round(4).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
