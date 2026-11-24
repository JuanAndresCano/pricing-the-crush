"""Contraste del placebo de ventana (experimento E4): ¿la caída extra de la Pregunta 2 aparece también sin anuncio?

Uso:
    python scripts/17_placebo.py

Lee data/placebo (scripts/16_placebo_download.py), las cadenas de SPY y results/q2_path.csv. Escribe
results/placebo_groups.csv (caída extra real y placebo por horizonte), results/placebo_effect.csv (regresión conjunta y
diferencia por evento) y results/placebo_table.csv (una fila por evento, vencimiento y día).
Hipótesis y lectura en docs/experimentos.md.
"""

from __future__ import annotations

import argparse
import sys

import pandas as pd

from crush.analysis.crush import control_slices
from crush.analysis.gradual import cluster_ols
from crush.analysis.placebo import group_report, paired_by_event, placebo_table, real_table, stacked_effect, _clean
from crush.config import CONTROL_DIR, PLACEBO_DIR, RESULTS_DIR
from crush.data.control import load_control
from crush.data.placebo_days import load_placebo, placebo_days
from crush.market.rates import DEFAULT_RATE


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--control", default="SPY")
    ap.add_argument("--r", type=float, default=DEFAULT_RATE)
    args = ap.parse_args()

    path = RESULTS_DIR / "q2_path.csv"
    if not path.exists() or not PLACEBO_DIR.exists():
        print("Faltan results/q2_path.csv o data/placebo. Corre el pipeline y scripts/16_placebo_download.py.")
        return 1
    tables = []
    for f in sorted(PLACEBO_DIR.glob("*_*")):
        if not (f / "meta.json").exists():
            continue
        meta, days = load_placebo(f)
        if 0 not in days or len(days) < 2:
            continue
        ctrl = control_slices(load_control(CONTROL_DIR, "theta", args.control, placebo_days(meta)), args.r)
        t = placebo_table(meta, days, ctrl, args.r)
        if len(t):
            tables.append(t)
    if not tables:
        print("No hay datos de placebo utilizables.")
        return 1
    placebo = pd.concat(tables, ignore_index=True)
    real = real_table(pd.read_csv(path))
    real["announcement_date"] = real["announcement_date"].astype(str)
    # solo eventos que tienen placebo, para comparar lo mismo con lo mismo
    keys = set(zip(placebo["ticker"], placebo["announcement_date"]))
    real_m = real[[(a, b) in keys for a, b in zip(real["ticker"], real["announcement_date"])]]

    groups = pd.DataFrame(group_report(real_m, "real (Merton, g_model = 1)") + group_report(placebo, "placebo"))
    beta_real = cluster_ols(_clean(real_m)["ln_spy"], _clean(real_m)["ln_g"], _clean(real_m)["announcement_date"])["beta"]
    effect = [
        {"analysis": "regresión conjunta", **stacked_effect(real_m, placebo)},
        {"analysis": "regresión conjunta, control por días restantes", **stacked_effect(real_m, placebo, controls_days_left=True)},
        {"analysis": f"diferencia por evento (beta real {beta_real:.3f})", **paired_by_event(real_m, placebo, beta_real)},
    ]
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    placebo.to_csv(RESULTS_DIR / "placebo_table.csv", index=False)
    groups.to_csv(RESULTS_DIR / "placebo_groups.csv", index=False)
    pd.DataFrame(effect).to_csv(RESULTS_DIR / "placebo_effect.csv", index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    print(f"Eventos con placebo: {len(keys)}; filas placebo {len(placebo)}, filas reales comparables {len(real_m)}")
    print(groups.round(4).to_string(index=False))
    print(pd.DataFrame(effect).round(4).T.to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
