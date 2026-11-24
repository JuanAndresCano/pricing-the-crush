"""Contraste de crush gradual con otro activo de control (por defecto QQQ) en lugar de SPY.

Uso (primero baja las cadenas del control con el Theta Terminal abierto):
    python scripts/10_control.py --symbol QQQ --strike-range 40
    python scripts/15_alt_control.py --control QQQ

Hace lo mismo que scripts/09_crush_path.py y scripts/11_gradual_test.py, pero escribe en archivos con el símbolo
en el nombre y no toca los resultados principales:
    results/q2_path_<SYM>.csv, results/q2_gradual_regression_<SYM>.csv, results/q2_gradual_tests_<SYM>.csv
Motivación: SPY puede descontar mal la volatilidad de las tecnológicas; QQQ se parece más a su canasta.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from crush.analysis.crush import control_slices, estimate_control_beta, path_table
from crush.analysis.gradual import gradual_report
from crush.config import CONTROL_DIR, PROCESSED_DIR, RAW_DIR, RESULTS_DIR
from crush.data.control import event_days, load_control
from crush.data.extra_days import load_extra_days
from crush.data.source import event_dir, load_event
from crush.market.rates import DEFAULT_RATE


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="theta")
    ap.add_argument("--control", default="QQQ")
    ap.add_argument("--r", type=float, default=DEFAULT_RATE)
    ap.add_argument("--min-days-left", type=int, default=3)
    args = ap.parse_args()

    cal_path = PROCESSED_DIR / "calibration.csv"
    if not cal_path.exists():
        print("Falta data/processed/calibration.csv. Corre primero el pipeline.")
        return 1
    jobs = []
    for _, row in pd.read_csv(cal_path).iterrows():
        folder = event_dir(Path(RAW_DIR), args.source, row["ticker"], row["announcement_date"])
        if not (folder / "meta.json").exists():
            continue
        extra = load_extra_days(folder)
        if not extra:
            continue
        data = load_event(folder)
        ctrl = control_slices(load_control(CONTROL_DIR, args.source, args.control, event_days(data.meta)), args.r)
        jobs.append((data, extra, row, ctrl))
    if not jobs or not any(c for *_, c in jobs):
        print(f"No hay datos de {args.control}. Corre scripts/10_control.py --symbol {args.control}.")
        return 1

    def build(beta: float) -> pd.DataFrame:
        return pd.concat([path_table(d, e, r, args.r, min_days_left=args.min_days_left, control=c, control_beta=beta) for d, e, r, c in jobs], ignore_index=True)

    full = build(1.0)
    est = estimate_control_beta(full)
    print(f"Beta estimado de {args.control}: {est['beta']:.3f}; constante {est['alpha']:+.4f}; n = {est['n']}")
    if np.isfinite(est["beta"]):
        full = build(est["beta"])
    reg, tests = gradual_report(full)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    full.to_csv(RESULTS_DIR / f"q2_path_{args.control}.csv", index=False)
    reg.to_csv(RESULTS_DIR / f"q2_gradual_regression_{args.control}.csv", index=False)
    tests.to_csv(RESULTS_DIR / f"q2_gradual_tests_{args.control}.csv", index=False)
    pd.set_option("display.width", 200)
    print(reg.round(4).to_string(index=False))
    print(tests.round(4).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
