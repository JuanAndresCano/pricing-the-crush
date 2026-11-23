"""Pregunta 2, parte 2: ¿el crush es instantáneo o sigue bajando en los días siguientes?

Uso:
    python scripts/09_crush_path.py                 # beta del control estimado con todos los eventos
    python scripts/09_crush_path.py --beta 0.5      # beta fijo

Lee data/raw/theta (con los días extra de scripts/08_extra_days.py), data/processed/calibration.csv y, si están
descargadas (scripts/10_control.py), las cadenas de SPY para descontar el movimiento general del mercado. Escribe
results/q2_path.csv (una fila por evento, vencimiento, día y modelo) y results/q2_path_summary.csv.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from crush.analysis.crush import control_slices, estimate_control_beta, path_table, summarize_path
from crush.config import CONTROL_DIR, PROCESSED_DIR, RAW_DIR, RESULTS_DIR
from crush.data.control import event_days, load_control
from crush.data.extra_days import load_extra_days
from crush.data.source import event_dir, load_event
from crush.market.rates import DEFAULT_RATE


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="theta")
    ap.add_argument("--root", default=str(RAW_DIR))
    ap.add_argument("--r", type=float, default=DEFAULT_RATE)
    ap.add_argument("--control", default="SPY", help="activo de control para descontar el mercado (se usa si ya está descargado)")
    ap.add_argument("--beta", type=float, default=None, help="sensibilidad a la volatilidad del control; por defecto se estima")
    ap.add_argument("--min-days-left", type=int, default=3, help="descarta vencimientos con menos días hábiles que estos")
    args = ap.parse_args()

    path = PROCESSED_DIR / "calibration.csv"
    if not path.exists():
        print(f"Falta {path}. Corre primero scripts/03_calibrate.py.")
        return 1
    jobs = []
    for _, row in pd.read_csv(path).iterrows():
        folder = event_dir(Path(args.root), args.source, row["ticker"], row["announcement_date"])
        if not (folder / "meta.json").exists():
            print(f"[omitido] {folder.name}: no está descargado")
            continue
        extra = load_extra_days(folder)
        if not extra:
            print(f"[omitido] {folder.name}: sin días extra (corre scripts/08_extra_days.py)")
            continue
        data = load_event(folder)
        ctrl = control_slices(load_control(CONTROL_DIR, args.source, args.control, event_days(data.meta)), args.r)
        if not ctrl:
            print(f"[aviso] {folder.name}: sin datos de {args.control}; las columnas ajustadas quedan vacías (scripts/10_control.py)")
        jobs.append((data, extra, row, ctrl))
    if not jobs:
        return 1

    def build(beta: float) -> pd.DataFrame:
        return pd.concat(
            [path_table(d, e, r, args.r, min_days_left=args.min_days_left, control=c, control_beta=beta) for d, e, r, c in jobs],
            ignore_index=True,
        )

    full = build(1.0 if args.beta is None else args.beta)
    if args.beta is None:
        est = estimate_control_beta(full)
        print(f"Beta estimado del control: {est['beta']:.3f}; constante (caída extra tras descontar el mercado): {est['alpha']:+.4f}; n = {est['n']}")
        if np.isfinite(est["beta"]):
            full = build(est["beta"])
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    full.to_csv(RESULTS_DIR / "q2_path.csv", index=False)
    summary = summarize_path(full)
    summary.to_csv(RESULTS_DIR / "q2_path_summary.csv", index=False)
    print(summary.round(4).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
