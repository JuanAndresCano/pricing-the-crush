"""Pregunta 2: razón de crush observada vs la de Merton-evento y Bates-evento.

Uso:
    python scripts/05_crush.py --r 0.04

Lee data/raw/ y data/processed/calibration.csv. Escribe results/q2_crush.csv (una fila por evento, vencimiento y
modelo) y results/q2_summary.csv (promedios por modelo y plazo).
"""

from __future__ import annotations

import argparse
import sys

import pandas as pd

from crush.analysis.crush import crush_table, summarize_crush
from crush.config import PROCESSED_DIR, RAW_DIR, RESULTS_DIR
from crush.data.source import event_dir, load_event


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="massive")
    ap.add_argument("--r", type=float, default=0.04)
    args = ap.parse_args()

    path = PROCESSED_DIR / "calibration.csv"
    if not path.exists():
        print(f"Falta {path}. Corre primero scripts/03_calibrate.py.")
        return 1
    cal = pd.read_csv(path)
    tables = []
    for _, row in cal.iterrows():
        folder = event_dir(RAW_DIR, args.source, row["ticker"], row["announcement_date"])
        if not (folder / "meta.json").exists():
            print(f"[omitido] {folder.name}: no está descargado")
            continue
        tables.append(crush_table(load_event(folder), row, args.r))
    if not tables:
        return 1
    full = pd.concat(tables, ignore_index=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    full.to_csv(RESULTS_DIR / "q2_crush.csv", index=False)
    summary = summarize_crush(full)
    summary.to_csv(RESULTS_DIR / "q2_summary.csv", index=False)
    print(summary.round(4).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
