"""Calibra Merton-evento y Bates-evento a cada evento descargado y compara sus strikes con el del mercado.

Uso:
    python scripts/03_calibrate.py --r 0.04 --starts 3

Escribe data/processed/calibration.csv (una fila por evento). Es lento: unos minutos por evento y modelo.
"""

from __future__ import annotations

import argparse
import sys

import pandas as pd

from crush.calibration.real import calibrate_event, calibration_row
from crush.config import PROCESSED_DIR, RAW_DIR
from crush.data.source import load_event


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="massive")
    ap.add_argument("--r", type=float, default=0.04)
    ap.add_argument("--starts", type=int, default=3)
    args = ap.parse_args()

    folders = sorted(p for p in (RAW_DIR / args.source).glob("*_*") if (p / "meta.json").exists())
    if not folders:
        print(f"No hay eventos en {RAW_DIR / args.source}. Corre primero scripts/01_download.py.")
        return 1
    out = PROCESSED_DIR / "calibration.csv"
    rows = []
    for f in folders:
        data = load_event(f)
        cal = calibrate_event(data, args.r, n_starts=args.starts)
        if cal is None:
            print(f"[omitido] {f.name}: menos de 2 vencimientos utilizables")
            continue
        row = calibration_row(data, cal)
        rows.append(row)
        print(f"[listo] {f.name}: RMSE_IV Merton {row['rmse_iv_merton']:.4f}, Bates {row['rmse_iv_bates']:.4f}")
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(out, index=False)  # se guarda tras cada evento
    print(f"{len(rows)} eventos -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
