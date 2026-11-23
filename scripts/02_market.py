"""Calcula lo que dice el mercado de cada evento descargado: forward, volatilidad ATM, strike, eps, sigma_b y CR.

Uso:
    python scripts/02_market.py --r 0.04

Lee data/raw/<fuente>/<TICKER>_<fecha>/ y escribe data/processed/market_slices.csv (un renglón por día y
vencimiento) y data/processed/market_events.csv (un renglón por evento).
"""

from __future__ import annotations

import argparse
import sys

import pandas as pd

from crush.config import PROCESSED_DIR, RAW_DIR
from crush.data.source import load_event
from crush.market.event_mkt import event_row, slice_table


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="theta")
    ap.add_argument("--r", type=float, default=0.04, help="tasa libre de riesgo anual, constante")
    args = ap.parse_args()

    folders = sorted(p for p in (RAW_DIR / args.source).glob("*_*") if (p / "meta.json").exists())
    if not folders:
        print(f"No hay eventos en {RAW_DIR / args.source}. Corre primero scripts/01_download.py.")
        return 1
    slices, rows = [], []
    for f in folders:
        data = load_event(f)
        slices.append(slice_table(data, args.r))
        rows.append(event_row(data, args.r))
        print(f"[listo] {f.name}")
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    pd.concat(slices, ignore_index=True).to_csv(PROCESSED_DIR / "market_slices.csv", index=False)
    pd.DataFrame(rows).to_csv(PROCESSED_DIR / "market_events.csv", index=False)
    print(f"{len(rows)} eventos -> {PROCESSED_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
