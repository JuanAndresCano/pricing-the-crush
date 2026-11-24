"""Movimiento realizado contra el implícito en cada evento descargado.

Uso:
    python scripts/07_realized.py                       # eventos de data/raw/theta
    python scripts/07_realized.py --root data/smoke     # eventos de prueba (no son del estudio)

Escribe results/realized_vs_implied.csv (una fila por evento) y muestra el resumen. Si existe
data/processed/calibration.csv agrega el eps de cada modelo y R^2 / eps_modelo.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from crush.analysis.realized import add_history, merge_models, realized_row, summarize_realized
from crush.config import PROCESSED_DIR, RAW_DIR, RESULTS_DIR
from crush.data.source import load_event
from crush.market.rates import DEFAULT_RATE


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="theta")
    ap.add_argument("--root", default=str(RAW_DIR), help="carpeta que contiene <fuente>/<TICKER>_<fecha>/")
    ap.add_argument("--r", type=float, default=DEFAULT_RATE, help="tasa de respaldo si el evento no trae una")
    args = ap.parse_args()

    folders = sorted(p for p in (Path(args.root) / args.source).glob("*_*") if (p / "meta.json").exists())
    if not folders:
        print(f"No hay eventos en {Path(args.root) / args.source}. Corre primero scripts/01_download.py.")
        return 1
    rows = []
    for f in folders:
        rows.append(realized_row(load_event(f), args.r))
        print(f"[listo] {f.name}")
    df = add_history(pd.DataFrame(rows))
    cal = PROCESSED_DIR / "calibration.csv"
    if cal.exists():
        df = merge_models(df, pd.read_csv(cal))
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / "realized_vs_implied.csv"
    df.to_csv(out, index=False)

    show = ["ticker", "announcement_date", "R_e", "implied_move_atm", "z_atm", "r2_over_eps_atm", "vrp_atm"]
    pd.set_option("display.width", 200)
    print("\n" + df[show].round(4).to_string(index=False))
    for how in ("atm", "strike"):
        print(f"\nResumen ({how}):", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in summarize_realized(df, how).items()})
    print(f"\n-> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
