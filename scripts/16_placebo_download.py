"""Descarga la ventana placebo (sin anuncio) de cada evento y los días de SPY que necesita. Experimento E4.

Uso (con el Theta Terminal abierto):
    python scripts/16_placebo_download.py
    python scripts/16_placebo_download.py --limit 2        # prueba con dos eventos

Para cada evento en data/raw/theta toma un día 0 de placebo 20 días hábiles después del día 0 real y descarga ese día y
los +1, +2 y +4 (ver data/placebo_days.py), y las cadenas de SPY de esos mismos días (data/control/theta/SPY).
Es reanudable. Cuesta unos 3 pedidos por día y unos 70 minutos en total con el límite del plan gratis.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from crush.analysis.placebo import PLACEBO_OFFSET
from crush.config import CONTROL_DIR, PLACEBO_DIR, RAW_DIR
from crush.data.control import ensure_control_day
from crush.data.placebo_days import download_placebo, placebo_days
from crush.data.theta_client import ThetaError, ThetaSource


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--offset", type=int, default=PLACEBO_OFFSET)
    ap.add_argument("--control", default="SPY")
    ap.add_argument("--strike-range", type=int, default=40)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    folders = sorted(p for p in (RAW_DIR / "theta").glob("*_*") if (p / "meta.json").exists())
    if args.limit:
        folders = folders[: args.limit]
    stock, ctrl = ThetaSource(), ThetaSource(strike_range=args.strike_range)
    for f in folders:
        meta = json.loads((f / "meta.json").read_text(encoding="utf-8"))
        try:
            pm = download_placebo(stock, meta, PLACEBO_DIR, args.offset)
            days = placebo_days(pm)
            missing = [d for d in days if ensure_control_day(ctrl, args.control, d, CONTROL_DIR) is None]
        except ThetaError as exc:
            print(f"[FALLÓ] {f.name}: {exc}")
            continue
        print(f"[listo] {f.name}: placebo día 0 {pm.get('day0')}, {len(days)} días" + (f"; sin {args.control}: {missing}" if missing else ""), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
