"""Descarga la cadena de SPY (activo de control) de cada día que usa un evento, para descontar el mercado general.

Uso (con el Theta Terminal abierto):
    python scripts/10_control.py
    python scripts/10_control.py --root data/smoke --symbol SPY

Cada día se guarda una sola vez en data/control/theta/SPY/ y se comparte entre eventos. Cuesta unos 3 pedidos por día
nuevo (precio, tasa y cadena cercana al dinero). Es reanudable. Corre después de scripts/08_extra_days.py.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from crush.config import CONTROL_DIR, RAW_DIR
from crush.data.control import download_control
from crush.data.theta_client import ThetaError, ThetaSource


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(RAW_DIR), help="carpeta que contiene theta/<TICKER>_<fecha>/")
    ap.add_argument("--symbol", default="SPY")
    ap.add_argument("--strike-range", type=int, default=15, help="strikes a cada lado del precio (SPY tiene paso de 1 dólar)")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    folders = sorted(p for p in (Path(args.root) / "theta").glob("*_*") if (p / "meta.json").exists())
    if not folders:
        print(f"No hay eventos en {Path(args.root) / 'theta'}. Corre primero scripts/01_download.py.")
        return 1
    src = ThetaSource(strike_range=args.strike_range)
    failed = 0
    for f in folders:
        meta = json.loads((f / "meta.json").read_text(encoding="utf-8"))
        try:
            ok = download_control(src, meta, args.symbol, CONTROL_DIR, force=args.force)
        except ThetaError as exc:
            failed += 1
            print(f"[FALLÓ] {f.name}: {exc}")
            continue
        missing = [d for d, v in ok.items() if not v]
        print(f"[listo] {f.name}: {len(ok) - len(missing)} días de {args.symbol}" + (f"; sin datos: {missing}" if missing else ""))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
