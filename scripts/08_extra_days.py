"""Descarga los días posteriores al anuncio (por defecto +1, +2 y +4 días hábiles después del cierre posterior).

Uso (con el Theta Terminal abierto):
    python scripts/08_extra_days.py
    python scripts/08_extra_days.py --root data/smoke --offsets 1 2 4

Es reanudable y las respuestas crudas se guardan en data/raw_api/theta/: lo ya pedido no se vuelve a pedir.
Cuesta unos 3 pedidos por día extra y evento (precio de la acción, tasa y cadena completa).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from crush.config import RAW_DIR
from crush.data.extra_days import DEFAULT_OFFSETS, download_extra_days
from crush.data.theta_client import ThetaError, ThetaSource


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(RAW_DIR), help="carpeta que contiene theta/<TICKER>_<fecha>/")
    ap.add_argument("--offsets", type=int, nargs="+", default=list(DEFAULT_OFFSETS))
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    folders = sorted(p for p in (Path(args.root) / "theta").glob("*_*") if (p / "meta.json").exists())
    if not folders:
        print(f"No hay eventos en {Path(args.root) / 'theta'}. Corre primero scripts/01_download.py.")
        return 1
    src = ThetaSource()
    failed = 0
    for f in folders:
        try:
            extra = download_extra_days(src, f, tuple(args.offsets), force=args.force)
        except ThetaError as exc:
            failed += 1
            print(f"[FALLÓ] {f.name}: {exc}")
            continue
        print(f"[listo] {f.name}: días extra {sorted(int(k) for k in extra)}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
