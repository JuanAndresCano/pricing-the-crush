"""Descarga las cadenas de opciones de cada anuncio del calendario y las guarda en data/raw/.

Uso:
    python scripts/01_download.py --events events/earnings_events.csv --limit 1
    python scripts/01_download.py                      # todos los eventos; los ya descargados se saltan

Es reanudable: si se corta, se vuelve a correr y solo baja lo que falta. No imprime la clave de la API.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from crush.config import RAW_DIR, ROOT
from crush.data.calendar import load_events
from crush.data.massive_client import MassiveError
from crush.data.theta_client import ThetaError
from crush.data.selection import SelectionParams
from crush.data.source import DataError, MassiveSource, download_and_cache


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", default=str(ROOT / "events" / "earnings_events.csv"))
    ap.add_argument("--source", default="massive", choices=["massive", "theta"])
    ap.add_argument("--limit", type=int, default=None, help="solo los primeros N eventos")
    ap.add_argument("--force", action="store_true", help="volver a descargar aunque ya exista")
    ap.add_argument("--include-unverified", action="store_true", help="incluir filas con verified distinto de true")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--n-near", type=int, default=2)
    ap.add_argument("--far-min-days", type=int, default=21)
    args = ap.parse_args()

    events = load_events(args.events)
    if not args.include_unverified:
        ok_rows = events["verified"].astype(str).str.strip().str.lower().isin(["true", "1", "yes", "si", "sí"])
        if (~ok_rows).any():
            print(f"Se omiten {int((~ok_rows).sum())} filas con verified distinto de true (usa --include-unverified para incluirlas).")
        events = events[ok_rows]
    if events.empty:
        print(f"{args.events} no tiene eventos verificados. Completa filas (ticker, announcement_date, timing, verified=true) y vuelve a correr.")
        return 1
    if args.limit:
        events = events.head(args.limit)

    if args.source == "theta":
        from crush.data.theta_client import ThetaSource

        src = ThetaSource()
        args.workers = 1  # el plan gratis limita el ritmo; ThetaSource pide una cadena por día y empresa
    else:
        src = MassiveSource()
    params = SelectionParams(n_near=args.n_near, far_min_days=args.far_min_days)
    ok = cached = 0
    failed: list[str] = []
    for _, ev in events.iterrows():
        tag = f"{ev['ticker']} {ev['announcement_date']} ({ev['timing']})"
        try:
            path, status = download_and_cache(
                src, ev["ticker"], ev["announcement_date"], ev["timing"],
                root=RAW_DIR, force=args.force, params=params, workers=args.workers,
            )
        except (DataError, MassiveError, ThetaError) as exc:
            failed.append(tag)
            print(f"[FALLÓ]  {tag}: {exc}")
            continue
        if status == "cached":
            cached += 1
            print(f"[caché]  {tag}")
        else:
            ok += 1
            n = json.loads((Path(path) / "meta.json").read_text(encoding="utf-8"))
            print(f"[listo]  {tag}: {n['n_contracts']} contratos, cotizaciones {n['n_quotes_pre']} antes y {n['n_quotes_post']} después")
    print(f"\nDescargados: {ok}, en caché: {cached}, fallidos: {len(failed)}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
