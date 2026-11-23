"""Prueba de punta a punta con datos reales: descarga un "evento" de ThetaData y lo pasa por la fase de mercado.

El evento puede ser de mentira (cualquier fecha pasada): sirve para ver que el flujo funciona con cotizaciones
reales, no para sacar conclusiones económicas. Los resultados van a data/smoke/, separado de data/raw/.

Uso (con el Theta Terminal abierto):
    python scripts/06_smoke_real.py --ticker AAPL --date 2026-11-18 --timing amc
"""

from __future__ import annotations

import argparse

import pandas as pd

from crush.config import DATA_DIR
from crush.data.source import download_and_cache, load_event
from crush.data.theta_client import ThetaSource
from crush.market.chain import CleanParams, build_slices, clean_quotes
from crush.market.event_mkt import event_row, slice_table


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", default="AAPL")
    ap.add_argument("--date", default="2026-11-18")
    ap.add_argument("--timing", default="amc", choices=["bmo", "amc"])
    ap.add_argument("--r", type=float, default=0.04, help="tasa de respaldo si la fuente no entrega una")
    ap.add_argument("--force", action="store_true", help="rehacer el evento (las respuestas crudas guardadas no se vuelven a pedir)")
    args = ap.parse_args()

    path, status = download_and_cache(ThetaSource(), args.ticker, args.date, args.timing, root=DATA_DIR / "smoke", workers=1, force=args.force)
    data = load_event(path)
    m = data.meta
    print(f"[{status}] {path}")
    print(f"tasas r_pre={m.get('r_pre')} r_post={m.get('r_post')}")
    print(f"contratos {m['n_contracts']}, cotizaciones pre {m['n_quotes_pre']}, post {m['n_quotes_post']}, spot {m['spot_pre']} -> {m['spot_post']}")

    pd.set_option("display.width", 200)
    for label, df in (("pre", data.pre), ("post", data.post)):
        c = clean_quotes(df, CleanParams())
        print(f"\n[{label}] válidas {int(c['valid'].sum())} de {len(c)}; bid cero {int((c['bid'] <= 0).sum())}; "
              f"spread relativo mediano {c.loc[c['valid'], 'rel_spread'].median():.3f}")
    print("\nPor vencimiento:")
    print(slice_table(data, args.r).round(4).to_string(index=False))
    print("\nEvento:")
    print({k: (round(v, 5) if isinstance(v, float) else v) for k, v in event_row(data, args.r).items()})


if __name__ == "__main__":
    main()
