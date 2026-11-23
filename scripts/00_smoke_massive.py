"""Prueba de acceso a Massive: ¿el plan trae contratos y cotizaciones históricas con bid y ask?

Uso:
    python scripts/00_smoke_massive.py --underlying AAPL --date 2026-10-29

No imprime la clave. Si algún endpoint falla, muestra el mensaje del servidor.
"""

from __future__ import annotations

import argparse
from datetime import date, timedelta

from crush.data.massive_client import MassiveClient, MassiveError


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--underlying", default="AAPL")
    ap.add_argument("--date", default="2026-10-29", help="día a consultar, YYYY-MM-DD (pasado)")
    ap.add_argument("--max-days", type=int, default=21, help="vencimientos hasta N días después de la fecha")
    args = ap.parse_args()

    client = MassiveClient()
    d0 = date.fromisoformat(args.date)
    d1 = d0 + timedelta(days=args.max_days)

    print(f"[1] Contratos de {args.underlying} vigentes el {d0}, con vencimiento hasta {d1}")
    try:
        contracts = client.option_contracts(
            args.underlying,
            as_of=d0.isoformat(),
            **{"expiration_date.gte": d0.isoformat(), "expiration_date.lte": d1.isoformat()},
        )
    except MassiveError as exc:
        print("    FALLÓ:", exc)
        return
    expiries = sorted({c["expiration_date"] for c in contracts})
    print(f"    {len(contracts)} contratos, vencimientos: {expiries}")
    if not contracts:
        return

    sample = contracts[len(contracts) // 2]
    print(f"[2] Cotizaciones del {d0} para {sample['ticker']} (strike {sample['strike_price']}, {sample['contract_type']})")
    try:
        quotes = client.option_quotes(sample["ticker"], d0.isoformat(), limit=1000)
    except MassiveError as exc:
        print("    FALLÓ:", exc)
        return
    print(f"    {len(quotes)} cotizaciones")
    if quotes:
        last = quotes[-1]
        print(f"    última: bid={last.get('bid_price')} ask={last.get('ask_price')} ts={last.get('sip_timestamp')}")
        print("    OK: el plan entrega cotizaciones históricas con bid y ask.")
    else:
        print("    Sin cotizaciones ese día para ese contrato (puede ser ilíquido). Prueba otro contrato u otro día.")


if __name__ == "__main__":
    main()
