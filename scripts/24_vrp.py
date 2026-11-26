"""Prima de riesgo de varianza hasta el vencimiento, ecuación (56) (experimento E11).

Uso: python scripts/24_vrp.py            (con el Theta Terminal abierto la primera vez)
Pide un rango de cierres diarios por empresa y año (queda en el caché de data/raw_api, así que repetir no gasta
pedidos). Escribe results/vrp_events.csv y results/vrp_summary.csv. Los vencimientos que todavía no han pasado se omiten.
"""

from __future__ import annotations

import argparse
import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from crush.analysis.vrp import closes_from_eod, realized_to_expiry, summarize_vrp
from crush.config import PROCESSED_DIR, RAW_DIR, RESULTS_DIR
from crush.data.source import event_dir, load_event
from crush.data.theta_client import ThetaClient
from crush.market.chain import build_slices
from crush.market.rates import DEFAULT_RATE, event_rates


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--end", default=str(date.today() - timedelta(days=1)), help="último cierre disponible (AAAA-MM-DD)")
    args = ap.parse_args()
    cal = pd.read_csv(PROCESSED_DIR / "calibration.csv")
    client = ThetaClient()
    closes: dict[str, pd.Series] = {}
    first_year = int(min(cal["announcement_date"].astype(str))[:4])
    for ticker in sorted(cal["ticker"].unique()):
        parts = []
        for year in range(first_year, int(args.end[:4]) + 1):
            params = {"symbol": ticker, "start_date": f"{year}0101", "end_date": min(f"{year}1231", args.end.replace("-", ""))}
            parts.append(closes_from_eod(client.get_csv("stock/history/eod", params)))
        closes[ticker] = pd.concat(parts).sort_index()
        print(f"[cierres] {ticker}: {len(closes[ticker])} días", flush=True)

    rows, skipped = [], []
    for _, row in cal.iterrows():
        folder = event_dir(Path(RAW_DIR), "theta", row["ticker"], row["announcement_date"])
        if not (folder / "meta.json").exists():
            continue
        data = load_event(folder)
        m = data.meta
        slices = build_slices(data.pre, m["pre_day"], m["spot_pre"], event_rates(m, DEFAULT_RATE)[0])[:2]
        for i, s in enumerate(slices, start=1):
            rv = realized_to_expiry(closes[row["ticker"]], m["pre_day"], s.expiry)
            if rv is None:
                skipped.append((folder.name, i, str(s.expiry)))
                continue
            first = closes[row["ticker"]].loc[str(pd.Timestamp(m["pre_day"]).date())]
            k = float(row[f"k_mkt_{i}"])
            rows.append({"ticker": row["ticker"], "announcement_date": row["announcement_date"], "maturity": i, "expiry": str(pd.Timestamp(s.expiry).date()),
                         "k_mkt": k, **rv, "vrp": k - rv["rv"], "spot_check": float(first / m["spot_pre"] - 1.0)})
    t = pd.DataFrame(rows)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    t.to_csv(RESULTS_DIR / "vrp_events.csv", index=False)
    s = summarize_vrp(t)
    s.to_csv(RESULTS_DIR / "vrp_summary.csv", index=False)
    pd.set_option("display.width", 220)
    print(f"{len(t)} ventanas; omitidas {len(skipped)}: {skipped}")
    print(f"splits sospechosos: {int(t['split_suspect'].sum())}; diferencia máxima con el precio previo guardado: {t['spot_check'].abs().max():.5f}")
    print(s.round(4).T.to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
