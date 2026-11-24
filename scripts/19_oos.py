"""Validación fuera de muestra (experimento E6): calibra con strikes alternos y evalúa con los demás.

Uso: python scripts/19_oos.py --workers 5
Reanudable. Escribe data/processed/oos.csv y results/oos_summary.csv. No descarga nada. Ver analysis/oos.py.
"""

from __future__ import annotations

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd

from crush.analysis.compare import clustered_wilcoxon
from crush.analysis.oos import oos_row
from crush.config import PROCESSED_DIR, RAW_DIR, RESULTS_DIR
from crush.data.source import load_event

OUT = PROCESSED_DIR / "oos.csv"


def _job(args):
    folder, r, starts = args
    return folder.name, oos_row(load_event(folder), r, n_starts=starts)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["announcement_date"] = d["announcement_date"].astype(str)
    rows = []
    d_rmse = d["rmse_test_merton"] - d["rmse_test_bates"]  # > 0: Bates ajusta mejor fuera de muestra
    w = clustered_wilcoxon(d_rmse, d["announcement_date"])
    rows.append({"measure": "RMSE_IV fuera de muestra (Merton - Bates)", **{k: w[k] for k in ("n_events", "n_dates", "share_bates_closer", "mean_d", "pvalue")},
                 "mean_merton": float(d["rmse_test_merton"].mean()), "mean_bates": float(d["rmse_test_bates"].mean())})
    for i in (1, 2):
        dd = d[f"relerr_merton_{i}"].abs() - d[f"relerr_bates_{i}"].abs()
        w = clustered_wilcoxon(dd, d["announcement_date"])
        rows.append({"measure": f"error del strike fuera de muestra, vencimiento {i} (Merton - Bates)",
                     **{k: w[k] for k in ("n_events", "n_dates", "share_bates_closer", "mean_d", "pvalue")},
                     "mean_merton": float(d[f"relerr_merton_{i}"].abs().mean()), "mean_bates": float(d[f"relerr_bates_{i}"].abs().mean())})
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="theta")
    ap.add_argument("--r", type=float, default=0.04)
    ap.add_argument("--starts", type=int, default=2)
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    cal = pd.read_csv(PROCESSED_DIR / "calibration.csv")
    wanted = {f"{r.ticker}_{r.announcement_date}" for r in cal.itertuples()}
    folders = sorted(p for p in (RAW_DIR / args.source).glob("*_*") if (p / "meta.json").exists() and p.name in wanted)
    if args.limit:
        folders = folders[: args.limit]
    rows = pd.read_csv(OUT).to_dict("records") if OUT.exists() else []
    have = {f"{r['ticker']}_{r['announcement_date']}" for r in rows}
    jobs = [(f, args.r, args.starts) for f in folders if f.name not in have]
    print(f"{len(folders)} eventos, {len(jobs)} pendientes", flush=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for fut in as_completed([ex.submit(_job, j) for j in jobs]):
            name, row = fut.result()
            if row is None:
                print(f"[omitido] {name}", flush=True)
                continue
            rows.append(row)
            pd.DataFrame(rows).to_csv(OUT, index=False)
            print(f"[listo] {name}: RMSE fuera de muestra Merton {row['rmse_test_merton']:.4f}, Bates {row['rmse_test_bates']:.4f}", flush=True)
    s = summarize(pd.DataFrame(rows))
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    s.to_csv(RESULTS_DIR / "oos_summary.csv", index=False)
    pd.set_option("display.width", 220)
    print(s.round(4).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
