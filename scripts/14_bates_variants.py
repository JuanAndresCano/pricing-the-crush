"""Experimento: ¿la conclusión de la Pregunta 1 cambia si Bates se calibra con θ libre o con κ más amplio?

Uso:
    python scripts/14_bates_variants.py                  # todos los eventos descargados, 4 procesos
    python scripts/14_bates_variants.py --workers 6 --limit 3

No descarga nada: solo lee data/raw/theta y data/processed/calibration.csv (Merton y Bates principal).
Es reanudable: escribe data/processed/bates_variants.csv después de cada evento y salta los ya hechos.
Al terminar escribe results/bates_variants_q1.csv: el contraste de Wilcoxon agrupado por fecha de cada variante
contra Merton y contra Bates principal. Variantes en calibration/variants.py.
"""

from __future__ import annotations

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd

from crush.analysis.compare import clustered_wilcoxon
from crush.calibration.variants import VARIANTS, bates_variant_row
from crush.config import PROCESSED_DIR, RAW_DIR, RESULTS_DIR
from crush.data.source import load_event

OUT = PROCESSED_DIR / "bates_variants.csv"


def _job(args):
    folder, r, variant, starts = args
    row = bates_variant_row(load_event(folder), r, variant, n_starts=starts)
    return folder.name, variant, row


def summarize(variants: pd.DataFrame, cal: pd.DataFrame) -> pd.DataFrame:
    """Compara cada variante con Merton y con Bates principal (error relativo absoluto del strike, vencimientos 1 y 2)."""
    cal = cal.copy()
    cal["announcement_date"] = cal["announcement_date"].astype(str)
    rows = []
    for variant, v in variants.groupby("variant"):
        v = v.copy()
        v["announcement_date"] = v["announcement_date"].astype(str)
        m = v.merge(cal, on=["ticker", "announcement_date"], suffixes=("", "_cal"))
        if len(m) < 6:
            continue
        for label, ref_cols in (("Merton", ("relerr_merton_1", "relerr_merton_2")), ("Bates principal", ("relerr_bates_1", "relerr_bates_2"))):
            d = ((m[ref_cols[0]].abs() - m["relerr_1"].abs()) + (m[ref_cols[1]].abs() - m["relerr_2"].abs())) / 2  # > 0: la variante se acerca más
            w = clustered_wilcoxon(d, m["announcement_date"])
            rows.append({"variant": variant, "reference": label, "n_events": w["n_events"], "n_dates": w["n_dates"],
                         "share_variant_closer": w["share_bates_closer"], "mean_d": w["mean_d"], "pvalue": w["pvalue"],
                         "mae_variant": float(np.mean((m["relerr_1"].abs() + m["relerr_2"].abs()) / 2)),
                         "rmse_iv_variant": float(m["rmse_iv"].mean()), "rmse_iv_bates_principal": float(m["rmse_iv_bates"].mean()),
                         "share_kappa_at_lower_bound": float((m["p_kappa"] <= 0.1001).mean()) if "p_kappa" in m else np.nan})
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="theta")
    ap.add_argument("--r", type=float, default=0.04)
    ap.add_argument("--starts", type=int, default=3)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--variants", default="theta_libre,kappa_ancha", help=f"separadas por coma; opciones: {','.join(VARIANTS)}")
    ap.add_argument("--limit", type=int, default=0, help="solo los primeros N eventos (para probar)")
    args = ap.parse_args()

    cal_path = PROCESSED_DIR / "calibration.csv"
    if not cal_path.exists():
        print("Falta data/processed/calibration.csv. Corre primero scripts/03_calibrate.py.")
        return 1
    cal = pd.read_csv(cal_path)
    wanted = {f"{r.ticker}_{r.announcement_date}" for r in cal.itertuples()}
    folders = sorted(p for p in (RAW_DIR / args.source).glob("*_*") if (p / "meta.json").exists() and p.name in wanted)
    if args.limit:
        folders = folders[: args.limit]
    done = pd.read_csv(OUT) if OUT.exists() else pd.DataFrame()
    rows = done.to_dict("records")
    have = {(f"{r['ticker']}_{r['announcement_date']}", r["variant"]) for r in rows}
    jobs = [(f, args.r, v, args.starts) for f in folders for v in args.variants.split(",") if (f.name, v) not in have]
    print(f"{len(folders)} eventos, {len(jobs)} calibraciones pendientes con {args.workers} procesos")
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for fut in as_completed([ex.submit(_job, j) for j in jobs]):
            name, variant, row = fut.result()
            if row is None:
                print(f"[omitido] {name} {variant}")
                continue
            rows.append(row)
            pd.DataFrame(rows).to_csv(OUT, index=False)
            print(f"[listo] {name} {variant}: RMSE_IV {row['rmse_iv']:.4f}")
    if rows:
        s = summarize(pd.DataFrame(rows), cal)
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        s.to_csv(RESULTS_DIR / "bates_variants_q1.csv", index=False)
        pd.set_option("display.width", 220)
        print(s.round(4).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
