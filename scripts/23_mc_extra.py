"""Monte Carlo: valor del swap tras el anuncio, vol crush con c, colas del pago y variable de control (experimento E10).

Uso: python scripts/23_mc_extra.py --paths 100000
Lee data/processed/calibration.csv y results/c_table.csv (c del día 0). Escribe results/mc_extra_events.csv y
results/mc_extra_summary.csv. No descarga nada. Ver analysis/montecarlo_extra.py.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd

from crush.analysis.montecarlo_extra import mc_extra
from crush.config import PROCESSED_DIR, RESULTS_DIR


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--paths", type=int, default=100_000)
    args = ap.parse_args()
    cal = pd.read_csv(PROCESSED_DIR / "calibration.csv")
    ct = pd.read_csv(RESULTS_DIR / "c_table.csv")
    c0 = ct[ct["k"] == 0].groupby(["ticker", "announcement_date", "model"])["c_raw"].median()
    rows = []
    for n, (_, r) in enumerate(cal.iterrows()):
        for tag in ("merton", "bates"):
            c = float(c0.get((r["ticker"], r["announcement_date"], tag), np.nan))
            rows += [mc_extra(r, tag, i, c, args.paths, seed=n) for i in (1, 2)]
    df = pd.DataFrame(rows)
    df["se_ratio_cv"] = df["se_cv"] / df["se"]
    df.to_csv(RESULTS_DIR / "mc_extra_events.csv", index=False)
    cols = [c for c in df.columns if c not in ("ticker", "announcement_date", "model", "maturity")]
    s = df.groupby(["model", "maturity"])[cols].mean().reset_index()
    s.to_csv(RESULTS_DIR / "mc_extra_summary.csv", index=False)
    pd.set_option("display.width", 200)
    print(s.round(4).T.to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
