"""Monte Carlo de cada evento con sus parámetros calibrados (experimento E7). No descarga nada.

Uso: python scripts/20_monte_carlo.py --paths 100000
Escribe results/mc_events.csv (una fila por evento, modelo y vencimiento) y results/mc_summary.csv.
Lee data/processed/calibration.csv. Ver analysis/montecarlo.py.
"""

from __future__ import annotations

import argparse
import sys

import pandas as pd

from crush.analysis.montecarlo import mc_summary
from crush.config import PROCESSED_DIR, RESULTS_DIR


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--paths", type=int, default=100_000)
    ap.add_argument("--b", type=float, default=0.0, help="costo de acarreo r - q de la simulación (0: los log-retornos del forward)")
    args = ap.parse_args()
    cal = pd.read_csv(PROCESSED_DIR / "calibration.csv")
    rows = [mc_summary(r, tag, i, args.paths, seed=n, b=args.b)
            for n, (_, r) in enumerate(cal.iterrows()) for tag in ("merton", "bates") for i in (1, 2)]
    df = pd.DataFrame(rows)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(RESULTS_DIR / "mc_events.csv", index=False)
    g = df.groupby(["model", "maturity"])
    s = g.agg(n=("z", "size"), mean_rel_diff=("rel_diff", "mean"), median_rel_diff=("rel_diff", "median"),
              share_within_2se=("z", lambda z: float((z.abs() <= 2).mean())), mean_abs_z=("z", lambda z: float(z.abs().mean())),
              martingale_mean=("martingale", "mean"), pay_mean=("pay_mean", "mean"), pay_sd=("pay_sd", "mean"),
              pay_q05=("pay_q05", "mean"), prob_loss=("prob_loss", "mean"), event_share=("event_share", "mean")).reset_index()
    s.to_csv(RESULTS_DIR / "mc_summary.csv", index=False)
    pd.set_option("display.width", 250)
    print(s.round(4).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
