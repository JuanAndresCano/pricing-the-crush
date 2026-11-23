"""Compara Merton-evento y Bates-evento contra el strike del mercado (Pregunta 1).

Uso:
    python scripts/04_compare.py

Lee data/processed/calibration.csv y escribe en results/: q1_summary.csv (Wilcoxon agrupado por fecha) y
q1_decomposition.csv (descomposición (45) por evento y vencimiento).
"""

from __future__ import annotations

import sys

import pandas as pd

from crush.analysis.compare import decompose, summarize_q1
from crush.config import PROCESSED_DIR, RESULTS_DIR


def main() -> int:
    path = PROCESSED_DIR / "calibration.csv"
    if not path.exists():
        print(f"Falta {path}. Corre primero scripts/03_calibrate.py.")
        return 1
    df = pd.read_csv(path)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    s = summarize_q1(df)
    s.to_csv(RESULTS_DIR / "q1_summary.csv")
    decompose(df).to_csv(RESULTS_DIR / "q1_decomposition.csv", index=False)
    print(s.round(4).to_string())
    print(
        f"\nBates gana por BIC en {s.attrs['bic_bates_wins']:.0%} de los eventos; "
        f"RMSE_IV medio: Merton {s.attrs['rmse_iv_merton']:.4f}, Bates {s.attrs['rmse_iv_bates']:.4f}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
