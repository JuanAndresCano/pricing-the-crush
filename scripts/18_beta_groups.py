"""Beta del control por empresa y por sector (experimento E5). Solo lectura sobre results/q2_path.csv.

Uso: python scripts/18_beta_groups.py
Escribe results/beta_by_sector.csv, results/beta_by_ticker.csv y results/beta_groups_tests.csv.
"""

from __future__ import annotations

import sys

import pandas as pd

from crush.analysis.beta_groups import alpha_with_group_beta, beta_by_group
from crush.config import RESULTS_DIR


def main() -> int:
    path = RESULTS_DIR / "q2_path.csv"
    if not path.exists():
        print("Falta results/q2_path.csv.")
        return 1
    p = pd.read_csv(path)
    rows = []
    pd.set_option("display.width", 200)
    for by in ("sector", "ticker"):
        tab, test = beta_by_group(p, by)
        tab.to_csv(RESULTS_DIR / f"beta_by_{by}.csv", index=False)
        rows.append({**test, **alpha_with_group_beta(p, by)})
        print(tab.round(3).to_string(index=False))
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS_DIR / "beta_groups_tests.csv", index=False)
    print(out.round(4).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
