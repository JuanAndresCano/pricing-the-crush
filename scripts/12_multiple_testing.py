"""Corrección por comparaciones múltiples de todos los contrastes del proyecto.

Uso:  python scripts/12_multiple_testing.py

Lee results/q1_summary.csv, q2_gradual_regression.csv, q2_gradual_tests.csv y realized_vs_implied.csv, y escribe
results/multiple_testing.csv con el p-valor original y el ajustado (Holm y Benjamini-Hochberg) dentro de cada
familia (Q1, Q2, realizado) y sobre todas las pruebas juntas.
"""

from __future__ import annotations

import sys

import pandas as pd

from crush.analysis.multiple import adjust_table
from crush.analysis.realized import summarize_realized
from crush.config import RESULTS_DIR


def main() -> int:
    rows = []
    q1 = pd.read_csv(RESULTS_DIR / "q1_summary.csv")
    for _, r in q1.iterrows():
        rows.append({"family": "Q1", "test": f"Bates vs Merton, vencimiento {int(r['maturity'])}", "pvalue": r["pvalue"]})
    reg = pd.read_csv(RESULTS_DIR / "q2_gradual_regression.csv")
    for _, r in reg.iterrows():
        rows.append({"family": "Q2", "test": f"{r['model']}: regresión, caída extra (alpha)", "pvalue": r["p_alpha"]})
    tests = pd.read_csv(RESULTS_DIR / "q2_gradual_tests.csv")
    for _, r in tests.iterrows():
        rows.append({"family": "Q2", "test": f"{r['model']}: Wilcoxon, horizonte {r['horizon']}", "pvalue": r["pvalue_wilcoxon"]})
    realized = summarize_realized(pd.read_csv(RESULTS_DIR / "realized_vs_implied.csv"), "atm")
    rows.append({"family": "Realizado", "test": "movimiento realizado vs implícito (ATM)", "pvalue": realized["pvalue_wilcoxon"]})
    out = adjust_table(pd.DataFrame(rows))
    out.to_csv(RESULTS_DIR / "multiple_testing.csv", index=False)
    show = out.copy()
    show["test"] = show["test"].str.slice(0, 48)
    print(show.round(4).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
