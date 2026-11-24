"""Contraste final de la Pregunta 2: ¿el crush sigue después del día 0, descontado el mercado y el modelo?

Uso:
    python scripts/11_gradual_test.py

Lee results/q2_path.csv (salida de scripts/09_crush_path.py, con el control SPY) y escribe
results/q2_gradual_regression.csv (alpha y beta por modelo, errores agrupados por fecha) y
results/q2_gradual_tests.csv (Wilcoxon por fecha, por modelo y horizonte). Ver analysis/gradual.py.

Lectura: alpha < 0 (o e < 0) significa que la volatilidad sigue bajando más de lo que explican el mercado y el modelo.
"""

from __future__ import annotations

import sys

import pandas as pd

from crush.analysis.gradual import gradual_report
from crush.config import RESULTS_DIR


def main() -> int:
    path = RESULTS_DIR / "q2_path.csv"
    if not path.exists():
        print(f"Falta {path}. Corre primero scripts/09_crush_path.py.")
        return 1
    table = pd.read_csv(path)
    if table["g_spy"].isna().all():
        print("La tabla no tiene control (g_spy vacío). Descarga SPY con scripts/10_control.py y repite scripts/09_crush_path.py.")
        return 1
    reg, tests = gradual_report(table)
    reg.to_csv(RESULTS_DIR / "q2_gradual_regression.csv", index=False)
    tests.to_csv(RESULTS_DIR / "q2_gradual_tests.csv", index=False)
    pd.set_option("display.width", 200)
    print("Regresión  ln g_obs - ln g_modelo = alpha + beta ln g_spy  (errores agrupados por fecha)")
    print(reg.round(4).to_string(index=False))
    print("\nWilcoxon por fecha sobre e = ln g_obs - beta ln g_spy - ln g_modelo")
    print(tests.round(4).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
