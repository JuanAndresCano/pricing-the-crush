"""Robustez de las Preguntas 1 y 2 por subconjuntos de eventos y con beta de validación cruzada.

Uso:
    python scripts/13_robustness.py

Lee data/processed/calibration.csv, results/q2_path.csv y events/earnings_events.csv. Escribe
results/robustness_q1.csv y results/robustness_q2.csv. Es solo lectura sobre los demás resultados: se puede
correr cuantas veces se quiera sin tocar nada del pipeline. Ver analysis/robustness.py.
"""

from __future__ import annotations

import sys

import pandas as pd

from crush.analysis.robustness import q1_robustness, q2_robustness
from crush.config import PROCESSED_DIR, RESULTS_DIR, ROOT


def main() -> int:
    cal_path, path_path = PROCESSED_DIR / "calibration.csv", RESULTS_DIR / "q2_path.csv"
    if not cal_path.exists() or not path_path.exists():
        print("Faltan data/processed/calibration.csv o results/q2_path.csv. Corre antes el pipeline (scripts/03 a 09).")
        return 1
    events = pd.read_csv(ROOT / "events" / "earnings_events.csv")
    events["announcement_date"] = events["announcement_date"].astype(str)
    q1 = q1_robustness(pd.read_csv(cal_path), events)
    q2 = q2_robustness(pd.read_csv(path_path), events)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    q1.to_csv(RESULTS_DIR / "robustness_q1.csv", index=False)
    q2.to_csv(RESULTS_DIR / "robustness_q2.csv", index=False)
    pd.set_option("display.width", 220)
    print("Pregunta 1 (promedio de los dos vencimientos, un contraste por escenario)")
    print(q1[q1["test"] == "promedio de vencimientos"].drop(columns=["test", "mae_merton", "mae_bates"]).round(4).to_string(index=False))
    print("\nPregunta 2 (caída extra alpha; *_cv usa el beta estimado sin la fecha evaluada)")
    print(q2.round(4).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
