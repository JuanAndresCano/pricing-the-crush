"""Pregunta 1 con la misma receta en mercado y modelo, con y sin extrapolación (experimento E9). No descarga nada.

Uso: python scripts/22_same_recipe.py
Escribe results/same_recipe_events.csv, results/same_recipe_summary.csv y results/strike_range_effect.csv.
Ver analysis/same_recipe.py.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

from crush.analysis.same_recipe import range_effect, same_recipe_rows, summarize_same_recipe
from crush.config import PROCESSED_DIR, RAW_DIR, RESULTS_DIR
from crush.data.source import event_dir, load_event
from crush.market.rates import DEFAULT_RATE


def main() -> int:
    cal = pd.read_csv(PROCESSED_DIR / "calibration.csv")
    rows = []
    for _, row in cal.iterrows():
        folder = event_dir(Path(RAW_DIR), "theta", row["ticker"], row["announcement_date"])
        if not (folder / "meta.json").exists():
            print(f"[omitido] {folder.name}", flush=True)
            continue
        rows += same_recipe_rows(load_event(folder), row, DEFAULT_RATE)
        print(f"[listo] {folder.name}", flush=True)
    table = pd.DataFrame(rows)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(RESULTS_DIR / "same_recipe_events.csv", index=False)
    s = summarize_same_recipe(table)
    s.to_csv(RESULTS_DIR / "same_recipe_summary.csv", index=False)
    eff = range_effect(table)
    eff.to_csv(RESULTS_DIR / "strike_range_effect.csv", index=False)
    pd.set_option("display.width", 250)
    print(s[["comparison", "recipe", "maturity", "n_events", "share_bates_closer", "pvalue", "mean_abs_rel_merton", "mean_abs_rel_bates"]].round(4).to_string(index=False))
    print(eff.round(4).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
