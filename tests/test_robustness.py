"""Pruebas de robustez: escenarios y beta con validación cruzada por fecha."""

import numpy as np
import pandas as pd
import pytest

from crush.analysis.gradual import _prepare
from crush.analysis.robustness import TARIFF_WINDOW, crossfit_alpha, q1_robustness, q2_robustness, scenarios


def _events(n=8):
    d = ["2025-02-01", "2025-04-15", "2025-07-01", "2025-10-01", "2026-01-10", "2026-04-10", "2026-07-10", "2026-10-10"][:n]
    return pd.DataFrame({"ticker": ["A", "B"] * (n // 2), "announcement_date": d, "timing": ["amc", "bmo"] * (n // 2)})


def test_scenarios_masks():
    ev = _events()
    s = scenarios(ev)
    assert s["2025"].sum() == 4 and s["2026"].sum() == 4
    assert not s["sin_choque_abril_2025"][1] and s["sin_choque_abril_2025"].sum() == 7  # solo cae 2025-04-15
    assert (s["sin_A"] == (ev["ticker"] != "A")).all() and TARIFF_WINDOW[0] < TARIFF_WINDOW[1]


def _path(n_dates=12, alpha=-0.03, beta=0.4, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for d in range(n_dates):
        mkt = rng.normal(0, 0.15)
        for k in (1, 2, 4):
            ln_spy = mkt * k / 2 + rng.normal(0, 0.02)
            for m in ("merton", "bates"):
                ln_g = alpha + beta * ln_spy + rng.normal(0, 0.01)
                rows.append({"ticker": "A" if d % 2 else "B", "announcement_date": f"2026-{1 + d // 28:02d}-{1 + d % 28:02d}", "expiry": "x", "k": k,
                             "model": m, "g_obs": np.exp(ln_g), "g_spy": np.exp(ln_spy), "g_model": 1.0})
    return pd.DataFrame(rows)


def test_crossfit_recovers_alpha_without_much_bias():
    t = _prepare(_path(), "merton")
    r = crossfit_alpha(t)
    assert r["alpha_cv"] == pytest.approx(-0.03, abs=0.01) and r["p_t_cv"] < 0.01 and r["n_dates"] == 12


def test_q2_robustness_runs_per_scenario_and_model():
    p = _path()
    ev = p[["ticker", "announcement_date"]].drop_duplicates().assign(timing="amc")
    out = q2_robustness(p, ev)
    assert {"todos", "sin_A", "sin_B"} <= set(out["scenario"]) and set(out["model"]) == {"merton", "bates"}
    assert (out[out["scenario"] == "todos"]["alpha"] < 0).all()


def test_q1_robustness_basic():
    rng = np.random.default_rng(1)
    n = 14
    cal = pd.DataFrame({"ticker": ["A", "B"] * (n // 2), "announcement_date": [f"2026-01-{i + 1:02d}" for i in range(n)]})
    for i in (1, 2):
        cal[f"relerr_merton_{i}"] = rng.normal(0.05, 0.01, n)
        cal[f"relerr_bates_{i}"] = rng.normal(0.02, 0.01, n)
        cal[f"k_mkt_{i}"] = 1.0
    cal["bic_merton"], cal["bic_bates"], cal["rmse_iv_merton"], cal["rmse_iv_bates"] = 1.0, 0.0, 0.02, 0.01
    ev = cal[["ticker", "announcement_date"]].assign(timing="amc")
    out = q1_robustness(cal, ev)
    r = out[(out["scenario"] == "todos") & (out["test"] == "promedio de vencimientos")].iloc[0]
    assert r["share_bates_closer"] > 0.9 and r["pvalue"] < 0.05
