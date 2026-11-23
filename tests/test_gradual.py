"""Contraste final de crush gradual: regresión agrupada por fecha y Wilcoxon por fecha."""

import numpy as np
import pandas as pd
import pytest

from crush.analysis.gradual import cluster_ols, gradual_report


def _table(n_dates=12, per_date=3, alpha=0.0, beta=0.4, noise=0.01, seed=1, ks=(1, 2, 4), models=("merton", "bates")):
    """g_obs = exp(alpha) * g_spy^beta * g_model * ruido, con g_spy variando por fecha (el mercado)."""
    rng = np.random.default_rng(seed)
    rows = []
    for d in range(n_dates):
        market = rng.normal(0.0, 0.15)  # un shock de mercado por fecha: lo comparten todos los eventos de ese día
        for j in range(per_date):
            for k in ks:
                ln_spy = market * (k / 2) + rng.normal(0, 0.02)
                for m in models:
                    ln_model = 0.0 if m == "merton" else -0.01 * k
                    ln_g = alpha + beta * ln_spy + ln_model + rng.normal(0, noise)
                    rows.append({"ticker": f"T{j}", "announcement_date": f"2026-{1 + d // 28:02d}-{1 + d % 28:02d}", "expiry": "2026-12-04",
                                 "k": k, "model": m, "g_obs": np.exp(ln_g), "g_spy": np.exp(ln_spy), "g_model": np.exp(ln_model)})
    return pd.DataFrame(rows)


def test_cluster_ols_recovers_known_line():
    rng = np.random.default_rng(0)
    clusters = np.repeat(np.arange(10), 5)
    x = rng.normal(size=50)
    y = 0.3 - 0.8 * x + rng.normal(0, 0.05, 50)
    r = cluster_ols(x, y, clusters)
    assert r["alpha"] == pytest.approx(0.3, abs=0.05) and r["beta"] == pytest.approx(-0.8, abs=0.05)
    assert r["n_clusters"] == 10 and r["p_beta"] < 0.001


def test_cluster_ols_not_enough_clusters_or_variation():
    assert np.isnan(cluster_ols([1, 2, 3, 4], [1, 2, 3, 4], ["a", "a", "b", "b"])["alpha"])  # 2 conglomerados
    assert np.isnan(cluster_ols([1, 1, 1, 1, 1, 1], [1, 2, 3, 4, 5, 6], list("aabbcc"))["alpha"])  # x constante


def test_cluster_se_larger_than_naive_when_errors_share_a_shock():
    rng = np.random.default_rng(3)
    G, m = 8, 10
    clusters = np.repeat(np.arange(G), m)
    shock = np.repeat(rng.normal(0, 0.1, G), m)
    x = rng.normal(size=G * m)
    y = shock + rng.normal(0, 0.01, G * m)  # el shock común hace que la constante sea incierta
    r = cluster_ols(x, y, clusters)
    naive_se = np.std(y, ddof=1) / np.sqrt(len(y))
    assert r["se_alpha"] > 2 * naive_se


def test_no_gradual_effect_is_not_significant_and_beta_is_recovered():
    reg, tests = gradual_report(_table(alpha=0.0, beta=0.4))
    m = reg[reg["model"] == "merton"].iloc[0]
    assert m["beta"] == pytest.approx(0.4, abs=0.05) and abs(m["alpha"]) < 0.01 and m["p_alpha"] > 0.05
    t = tests[(tests["model"] == "merton") & (tests["horizon"] == "all")].iloc[0]
    assert t["pvalue_wilcoxon"] > 0.05 and t["n_dates"] == 12


def test_gradual_effect_is_detected():
    reg, tests = gradual_report(_table(alpha=-0.03, beta=0.4, noise=0.01))
    m = reg[reg["model"] == "merton"].iloc[0]
    assert m["alpha"] == pytest.approx(-0.03, abs=0.01) and m["p_alpha"] < 0.05
    t = tests[(tests["model"] == "merton") & (tests["horizon"] == "all")].iloc[0]
    assert t["pvalue_wilcoxon"] < 0.05 and t["mean_e"] < 0 and t["share_negative"] > 0.9
    assert set(tests["horizon"]) == {"all", 1, 2, 4}


def test_model_that_already_expects_the_decline_leaves_no_residual():
    """Si la caída extra es la que el modelo predice (g_model < 1), el residuo del modelo es cero."""
    table = _table(alpha=0.0, beta=0.4, models=("bates",))
    reg, tests = gradual_report(table)
    assert abs(reg.iloc[0]["alpha"]) < 0.01
    assert tests[tests["horizon"] == "all"].iloc[0]["pvalue_wilcoxon"] > 0.05


def test_few_dates_give_no_pvalue():
    reg, tests = gradual_report(_table(n_dates=4, per_date=5, alpha=-0.05))
    assert np.isfinite(reg[reg["model"] == "merton"].iloc[0]["se_alpha"])  # 4 fechas alcanzan para la regresión
    assert tests[tests["model"] == "merton"]["pvalue_wilcoxon"].isna().all()  # pero no para Wilcoxon


def test_missing_control_gives_nan_regression_and_no_crash():
    t = _table(n_dates=6)
    t["g_spy"] = np.nan
    reg, tests = gradual_report(t)
    assert reg["alpha"].isna().all()
    assert tests["n_obs"].eq(0).all() and tests["pvalue_wilcoxon"].isna().all()
