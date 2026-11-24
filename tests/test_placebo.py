import numpy as np
import pandas as pd

from crush.analysis.placebo import cluster_multi, paired_by_event, real_table, stacked_effect
from crush.data.placebo_days import placebo_day0


def _make(n_dates=30, extra=-0.04, seed=0, beta=0.4):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_dates):
        date = f"2025-{1 + i // 28:02d}-{1 + i % 28:02d}"
        market = rng.normal(0, 0.03)
        for k in (1, 2, 4):
            rows.append({"ticker": "AAA", "announcement_date": date, "expiry": "x", "k": k, "days_left": 8 - k,
                         "g_obs": np.exp(beta * market + extra + rng.normal(0, 0.01)), "g_spy": np.exp(market)})
    return pd.DataFrame(rows)


def test_stacked_effect_recovers_extra_drop():
    real, placebo = _make(extra=-0.05, seed=1), _make(extra=-0.01, seed=2)
    r = stacked_effect(real, placebo)
    assert abs(r["c_real_minus_placebo"] - (-0.04)) < 0.01
    assert r["p_c_real_minus_placebo"] < 0.01
    assert abs(r["alpha_placebo"] - (-0.01)) < 0.01


def test_stacked_effect_no_difference():
    r = stacked_effect(_make(extra=-0.02, seed=3), _make(extra=-0.02, seed=4))
    assert abs(r["c_real_minus_placebo"]) < 0.01
    assert r["p_c_real_minus_placebo"] > 0.05


def test_paired_by_event_sign():
    out = paired_by_event(_make(extra=-0.05, seed=5), _make(extra=0.0, seed=6), beta=0.4)
    assert out["mean_diff"] < -0.03 and out["pvalue_wilcoxon"] < 0.01 and out["share_real_lower"] > 0.9


def test_cluster_multi_matches_ols_coefficients():
    rng = np.random.default_rng(0)
    x = rng.normal(size=200)
    y = 1.0 + 2.0 * x + rng.normal(size=200) * 0.1
    res = cluster_multi(np.column_stack([np.ones(200), x]), y, np.repeat(np.arange(20), 10))
    assert np.allclose(res["coef"], [1.0, 2.0], atol=0.05)


def test_real_table_keeps_merton_and_positive_k():
    path = pd.DataFrame({"model": ["merton", "bates", "merton"], "k": [1, 1, 0], "ticker": "A", "announcement_date": "d",
                         "expiry": "e", "days_left": 5, "g_obs": 1.0, "g_spy": 1.0})
    assert len(real_table(path)) == 1


def test_placebo_day0_is_offset_trading_days():
    assert str(placebo_day0("2025-05-02", 20)) == "2025-06-02"
