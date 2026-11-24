import numpy as np
import pandas as pd

from crush.analysis.beta_groups import beta_by_group


def _path(betas, n_dates=30, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    tick = list(betas)
    for i in range(n_dates):
        m = rng.normal(0, 0.1)
        for t in tick:
            for k in (1, 2):
                rows.append({"model": "merton", "k": k, "ticker": t, "announcement_date": f"d{i}", "g_spy": np.exp(m),
                             "g_obs": np.exp(betas[t] * m + rng.normal(0, 0.01))})
    return pd.DataFrame(rows)


def test_beta_by_group_detects_heterogeneity():
    tab, test = beta_by_group(_path({"AAPL": 0.2, "MSFT": 0.8}), "ticker")
    assert abs(tab.set_index("ticker").loc["AAPL", "beta"] - 0.2) < 0.05 and test["pvalue"] < 0.01


def test_beta_by_group_equal_betas_not_rejected():
    _, test = beta_by_group(_path({"AAPL": 0.4, "MSFT": 0.4}, seed=3), "ticker")
    assert test["pvalue"] > 0.05
