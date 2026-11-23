import numpy as np
import pandas as pd
import pytest

from crush.analysis.compare import clustered_wilcoxon, decompose, holdout_rmse, split_chain, strike_errors, summarize_q1
from crush.calibration.quotes import synthetic_chain
from crush.models.event_models import BatesEvent, MertonEvent

MT = np.array([0.25, 1.0, -0.01, 0.05])
BT = np.array([0.0625, 3.0, 0.0625, 0.5, -0.5, 1.0, -0.01, 0.05])


def _table(n_dates=10, per_date=3, bates_better=True, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for d in range(n_dates):
        for j in range(per_date):
            row = {
                "ticker": f"T{j}", "announcement_date": f"2026-{1 + d // 4:02d}-{10 + d % 4}",
                "bic_merton": 0.0, "bic_bates": -1.0 if bates_better else 1.0, "rmse_iv_merton": 0.01, "rmse_iv_bates": 0.005,
            }
            for n, v in zip(MertonEvent().names, MT):
                row[f"merton_{n}"] = v
            for n, v in zip(BatesEvent().names, BT):
                row[f"bates_{n}"] = v
            for i in (1, 2):
                row[f"T_{i}"] = 5 * i / 252
                row[f"k_mkt_{i}"] = 0.1
                row[f"k_merton_{i}"] = MertonEvent().k_repl(MT, row[f"T_{i}"])
                row[f"k_bates_{i}"] = BatesEvent().k_repl(BT, row[f"T_{i}"])
                a = abs(rng.normal(0.05, 0.01))
                row[f"relerr_merton_{i}"] = (a + 0.05) * rng.choice([-1, 1])
                row[f"relerr_bates_{i}"] = (a if bates_better else a + 0.1) * rng.choice([-1, 1])
            rows.append(row)
    return pd.DataFrame(rows)


def test_strike_errors_sign():
    e = strike_errors(_table())
    assert (e["d_1"] > 0).all()  # Bates siempre más cerca


def test_decomposition_adds_up():
    d = decompose(_table(2, 1))
    assert np.allclose(d["diff_total"], d["K_bates"] - d["K_merton"])
    assert np.allclose(d["diff_total"], d["diff_continuous"] + d["diff_event"])


def test_wilcoxon_clusters_by_date():
    t = _table(10, 3)
    r = clustered_wilcoxon(strike_errors(t)["d_1"], t["announcement_date"])
    assert r["n_events"] == 30 and r["n_dates"] == 10 and r["pvalue"] < 0.01 and r["share_bates_closer"] == 1.0


def test_wilcoxon_few_dates_gives_nan():
    t = _table(3, 5)
    assert np.isnan(clustered_wilcoxon(strike_errors(t)["d_1"], t["announcement_date"])["pvalue"])


def test_no_difference_is_not_significant():
    t = _table(12, 2, bates_better=False)
    t["relerr_bates_1"] = t["relerr_merton_1"] * np.where(np.arange(len(t)) % 2 == 0, 0.9, 1.1)
    r = clustered_wilcoxon(strike_errors(t)["d_1"], t["announcement_date"])
    assert r["pvalue"] > 0.05


def test_summary_counts_bic():
    s = summarize_q1(_table(8, 2))
    assert s.attrs["bic_bates_wins"] == 1.0 and set(s.index) == {1, 2}


def test_split_chain_partitions_each_expiry():
    c = synthetic_chain(MertonEvent(), MT)
    tr, te = split_chain(c)
    assert len(tr) + len(te) == len(c) and abs(len(tr) - len(te)) <= 3


@pytest.mark.slow
def test_holdout_rmse_small_without_noise():
    c = synthetic_chain(MertonEvent(), MT)
    r = holdout_rmse(MertonEvent(), c, popsize=8, maxiter=40)
    assert r["rmse_test"] < 5e-3
