import pandas as pd

from crush.analysis.montecarlo import mc_summary


def _row():
    return pd.Series({
        "ticker": "AAA", "announcement_date": "2026-01-01", "T_1": 8 / 252, "k_mkt_1": 0.35, "k_merton_1": 0.34, "k_bates_1": 0.35,
        "merton_sigma": 0.25, "merton_Lambda_e": 1.0, "merton_mu_e": -0.005, "merton_sigma_e": 0.05,
        "bates_v0": 0.06, "bates_kappa": 3.0, "bates_theta": 0.06, "bates_xi": 0.5, "bates_rho": -0.5,
        "bates_Lambda_e": 1.0, "bates_mu_e": -0.005, "bates_sigma_e": 0.05,
    })


def test_mc_matches_closed_form_merton():
    r = mc_summary(_row(), "merton", 1, n_paths=200_000, seed=1)
    assert abs(r["z"]) < 3 and abs(r["martingale"] - 1.0) < 0.01


def test_mc_matches_closed_form_bates():
    r = mc_summary(_row(), "bates", 1, n_paths=200_000, seed=2)
    assert abs(r["rel_diff"]) < 0.02 and abs(r["martingale"] - 1.0) < 0.01
