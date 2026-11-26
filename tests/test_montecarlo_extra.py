import numpy as np
import pytest

from crush.analysis.montecarlo_extra import DT, _simulate, control_variate, mc_extra, value_after_event
from crush.simulation.rv import realized_variance
from test_montecarlo import _row

P = {"v0": 0.06, "kappa": 3.0, "theta": 0.05, "xi": 0.5, "rho": -0.5, "Lambda_e": 1.0, "mu_e": -0.005, "sigma_e": 0.05}
PM = {"sigma": 0.25, "Lambda_e": 1.0, "mu_e": -0.005, "sigma_e": 0.05}


def test_crush_one_changes_nothing_and_small_c_lowers_variance():
    base, _ = _simulate(P, "bates", 6, 20_000, 1)
    same, _ = _simulate(P, "bates", 6, 20_000, 1, c=1.0)
    low, _ = _simulate(P, "bates", 6, 20_000, 1, c=0.5)
    assert np.allclose(base, same)
    assert np.allclose(base[:, 0], low[:, 0])  # el paso del anuncio no cambia
    assert realized_variance(low).mean() < realized_variance(base).mean()


def test_merton_value_after_event_matches_equation_35():
    n, k_var = 5, 0.2
    lr, var = _simulate(PM, "merton", n, 1000, 2)
    v = value_after_event(lr, PM, "merton", var, k_var)
    T = n * DT
    manual = (DT / T) * lr[:, 0] ** 2 / DT + ((T - DT) / T) * PM["sigma"] ** 2 - k_var
    assert np.allclose(v, manual)


def test_control_variate_keeps_mean_and_does_not_raise_error():
    lr, var = _simulate(P, "bates", 8, 100_000, 3)
    rv = realized_variance(lr)
    mean, se = control_variate(rv, var, P)
    plain_se = rv.std(ddof=1) / np.sqrt(rv.size)
    assert abs(mean - rv.mean()) < 4 * plain_se and se <= plain_se * 1.0001


@pytest.mark.parametrize("tag", ["merton", "bates"])
def test_mc_extra_row_has_the_new_measures(tag):
    r = mc_extra(_row(), tag, 1, c=0.9, n_paths=20_000)
    assert 0.0 <= r["corr2_value_event"] <= 1.0 and r["value_q01"] <= r["value_q05"] and r["k_hat_c"] < r["k_hat"]
