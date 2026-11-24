from types import SimpleNamespace

import numpy as np
import pytest

from crush.analysis.crush import model_atm_iv
from crush.analysis.forward_c import forward_var, solve_c, theta_with_c
from crush.models.event_models import BatesEvent, MertonEvent

SLICE = SimpleNamespace(F=100.0, T=8 / 252, r=0.04)
MERTON = np.array([0.30, 1.5, -0.01, 0.05])
BATES = np.array([0.09, 3.0, 0.07, 0.6, -0.5, 1.5, -0.01, 0.05])


def test_forward_var_of_constant_term_structure_is_constant():
    assert forward_var(0.02, 0.09, 0.05, 0.09) == pytest.approx(0.09)


def test_forward_var_additivity():
    # tramo [T1, T2]: la varianza total de T2 es la de T1 más la del tramo
    k1, k2 = 0.25, 0.16
    f = forward_var(0.02, k1, 0.05, k2)
    assert 0.02 * k1 + (0.05 - 0.02) * f == pytest.approx(0.05 * k2)


@pytest.mark.parametrize("model,theta", [(MertonEvent(), MERTON), (BatesEvent(), BATES)])
def test_solve_c_recovers_known_c(model, theta):
    dt = 1 / 252
    for c_true in (0.6, 1.0, 1.3):
        target = model_atm_iv(model, theta_with_c(model, theta, dt, c_true), SLICE.F, SLICE.T, SLICE.r, False)
        assert solve_c(model, theta, SLICE, dt, target) == pytest.approx(c_true, rel=1e-4)


def test_solve_c_returns_nan_without_target():
    assert np.isnan(solve_c(MertonEvent(), MERTON, SLICE, 1 / 252, float("nan")))


def test_theta_with_c_scales_merton_variance():
    th = theta_with_c(MertonEvent(), MERTON, 1 / 252, 0.25)
    assert th[0] == pytest.approx(MERTON[0] * 0.5)
