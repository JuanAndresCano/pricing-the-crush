"""Calibración de un evento a partir de cotizaciones bid/ask simuladas (ver tests/test_market.py)."""

import numpy as np
import pytest

from crush.calibration.calibrate import calibrate
from crush.calibration.quotes import synthetic_chain
from crush.calibration.real import calibrate_event, calibration_row, chain_from_slices, theta_from_far
from crush.market.chain import build_slices
from crush.models.event_models import BatesEvent, MertonEvent
from tests.test_market import EXPIRIES, PRE, R, S0, THETA, event  # noqa: F401


def test_fixed_parameters_are_respected_and_counted():
    m = BatesEvent()
    true = np.array([0.09, 3.0, 0.0625, 0.5, -0.5, 1.0, -0.01, 0.05])
    chain = synthetic_chain(m, true)
    res = calibrate(m, chain, popsize=8, maxiter=30, fixed={"theta": 0.0625})
    assert res.theta[2] == 0.0625
    assert res.rmse_iv < 5e-3
    with pytest.raises(ValueError):
        calibrate(m, chain, fixed={"nope": 1.0})


def test_theta_from_far_uses_longest_expiry(event):  # noqa: F811
    s = build_slices(event.pre, PRE, S0, R)
    th, strong = theta_from_far(s)
    assert th == pytest.approx(s[-1].atm_iv**2) and strong  # 25 días hábiles >= 21


@pytest.mark.slow
def test_calibrate_merton_event_from_market_chain(event):  # noqa: F811
    cal = calibrate_event(event, R, n_starts=1, popsize=10, maxiter=60)
    row = calibration_row(event, cal)
    assert row["rmse_iv_merton"] < 5e-3  # el dato salió de Merton-evento
    assert row["merton_sigma"] == pytest.approx(THETA[0], rel=0.1)
    assert abs(row["relerr_merton_1"]) < 0.05
