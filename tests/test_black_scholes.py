import numpy as np
import pytest

from crush.models.black_scholes import (
    bs_price,
    bs_vega,
    implied_vol,
    implied_vol_array,
    price_bounds,
)


def test_atm_reference_value():
    # F = K = 100, T = 1, sigma = 0.2, r = 0: C = 100 (2 N(0.1) - 1) = 7.9655674554
    assert bs_price(100, 100, 1.0, 0.2, 0.0, True) == pytest.approx(7.9655674554, abs=1e-9)


@pytest.mark.parametrize("r", [0.0, 0.03, 0.10])
def test_put_call_parity(r):
    # Prueba 4: C - P = e^{-rT} (F - K) en todos los strikes
    K = np.linspace(60, 160, 21)
    F, T, sigma = 100.0, 10 / 252, 0.45
    c = bs_price(F, K, T, sigma, r, True)
    p = bs_price(F, K, T, sigma, r, False)
    np.testing.assert_allclose(c - p, np.exp(-r * T) * (F - K), atol=1e-12)


def test_vega_matches_finite_difference():
    F, K, T, sigma, r = 100.0, 105.0, 0.1, 0.3, 0.02
    h = 1e-6
    fd = (bs_price(F, K, T, sigma + h, r) - bs_price(F, K, T, sigma - h, r)) / (2 * h)
    assert bs_vega(F, K, T, sigma, r) == pytest.approx(float(fd), rel=1e-6)


@pytest.mark.parametrize("is_call", [True, False])
@pytest.mark.parametrize("T", [5 / 252, 10 / 252, 1.0])
@pytest.mark.parametrize("sigma", [0.1, 0.25, 0.6, 1.5])
def test_implied_vol_round_trip(T, sigma, is_call):
    # Prueba 6: precio -> volatilidad implícita -> precio
    F, r = 100.0, 0.04
    for K in np.linspace(0.7 * F, 1.4 * F, 15):
        price = float(bs_price(F, K, T, sigma, r, is_call))
        lo, _ = price_bounds(F, K, T, r, is_call)
        if price - lo < 1e-7:  # sin valor temporal no hay información sobre sigma
            continue
        iv = implied_vol(price, F, K, T, r, is_call)
        assert np.isfinite(iv)
        assert float(bs_price(F, K, T, iv, r, is_call)) == pytest.approx(price, abs=1e-10)


def test_implied_vol_recovers_sigma_near_the_money():
    F, K, T, sigma, r = 100.0, 100.0, 5 / 252, 0.6, 0.04
    price = float(bs_price(F, K, T, sigma, r, True))
    assert implied_vol(price, F, K, T, r, True) == pytest.approx(sigma, abs=1e-8)


def test_implied_vol_nan_outside_no_arbitrage_bounds():
    F, K, T, r = 100.0, 100.0, 0.1, 0.02
    lo, hi = price_bounds(F, K, T, r, True)
    assert np.isnan(implied_vol(lo - 0.01, F, K, T, r, True))  # por debajo del valor intrínseco
    assert np.isnan(implied_vol(hi + 0.01, F, K, T, r, True))  # por encima del techo
    assert np.isnan(implied_vol(1.0, F, K, 0.0, r, True))  # T = 0


def test_zero_vol_returns_intrinsic_value():
    assert float(bs_price(110.0, 100.0, 0.1, 0.0, 0.0, True)) == pytest.approx(10.0)
    assert float(bs_price(90.0, 100.0, 0.1, 0.0, 0.0, False)) == pytest.approx(10.0)


def test_implied_vol_array_matches_scalar():
    F, T, r = 100.0, 10 / 252, 0.03
    K = np.array([90.0, 100.0, 110.0])
    is_call = np.array([False, True, True])
    price = np.array([float(bs_price(F, k, T, 0.4, r, c)) for k, c in zip(K, is_call)])
    iv = implied_vol_array(price, F, K, T, r, is_call)
    np.testing.assert_allclose(iv, 0.4, atol=1e-8)
