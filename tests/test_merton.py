import numpy as np
import pytest

from crush.market.strike_mkt import k_repl_from_otm_prices
from crush.models.black_scholes import bs_price
from crush.models.fourier import gil_pelaez_call
from crush.models.merton import (
    MertonParams,
    jump_share,
    k_qv,
    k_repl,
    merton_call,
    merton_cf,
    merton_put,
)

F0, R = 100.0, 0.03
T = 10 / 252
# Ventana de earnings: lam T ~ 1, salto de unos 7% con sesgo bajista
EVENT = MertonParams(sigma=0.25, lam=25.0, mu_j=-0.02, sigma_j=0.07)
PARAMS = [
    EVENT,
    MertonParams(sigma=0.30, lam=5.0, mu_j=0.03, sigma_j=0.05),
    MertonParams(sigma=0.15, lam=60.0, mu_j=-0.04, sigma_j=0.04),
]


@pytest.mark.parametrize("p", PARAMS)
def test_characteristic_function_martingale_condition(p):
    # Prueba 1: phi(-i) = F0
    assert merton_cf(-1j, T, F0, p) == pytest.approx(F0, rel=1e-12)
    assert merton_cf(0.0, T, F0, p) == pytest.approx(1.0, abs=1e-14)


@pytest.mark.parametrize("p", PARAMS)
def test_series_matches_fourier_inversion(p):
    # Prueba 2: la serie (9) y Gil-Pelaez dan el mismo precio
    K = np.linspace(80, 125, 19)
    series = merton_call(F0, K, T, R, p)
    fourier = gil_pelaez_call(lambda u: merton_cf(u, T, F0, p), F0, K, T, R)
    np.testing.assert_allclose(series, fourier, atol=1e-6)


def test_fourier_converges_when_integration_limit_doubles():
    K = np.linspace(85, 115, 7)
    phi = lambda u: merton_cf(u, T, F0, EVENT)
    a = gil_pelaez_call(phi, F0, K, T, R, u_max=1000, panels=500)
    b = gil_pelaez_call(phi, F0, K, T, R, u_max=2000, panels=1000)
    np.testing.assert_allclose(a, b, atol=1e-9)


def test_series_matches_textbook_form_with_modified_intensity():
    # Forma del README: sum_n e^{-lam' T}(lam' T)^n/n! BS(S0,K,T,sigma_n,r_n), lam' = lam (1+k)
    from math import exp, factorial, log

    p = EVENT
    S0 = F0 * exp(-R * T)
    K = 104.0
    lamp = p.lam * (1 + p.k)
    total = 0.0
    for n in range(40):
        sig_n = np.sqrt(p.sigma**2 + n * p.sigma_j**2 / T)
        r_n = R - p.lam * p.k + n * log(1 + p.k) / T
        w = exp(-lamp * T) * (lamp * T) ** n / factorial(n)
        total += w * float(bs_price(S0 * exp(r_n * T), K, T, sig_n, r_n, True))
    assert merton_call(F0, K, T, R, p) == pytest.approx(total, abs=1e-10)


def test_reduces_to_black_scholes_without_jumps():
    p = MertonParams(sigma=0.3, lam=0.0, mu_j=0.0, sigma_j=0.0)
    K = np.array([90.0, 100.0, 110.0])
    np.testing.assert_allclose(merton_call(F0, K, T, R, p), bs_price(F0, K, T, 0.3, R, True), atol=1e-12)


@pytest.mark.parametrize("p", PARAMS)
def test_put_call_parity(p):
    K = np.linspace(85, 115, 7)
    c, put = merton_call(F0, K, T, R, p), merton_put(F0, K, T, R, p)
    np.testing.assert_allclose(c - put, np.exp(-R * T) * (F0 - K), atol=1e-12)


@pytest.mark.parametrize("p", PARAMS)
def test_k_qv_equals_variance_of_log_return(p):
    # (11): Var[ln S_T] = T (sigma^2 + lam (mu_j^2 + sigma_j^2)) = -d^2/du^2 ln phi(u) en u = 0
    h = 1e-3
    f = lambda u: np.log(merton_cf(u, T, F0, p)).real
    second = (f(h) - 2 * f(0.0) + f(-h)) / h**2
    assert -second == pytest.approx(T * k_qv(p), rel=1e-5)


@pytest.mark.parametrize("p", PARAMS)
def test_fine_strip_integral_matches_closed_form_k_repl(p):
    # Prueba 5: la integral (4) con precios del modelo reproduce la fórmula cerrada (13)
    K = np.linspace(0.3 * F0, 3.0 * F0, 40001)
    q = np.where(K < F0, merton_put(F0, K, T, R, p), merton_call(F0, K, T, R, p))
    assert k_repl_from_otm_prices(K, q, T, R) == pytest.approx(k_repl(p), rel=1e-4)


def test_sign_of_replication_bias_follows_jump_sign():
    down = MertonParams(0.25, 25.0, -0.05, 0.05)
    up = MertonParams(0.25, 25.0, 0.05, 0.05)
    assert k_repl(down) < k_qv(down)
    assert k_repl(up) > k_qv(up)


def test_bias_is_close_to_lam_third_moment_over_three():
    p = MertonParams(sigma=0.25, lam=25.0, mu_j=-0.03, sigma_j=0.03)
    third = p.mu_j**3 + 3 * p.mu_j * p.sigma_j**2
    fourth = p.mu_j**4 + 6 * p.mu_j**2 * p.sigma_j**2 + 3 * p.sigma_j**4
    approx = p.lam * third / 3 + p.lam * fourth / 12  # con el término de cuarto orden
    assert k_repl(p) - k_qv(p) == pytest.approx(approx, rel=0.05)


def test_jump_share_between_zero_and_one():
    assert 0 < jump_share(EVENT) < 1
