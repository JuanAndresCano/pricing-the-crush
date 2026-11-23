import numpy as np
import pytest

from crush.market.strike_mkt import k_repl_from_otm_prices
from crush.models.bates import (
    BatesParams,
    bates_call,
    bates_cf,
    bates_put,
    jump_share,
    k_qv,
    k_repl,
)
from crush.models.black_scholes import bs_price
from crush.models.fourier import gil_pelaez_call
from crush.models.heston import HestonParams, heston_cf, v_bar
from crush.models.merton import MertonParams, merton_call, merton_cf

F0, R = 100.0, 0.03
T = 10 / 252

HESTON = BatesParams(v0=0.09, kappa=6.0, theta=0.0625, xi=0.8, rho=-0.6)  # sin saltos
BATES = BatesParams(v0=0.09, kappa=6.0, theta=0.0625, xi=0.8, rho=-0.6, lam=25.0, mu_j=-0.02, sigma_j=0.07)
LONG = BatesParams(v0=0.04, kappa=1.5, theta=0.05, xi=0.5, rho=-0.7, lam=0.5, mu_j=-0.1, sigma_j=0.15)


@pytest.mark.parametrize("p,TT", [(HESTON, T), (BATES, T), (LONG, 2.0)])
def test_characteristic_function_martingale_condition(p, TT):
    # Prueba 1: phi(-i) = F0
    assert bates_cf(-1j, TT, F0, p) == pytest.approx(F0, rel=1e-10)
    assert bates_cf(0.0, TT, F0, p) == pytest.approx(1.0, abs=1e-12)


def test_v_bar_limits():
    p = HestonParams(0.09, 6.0, 0.04, 0.8, -0.6)
    assert v_bar(1e-9, p) == pytest.approx(p.v0, rel=1e-6)  # plazo corto: manda la varianza de hoy
    assert v_bar(1e3, p) == pytest.approx(p.theta, rel=1e-2)  # plazo largo: manda theta
    # (15) coincide con el promedio numérico de E[v_t]
    t = np.linspace(0, 0.5, 200001)
    ev = p.theta + (p.v0 - p.theta) * np.exp(-p.kappa * t)
    assert v_bar(0.5, p) == pytest.approx(float(np.trapezoid(ev, t) / 0.5), rel=1e-8)


def test_heston_with_tiny_xi_reduces_to_black_scholes_with_vbar():
    # Prueba 3: lam = 0 y xi muy pequeño -> Black-Scholes con varianza vbar(T).
    # Con rho != 0 queda un sesgo de primer orden en rho*xi (la covarianza precio-varianza), así que
    # el límite exacto se prueba con rho = 0, donde el efecto es de segundo orden en xi.
    p = BatesParams(v0=0.09, kappa=6.0, theta=0.0625, xi=1e-3, rho=0.0)
    K = np.linspace(85, 118, 12)
    sigma = np.sqrt(v_bar(T, p.heston))
    np.testing.assert_allclose(bates_call(F0, K, T, R, p), bs_price(F0, K, T, sigma, R, True), atol=1e-5)


def test_xi_zero_branch_matches_tiny_xi_limit():
    a = BatesParams(v0=0.09, kappa=6.0, theta=0.0625, xi=0.0, rho=0.0)
    b = BatesParams(v0=0.09, kappa=6.0, theta=0.0625, xi=1e-4, rho=0.0)
    u = np.linspace(0.1, 50, 40)
    np.testing.assert_allclose(bates_cf(u, T, F0, a), bates_cf(u, T, F0, b), rtol=1e-4, atol=1e-6)


def test_bates_with_constant_variance_reduces_to_merton():
    # Prueba 3: xi = 0 y v0 = theta -> Merton con sigma^2 = theta
    theta = 0.0625
    bp = BatesParams(v0=theta, kappa=3.0, theta=theta, xi=0.0, rho=-0.5, lam=25.0, mu_j=-0.02, sigma_j=0.07)
    mp = MertonParams(sigma=np.sqrt(theta), lam=25.0, mu_j=-0.02, sigma_j=0.07)
    u = np.array([0.5, 3.0, 20.0 + 1j, -1j])
    np.testing.assert_allclose(bates_cf(u, T, F0, bp), merton_cf(u, T, F0, mp), rtol=1e-12)
    K = np.linspace(85, 118, 12)
    np.testing.assert_allclose(bates_call(F0, K, T, R, bp), merton_call(F0, K, T, R, mp), atol=1e-6)


@pytest.mark.parametrize("p,TT", [(BATES, T), (LONG, 2.0)])
def test_fourier_converges_when_integration_limit_doubles(p, TT):
    K = np.linspace(0.85 * F0, 1.15 * F0, 7)
    a = bates_call(F0, K, TT, R, p, u_max=1000, panels=500)
    b = bates_call(F0, K, TT, R, p, u_max=2000, panels=1000)
    np.testing.assert_allclose(a, b, atol=1e-8)


@pytest.mark.parametrize("p,TT", [(BATES, T), (LONG, 1.0)])
def test_prices_respect_no_arbitrage(p, TT):
    K = np.linspace(0.8 * F0, 1.25 * F0, 40)
    c = bates_call(F0, K, TT, R, p)
    disc = np.exp(-R * TT)
    assert np.all(c >= disc * np.maximum(F0 - K, 0) - 1e-8)  # sobre el valor intrínseco
    assert np.all(np.diff(c) < 0)  # decreciente en K
    assert np.all(np.diff(c, 2) > -1e-9)  # convexo en K (densidad no negativa)


def test_negative_rho_gives_negative_skew_and_positive_rho_the_opposite():
    from crush.models.black_scholes import implied_vol

    def iv(p, K):
        price = float(bates_call(F0, K, T, R, p))
        return implied_vol(price, F0, K, T, R, True)

    neg = BatesParams(v0=0.0625, kappa=6.0, theta=0.0625, xi=0.8, rho=-0.6)
    pos = BatesParams(v0=0.0625, kappa=6.0, theta=0.0625, xi=0.8, rho=0.6)
    assert iv(neg, 90.0) > iv(neg, 110.0)  # efecto apalancamiento: caídas con más volatilidad
    assert iv(pos, 90.0) < iv(pos, 110.0)


def test_put_call_parity():
    K = np.linspace(85, 115, 7)
    c, put = bates_call(F0, K, T, R, BATES), bates_put(F0, K, T, R, BATES)
    np.testing.assert_allclose(c - put, np.exp(-R * T) * (F0 - K), atol=1e-12)


@pytest.mark.parametrize("p", [HESTON, BATES])
def test_fine_strip_integral_matches_closed_form_k_repl(p):
    # Prueba 5: la integral (4) con precios del modelo reproduce (23); sin saltos, vbar(T)
    K = np.geomspace(0.3 * F0, 3.0 * F0, 1801)
    calls = np.concatenate([
        bates_call(F0, K[i : i + 300], T, R, p, u_max=600, panels=300) for i in range(0, len(K), 300)
    ])
    puts = calls - np.exp(-R * T) * (F0 - K)
    q = np.where(K < F0, puts, calls)
    assert k_repl_from_otm_prices(K, q, T, R) == pytest.approx(k_repl(T, p), rel=2e-3)


def test_replication_bias_comes_only_from_jumps():
    # (24): K_repl - K_QV = 2 lam E[g(J)]; sin saltos no hay sesgo aunque la varianza sea estocástica
    assert k_repl(T, HESTON) == pytest.approx(k_qv(T, HESTON), rel=1e-15)
    assert k_repl(T, BATES) < k_qv(T, BATES)  # saltos bajistas
    assert 0 < jump_share(T, BATES) < 1
