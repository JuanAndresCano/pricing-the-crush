import numpy as np
import pytest

from crush.calibration.calibrate import calibrate
from crush.calibration.quotes import perturb_iv, synthetic_chain
from crush.calibration.selection import bic, rmse_iv
from crush.models.black_scholes import implied_vol
from crush.models.event import event_cf, split_event_variance
from crush.models.event_models import BatesEvent, MertonEvent
from crush.models.fourier import CallPricer
from crush.models.merton import MertonParams, k_repl as merton_k_repl, merton_call

R = 0.03
MERTON = MertonEvent()
TRUE_M = np.array([0.25, 1.5, -0.02, 0.07])  # sigma, Lambda_e, mu_e, sigma_e
BATES = BatesEvent()
TRUE_B = np.array([0.09, 6.0, 0.0625, 0.8, -0.6, 1.5, -0.02, 0.07])


# ---------- el factor de evento ----------


def test_event_factor_keeps_martingale_condition():
    assert event_cf(-1j, 1.5, -0.02, 0.07) == pytest.approx(1.0, abs=1e-12)
    assert event_cf(0.0, 1.5, -0.02, 0.07) == pytest.approx(1.0, abs=1e-14)


@pytest.mark.parametrize("T", [5 / 252, 10 / 252, 21 / 252])
def test_merton_event_equals_merton_with_lambda_T_equal_Lambda_e(T):
    # Con un solo vencimiento, Merton-evento es el Merton de la sección 2 con lambda T = Lambda_e
    F = 100 * np.exp(R * T)
    K = F * np.exp(np.linspace(-0.15, 0.15, 9))
    mp = MertonParams(TRUE_M[0], TRUE_M[1] / T, TRUE_M[2], TRUE_M[3])
    fast = MERTON.call_prices(TRUE_M, F, T, R, True, CallPricer(K))
    series = np.array([merton_call(F, k, T, R, mp) for k in K])
    np.testing.assert_allclose(fast, series, atol=1e-10)
    assert MERTON.k_repl(TRUE_M, T) == pytest.approx(merton_k_repl(mp), rel=1e-12)


def test_event_variance_is_fixed_across_maturities():
    # Corrección 1: con Lambda_e fijo, la bolsa de varianza no depende del plazo
    e = MERTON.eps_repl(TRUE_M)
    for T in (5 / 252, 10 / 252, 21 / 252):
        assert (MERTON.k_repl(TRUE_M, T) - MERTON.base(TRUE_M, T)) * T == pytest.approx(e, rel=1e-12)


def test_constant_lambda_cannot_reproduce_an_event_step_but_fixed_event_can():
    # Ecuación (26) aplicada a volatilidades ATM del modelo, con dos vencimientos que cruzan el anuncio
    T1, T2 = 10 / 252, 21 / 252

    def iv_event(T):
        F = 100 * np.exp(R * T)
        c = float(MERTON.call_prices(TRUE_M, F, T, R, True, CallPricer([F]))[0])
        return implied_vol(c, F, F, T, R, True)

    def iv_const(T, lam):
        F = 100 * np.exp(R * T)
        return implied_vol(merton_call(F, F, T, R, MertonParams(0.25, lam, -0.02, 0.07)), F, F, T, R, True)

    fixed = split_event_variance(iv_event(T1), T1, iv_event(T2), T2)
    const = split_event_variance(iv_const(T1, TRUE_M[1] / T1), T1, iv_const(T2, TRUE_M[1] / T1), T2)
    # Evento fijo: recupera la volatilidad base y la mayor parte de la varianza de evento (ATM es aproximado)
    assert fixed.sigma_b == pytest.approx(0.25, abs=0.015)
    assert fixed.eps == pytest.approx(MERTON.eps_repl(TRUE_M), rel=0.25)
    # Lambda constante: la base absorbe los saltos (el doble de la verdadera) y el evento desaparece
    assert const.sigma_b > 1.8 * 0.25
    assert abs(const.eps) < 0.15 * MERTON.eps_repl(TRUE_M)


# ---------- calibración sintética de Merton-evento (prueba 10) ----------


@pytest.fixture(scope="module")
def merton_chain():
    return synthetic_chain(MERTON, TRUE_M)


def test_merton_event_parameters_are_recovered_without_noise(merton_chain):
    res = calibrate(MERTON, merton_chain, n_starts=1, seed=0, popsize=10, maxiter=60)
    assert res.rmse_iv < 1e-8
    np.testing.assert_allclose(res.theta, TRUE_M, rtol=1e-4, atol=1e-6)


def test_merton_event_recovery_with_implied_vol_noise(merton_chain):
    noisy = perturb_iv(merton_chain, 0.002, np.random.default_rng(1))  # 0,2 puntos de volatilidad
    res = calibrate(MERTON, noisy, n_starts=1, seed=0, popsize=10, maxiter=60)
    assert res.rmse_iv == pytest.approx(0.002, rel=0.35)  # el ajuste llega al nivel del ruido
    # Las cantidades que usa el estudio se recuperan mejor que 1%
    assert MERTON.eps_repl(res.theta) == pytest.approx(MERTON.eps_repl(TRUE_M), rel=0.01)
    for T in (5 / 252, 10 / 252, 21 / 252):
        assert MERTON.k_repl(res.theta, T) == pytest.approx(MERTON.k_repl(TRUE_M, T), rel=0.005)
    np.testing.assert_allclose(res.theta, TRUE_M, rtol=0.05)


def test_bic_follows_equation_42(merton_chain):
    from crush.calibration.selection import iv_errors

    noisy = perturb_iv(merton_chain, 0.002, np.random.default_rng(3))
    e = iv_errors(MERTON, TRUE_M, noisy)
    m = e.size
    expected = m * np.log(np.sum(e**2) / m) + MERTON.n_params * np.log(m)
    assert bic(MERTON, TRUE_M, noisy) == pytest.approx(expected, rel=1e-12)
    assert rmse_iv(MERTON, TRUE_M, noisy) == pytest.approx(np.sqrt(np.mean(e**2)), rel=1e-12)
    # Con el mismo ajuste, un modelo con 4 parámetros más tiene un BIC mayor en 4 ln(M)
    assert BATES.n_params * np.log(m) - MERTON.n_params * np.log(m) == pytest.approx(4 * np.log(m))


# ---------- calibración sintética de Bates-evento (prueba 10, lenta) ----------


@pytest.mark.slow
def test_bates_event_parameters_are_recovered_without_noise():
    chain = synthetic_chain(BATES, TRUE_B)
    res = calibrate(BATES, chain, n_starts=1, seed=0, popsize=12, maxiter=100)
    assert res.rmse_iv < 1e-8
    np.testing.assert_allclose(res.theta, TRUE_B, rtol=2e-3, atol=1e-4)
    assert BATES.eps_repl(res.theta) == pytest.approx(BATES.eps_repl(TRUE_B), rel=1e-4)
