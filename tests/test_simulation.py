import numpy as np
import pytest

from crush.config import TRADING_DAYS_PER_YEAR as A
from crush.models.bates import BatesParams, bates_call
from crush.models.bates import k_qv as bates_k_qv
from crush.models.heston import HestonParams
from crush.models.merton import MertonParams, merton_call
from crush.simulation.bates_qe import qe_variance_step, simulate_bates_logreturns
from crush.simulation.jumps import JumpSchedule
from crush.simulation.merton_mc import simulate_merton_logreturns
from crush.simulation.rv import (
    event_share,
    mean_and_stderr,
    realized_variance,
    rv_ex_event,
    swap_value,
)

DT = 1.0 / A
N_STEPS = 10
T = N_STEPS * DT
F0, R = 100.0, 0.03
S0 = F0 * np.exp(-R * T)
M = 400_000

MERTON = MertonParams(sigma=0.25, lam=25.0, mu_j=-0.02, sigma_j=0.07)
EVENT_STEP = 4
LAM_E, MU_E, SIG_E = 2.0, -0.02, 0.07


def call_mc(logret, K):
    st = S0 * np.exp(logret.sum(axis=1))
    return mean_and_stderr(np.exp(-R * T) * np.maximum(st - K, 0.0))


def exact_discrete_rv(sigma, schedule, b=R):
    """E[RV] exacta del esquema de Merton: suma de (varianza + media^2) de cada paso."""
    var = sigma**2 * DT + schedule.lam_dt * (schedule.mu**2 + schedule.sigma**2)
    mean = (b - 0.5 * sigma**2) * DT - schedule.lam_dt * schedule.k + schedule.lam_dt * schedule.mu
    return A / schedule.n * float(np.sum(var + mean**2))


# ---------- Merton ----------


@pytest.fixture(scope="module")
def merton_uniform():
    sch = JumpSchedule.constant(N_STEPS, DT, MERTON.lam, MERTON.mu_j, MERTON.sigma_j)
    lr = simulate_merton_logreturns(MERTON.sigma, sch, M, b=R, rng=np.random.default_rng(123))
    return sch, lr


@pytest.fixture(scope="module")
def merton_event():
    sch = JumpSchedule.with_event(N_STEPS, DT, EVENT_STEP, LAM_E, MU_E, SIG_E)
    lr = simulate_merton_logreturns(0.25, sch, M, b=R, rng=np.random.default_rng(321))
    return sch, lr


def test_merton_martingale(merton_uniform):
    # Prueba 8: E[S_T] e^{-bT} / S_0 = 1
    _, lr = merton_uniform
    mean, se = mean_and_stderr(np.exp(lr.sum(axis=1)))
    assert abs(mean - np.exp(R * T)) < 3 * se


@pytest.mark.parametrize("K", [92.0, 100.0, 108.0])
def test_merton_mc_price_matches_series(merton_uniform, K):
    _, lr = merton_uniform
    mc, se = call_mc(lr, K)
    assert abs(mc - merton_call(F0, K, T, R, MERTON)) < 4 * se


def test_merton_rv_matches_exact_discrete_expectation(merton_uniform):
    sch, lr = merton_uniform
    mean, se = mean_and_stderr(realized_variance(lr))
    assert abs(mean - exact_discrete_rv(MERTON.sigma, sch)) < 3 * se


def test_merton_event_day_rv_matches_exact_expectation(merton_event):
    sch, lr = merton_event
    mean, se = mean_and_stderr(realized_variance(lr))
    assert abs(mean - exact_discrete_rv(0.25, sch)) < 3 * se


def test_event_day_squared_return_includes_compensator_term(merton_event):
    # E[R_e^2] = sigma^2 D + Lambda (mu^2 + s^2) + (mean del paso)^2, con el compensador dentro de la media.
    # El término de la media es Lambda^2 (mu - k)^2, pequeño, no Lambda^2 mu^2 (hallazgo de la revisión).
    sch, lr = merton_event
    k = float(sch.k[EVENT_STEP])
    mean_step = (R - 0.5 * 0.25**2) * DT - LAM_E * k + LAM_E * MU_E
    expected = 0.25**2 * DT + LAM_E * (MU_E**2 + SIG_E**2) + mean_step**2
    mean, se = mean_and_stderr(lr[:, EVENT_STEP] ** 2)
    assert abs(mean - expected) < 3 * se
    assert abs(mean - (expected + LAM_E**2 * MU_E**2)) > 10 * se  # la versión sin compensador no cuadra


def test_event_variance_is_concentrated_on_the_announcement_day(merton_event):
    _, lr = merton_event
    share = event_share(lr, EVENT_STEP)
    # Cociente de esperanzas: (sigma^2 D + eps) / (sigma^2 T + eps). El promedio de los cocientes es menor
    # porque con probabilidad e^{-2} (13,5%) no cae ningún salto ese día.
    eps = LAM_E * (MU_E**2 + SIG_E**2)
    expected_ratio = (0.25**2 * DT + eps) / (0.25**2 * T + eps)
    assert (lr[:, EVENT_STEP] ** 2).mean() / (lr**2).sum(axis=1).mean() == pytest.approx(expected_ratio, abs=0.01)
    assert share.mean() > 5 * (1 / N_STEPS)  # muy por encima del reparto uniforme de 1/n
    # Sin el día del anuncio, la RV vuelve a la varianza de la difusión
    mean, se = mean_and_stderr(rv_ex_event(lr, EVENT_STEP))
    assert abs(mean - 0.25**2) < 3 * se + 2e-4


def test_same_seed_gives_identical_paths():
    sch = JumpSchedule.constant(N_STEPS, DT, 25.0, -0.02, 0.07)
    a = simulate_merton_logreturns(0.25, sch, 1000, rng=np.random.default_rng(7))
    b = simulate_merton_logreturns(0.25, sch, 1000, rng=np.random.default_rng(7))
    np.testing.assert_array_equal(a, b)


def test_antithetic_requires_even_number_of_paths():
    sch = JumpSchedule.constant(N_STEPS, DT, 0.0, 0.0, 0.0)
    with pytest.raises(ValueError):
        simulate_merton_logreturns(0.25, sch, 11)


# ---------- Bates (QE) ----------

BATES = BatesParams(v0=0.09, kappa=6.0, theta=0.0625, xi=0.8, rho=-0.6, lam=25.0, mu_j=-0.02, sigma_j=0.07)


@pytest.fixture(scope="module")
def bates_paths():
    sch = JumpSchedule.constant(N_STEPS, DT, BATES.lam, BATES.mu_j, BATES.sigma_j)
    lr, v = simulate_bates_logreturns(
        BATES.heston, sch, M, b=R, rng=np.random.default_rng(99), return_variance=True
    )
    return lr, v


def test_bates_martingale(bates_paths):
    lr, _ = bates_paths
    mean, se = mean_and_stderr(np.exp(lr.sum(axis=1)))
    assert abs(mean - np.exp(R * T)) < 3 * se + 1e-3


def test_bates_variance_mean_matches_theory(bates_paths):
    # QE reproduce el primer momento condicional: E[v_T] = theta + (v0 - theta) e^{-kappa T}
    _, v = bates_paths
    mean, se = mean_and_stderr(v[:, -1])
    assert abs(mean - (BATES.theta + (BATES.v0 - BATES.theta) * np.exp(-BATES.kappa * T))) < 3 * se
    assert np.all(v >= 0)


def test_bates_rv_matches_closed_form_k_qv(bates_paths):
    # Prueba 8: la RV simulada coincide con K_QV de (22), dentro del error estándar y la discretización
    lr, _ = bates_paths
    mean, se = mean_and_stderr(realized_variance(lr))
    k = bates_k_qv(T, BATES)
    assert abs(mean - k) < 3 * se + 0.01 * k


@pytest.mark.parametrize("K", [92.0, 100.0, 108.0])
def test_bates_mc_price_matches_fourier(bates_paths, K):
    # Validación independiente de la implementación por Fourier (Heston/Bates), que faltaba de F1
    lr, _ = bates_paths
    mc, se = call_mc(lr, K)
    assert abs(mc - float(bates_call(F0, K, T, R, BATES))) < 4 * se + 0.01


def test_qe_exponential_branch_keeps_variance_nonnegative_and_unbiased():
    # xi grande y v0 chico fuerzan psi > 1.5 (rama exponencial), con la condición de Feller violada
    p = HestonParams(v0=0.01, kappa=2.0, theta=0.04, xi=1.5, rho=-0.7)
    assert p.feller < 0
    rng = np.random.default_rng(5)
    n = 400_000
    v = np.full(n, p.v0)
    zv, u = rng.standard_normal(n), rng.random(n)
    vn = qe_variance_step(v, p.kappa, p.theta, p.xi, 1 / 12, zv, u)
    assert np.all(vn >= 0)
    assert np.mean(vn == 0) > 0.01  # masa en cero, como debe
    mean, se = mean_and_stderr(vn)
    assert abs(mean - (p.theta + (p.v0 - p.theta) * np.exp(-p.kappa / 12))) < 3 * se


# ---------- Valor del swap tras el anuncio ----------


def test_swap_value_after_announcement_is_Re2_minus_eps_over_T():
    # Prueba 9, ecuación (36): V = N (R_e^2 - eps) / T cuando la difusión sale como se esperaba
    sb2, eps, t, TT, N = 0.045, 0.00625, 5 / 252, 10 / 252, 100.0
    for Re in (0.0, 0.05, 0.079, 0.15):
        rv_0t = (sb2 * t + Re**2) / t
        v = swap_value(rv_0t, t, TT, k_forward=sb2, k_var=sb2 + eps / TT, notional=N)
        assert v == pytest.approx(N * (Re**2 - eps) / TT, abs=1e-9)


def test_simulated_swap_has_zero_mean_value_after_announcement(merton_event):
    # El crush por sí solo no genera pérdida: con el sobre bien valorado, el valor medio del swap es ~0
    _, lr = merton_event
    t_steps = 5
    t = t_steps * DT
    eps = LAM_E * (MU_E**2 + SIG_E**2)
    k_var = 0.25**2 + eps / T
    rv_0t = realized_variance(lr[:, :t_steps])
    v = swap_value(rv_0t, t, T, k_forward=0.25**2, k_var=k_var)
    mean, se = mean_and_stderr(v)
    assert abs(mean) < 3 * se + 1e-4
