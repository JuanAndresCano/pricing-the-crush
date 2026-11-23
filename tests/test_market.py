"""Fase de mercado: de cotizaciones bid/ask a forward, volatilidad implícita, strike y varianza de evento.

Se arma una cadena de mercado simulada con el modelo Merton-evento (precios exactos, spread del 2 %) y se
comprueba que el módulo recupera lo que el modelo conoce.
"""

import numpy as np
import pandas as pd
import pytest

from crush.data.source import EventData
from crush.market.chain import CleanParams, build_slices, clean_quotes, parity_forward
from crush.market.event_mkt import event_row, market_event, slice_table
from crush.market.strike_mkt import k_repl_discrete, k_repl_smooth
from crush.models.black_scholes import bs_price, implied_vol
from crush.models.event_models import MertonEvent
from crush.models.fourier import CallPricer

S0, R = 100.0, 0.04
MODEL = MertonEvent()
THETA = np.array([0.25, 1.0, -0.01, 0.05])  # sigma, Lambda_e, mu_e, sigma_e
PRE, POST = "2026-10-29", "2026-10-30"  # anuncio después del cierre del jueves 29 de octubre
EXPIRIES = {"2026-11-06": 6, "2026-11-13": 11, "2026-12-04": 25}  # días hábiles desde PRE


def _chain(day_offset: int, crosses: bool, spread: float = 0.02, stale_s: float = 10.0) -> pd.DataFrame:
    rows = []
    for exp, d in EXPIRIES.items():
        T = (d - day_offset) / 252.0
        F = S0 * np.exp(R * T)
        K = np.round(F * np.exp(np.linspace(-0.25, 0.25, 41)), 1)
        pricer = CallPricer(K)
        calls = MODEL.call_prices(THETA, F, T, R, crosses, pricer)
        puts = calls - np.exp(-R * T) * (F - K)
        for k, c, p in zip(K, calls, puts):
            for typ, px in (("call", c), ("put", p)):
                rows.append({
                    "option_ticker": f"X{exp}{typ}{k}", "underlying": "XYZ", "expiration_date": exp, "strike": float(k),
                    "option_type": typ, "days_to_expiry": d, "bid": px * (1 - spread / 2), "ask": px * (1 + spread / 2),
                    "bid_size": 10, "ask_size": 10, "quote_ts": 1, "quote_age_s": stale_s, "has_quote": True,
                })
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def event():
    meta = {"ticker": "XYZ", "announcement_date": PRE, "timing": "amc", "pre_day": PRE, "post_day": POST, "spot_pre": S0, "spot_post": S0}
    return EventData(_chain(0, True), _chain(1, False), meta)


def test_clean_quotes_flags_bad_quotes():
    df = _chain(0, True).head(6).copy()
    df.loc[0, "bid"] = 0.0  # bid cero
    df.loc[1, "ask"] = df.loc[1, "bid"] * 0.5  # cruzada
    df.loc[2, "ask"] = df.loc[2, "bid"] * 5  # spread enorme
    df.loc[3, "quote_age_s"] = 7200.0  # vieja
    df.loc[4, "has_quote"] = False
    valid = clean_quotes(df, CleanParams())["valid"].to_list()
    assert valid == [False, False, False, False, False, True]


def test_parity_forward_recovers_forward():
    T = 6 / 252
    F = S0 * np.exp(R * T)
    K = np.array([98.0, 99.0, 100.0, 101.0, 102.0])
    c = np.array(bs_price(F, K, T, 0.3, R, True))
    p = np.array(bs_price(F, K, T, 0.3, R, False))
    f, n = parity_forward(pd.DataFrame({"strike": K, "mid": c}), pd.DataFrame({"strike": K, "mid": p}), S0, T, R)
    assert n == 5 and f == pytest.approx(F, abs=1e-9)


def test_slices_forward_and_atm_vol(event):
    s = build_slices(event.pre, PRE, S0, R)
    assert [x.expiry for x in s] == list(EXPIRIES)
    for x in s:
        assert x.F == pytest.approx(S0 * np.exp(R * x.T), rel=2e-4)
        # la volatilidad implícita ATM del modelo, calculada sin pasar por el módulo
        pricer = CallPricer(np.array([x.F]))
        c = MODEL.call_prices(THETA, x.F, x.T, R, True, pricer)[0]
        assert x.atm_iv == pytest.approx(implied_vol(c, x.F, x.F, x.T, R, True), abs=3e-3)


def test_strike_matches_model_and_tail_matters(event):
    x = build_slices(event.pre, PRE, S0, R)[0]
    model = MODEL.k_repl(THETA, x.T, True)
    ext = k_repl_smooth(x.K, x.iv, x.T, R, x.F, extrapolate=True)
    trunc = k_repl_smooth(x.K, x.iv, x.T, R, x.F, extrapolate=False)
    disc = k_repl_discrete(x.K, x.iv, x.T, R, x.F)
    assert ext == pytest.approx(model, rel=0.03)
    assert trunc < ext  # sin la cola se subestima
    assert disc == pytest.approx(trunc, rel=0.03)


def test_event_variance_from_market_matches_model(event):
    pre = build_slices(event.pre, PRE, S0, R)
    ev = market_event(pre[0], pre[1], "strike")
    assert ev.eps == pytest.approx(MODEL.eps_repl(THETA), rel=0.25)
    assert ev.sigma_b == pytest.approx(THETA[0], rel=0.05)


def test_no_event_in_post_chain(event):
    row = event_row(event, R)
    assert abs(row["eps_post_control"]) < 0.1 * row["eps_strike"]
    assert row["cr_obs"] < 1.0  # la volatilidad cae al pasar el anuncio
    assert 0 < row["cr_split_strike"] < 1.0


def test_slice_table_shape(event):
    t = slice_table(event, R)
    assert set(t["when"]) == {"pre", "post"} and len(t) == 6
    assert (t["k_repl_extrap"] >= t["k_repl_trunc"]).all()
