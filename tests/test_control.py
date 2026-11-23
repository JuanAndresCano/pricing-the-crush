"""Activo de control (SPY): descarga compartida y descuento del movimiento general del mercado."""

import numpy as np
import pandas as pd
import pytest

from crush.analysis.crush import control_slices, estimate_control_beta, path_table
from crush.data.control import covers_spot, download_control, ensure_control_day, event_days, load_control
from crush.market.chain import build_slices
from crush.models.event_models import MertonEvent
from tests.test_extra_days import MERTON, _chain, _row, _world
from tests.test_market import POST, PRE, R, S0

# Factor de volatilidad del mercado en cada día hábil d desde PRE: el mercado entero sube y baja ±10 %
MARKET = {0: 1.0, 1: 0.95, 2: 0.90, 3: 1.12, 5: 0.88}
DAY = {0: PRE, 1: POST, 2: "2026-11-02", 3: "2026-11-03", 5: "2026-11-05"}


def _theta(sigma):
    t = MERTON.copy()
    t[0] = sigma
    return t


def test_event_days_lists_pre_post_and_extras():
    meta = {"pre_day": PRE, "post_day": POST, "extra_days": {"4": {"day": "2026-11-05"}, "1": {"day": "2026-11-02"}}}
    assert event_days(meta) == [PRE, POST, "2026-11-02", "2026-11-05"]


class FakeSpy:
    name = "fake"

    def __init__(self):
        self.list_calls = 0

    def spot(self, symbol, day):
        return 500.0

    def list_contracts(self, symbol, day):
        self.list_calls += 1
        rows = [{"ticker": f"SPY{e}{t}{k}", "contract_type": t, "expiration_date": e, "strike_price": float(k), "shares_per_contract": 100}
                for e in ("2026-10-28", "2026-11-13") for t in ("call", "put") for k in (495, 500, 505)]
        return rows

    def last_quote(self, option_ticker, day):
        return {"bid_price": 2.0, "ask_price": 2.1, "bid_size": 1, "ask_size": 1, "sip_timestamp": None}

    def rate(self, day):
        return 0.04


def test_control_day_is_stored_once_and_drops_expired_contracts(tmp_path):
    src = FakeSpy()
    f = ensure_control_day(src, "SPY", PRE, tmp_path)
    assert f is not None and src.list_calls == 1
    ensure_control_day(src, "SPY", PRE, tmp_path)  # ya está: sin pedidos nuevos
    assert src.list_calls == 1
    loaded = load_control(tmp_path, "fake", "SPY", [PRE, POST])
    assert list(loaded) == [PRE]  # POST no se descargó
    df, spot, r = loaded[PRE]
    assert spot == 500.0 and r == 0.04
    assert set(df["expiration_date"].astype(str)) == {"2026-11-13"}  # el contrato vencido el 28-oct no entra a una cadena del 29-oct


def test_download_control_reports_each_day(tmp_path):
    src = FakeSpy()
    meta = {"pre_day": PRE, "post_day": POST, "extra_days": {}}
    assert download_control(src, meta, "SPY", tmp_path) == {PRE: True, POST: True}


def _shock_world():
    """Una acción cuya volatilidad solo sigue al mercado después del anuncio, y SPY con el mismo factor de mercado."""
    data, extra = _world(MertonEvent(), MERTON)
    m = MertonEvent()
    data.post = _chain(m, _theta(0.25 * MARKET[1]), 1, False)
    extra = {k: (_chain(m, _theta(0.25 * MARKET[d]), d, False), DAY[d], S0, None) for k, d in {1: 2, 2: 3, 4: 5}.items()}
    ctrl = {}
    for d, day in DAY.items():
        spy = _chain(m, _theta(0.18 * MARKET[d]), d, False)
        ctrl[day] = {s.expiry: s for s in build_slices(spy, day, S0, R)}
    return data, extra, ctrl


def test_market_shock_is_removed_by_control():
    data, extra, ctrl = _shock_world()
    raw = path_table(data, extra, _row(), R)
    adj = path_table(data, extra, _row(), R, control=ctrl)
    far = lambda t: t[(t["model"] == "merton") & (t["k"] > 0) & (t["expiry"] == "2026-12-04")]
    assert far(raw)["g_gap"].abs().max() > 0.05  # sin control, el movimiento del mercado parece crush gradual
    assert far(adj)["g_gap_adj"].abs().max() < 0.01  # con control desaparece
    assert far(adj)["g_spy"].between(0.85, 1.20).all() and (far(adj)["g_spy"] - 1).abs().max() > 0.05


def test_without_control_adjusted_columns_are_nan():
    data, extra, _ = _shock_world()
    t = path_table(data, extra, _row(), R)
    assert t["g_gap_adj"].isna().all() and t["cr_obs_adj"].isna().all()
    partial = path_table(data, extra, _row(), R, control={PRE: {}})  # control incompleto: no se inventa nada
    assert partial["g_gap_adj"].isna().all()


def test_control_slices_builds_from_loaded_frames(tmp_path):
    spy = _chain(MertonEvent(), _theta(0.18), 0, False)
    ctrl = control_slices({PRE: (spy, S0, None)}, 0.04)
    assert set(ctrl[PRE]) >= {"2026-11-06", "2026-12-04"}


class FarSpy(FakeSpy):
    """Como ThetaData cuando el activo se mueve más que la ventana: los strikes quedan lejos del cierre."""

    def spot(self, symbol, day):
        return 560.0  # los strikes de FakeSpy van de 495 a 505


def test_chain_that_misses_the_money_is_rejected_and_not_stored(tmp_path):
    assert ensure_control_day(FarSpy(), "SPY", PRE, tmp_path) is None
    assert load_control(tmp_path, "fake", "SPY", [PRE]) == {}
    assert download_control(FarSpy(), {"pre_day": PRE, "post_day": POST, "extra_days": {}}, "SPY", tmp_path) == {PRE: False, POST: False}


def test_covers_spot():
    df = pd.DataFrame({"strike": [495.0, 500.0, 505.0]})
    assert covers_spot(df, 501.0)
    assert not covers_spot(df, 560.0)  # todos los strikes por debajo
    assert not covers_spot(pd.DataFrame({"strike": [495.0, 520.0]}), 507.0)  # hay strikes a ambos lados pero lejos


def _beta_world(beta: float):
    """La acción reacciona al mercado con sensibilidad `beta` (en volatilidad), no 1."""
    data, extra, ctrl = _shock_world()
    m = MertonEvent()
    data.post = _chain(m, _theta(0.25 * MARKET[1] ** beta), 1, False)
    extra = {k: (_chain(m, _theta(0.25 * MARKET[d] ** beta), d, False), DAY[d], S0, None) for k, d in {1: 2, 2: 3, 4: 5}.items()}
    return data, extra, ctrl


def test_unit_beta_overcorrects_and_estimated_beta_fixes_it():
    data, extra, ctrl = _beta_world(0.4)
    far = lambda t: t[(t["model"] == "merton") & (t["k"] > 0) & (t["expiry"] == "2026-12-04")]
    unit = path_table(data, extra, _row(), R, control=ctrl)  # beta 1
    assert far(unit)["g_gap_adj"].abs().max() > 0.03  # sobrecorrige
    est = estimate_control_beta(unit)
    assert est["beta"] == pytest.approx(0.4, abs=0.1) and est["n"] >= 3
    assert abs(est["alpha"]) < 0.02  # sin crush gradual extra: la constante es cero
    fixed = path_table(data, extra, _row(), R, control=ctrl, control_beta=est["beta"])
    assert far(fixed)["g_gap_adj"].abs().max() < 0.015
    assert far(fixed)["cr_obs_adj"].notna().all()


def test_estimate_control_beta_needs_data():
    data, extra, _ = _shock_world()
    no_control = path_table(data, extra, _row(), R)
    assert np.isnan(estimate_control_beta(no_control)["beta"])
