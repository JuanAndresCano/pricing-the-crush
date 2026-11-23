"""Tasa libre de riesgo por evento: lectura de ThetaData, respaldo y uso en la fase de mercado."""

import pytest

from crush.data.rawcache import DefinitiveError
from crush.data.source import download_and_cache, load_event
from crush.data.theta_client import ThetaSource
from crush.market.event_mkt import slice_table
from crush.market.rates import DEFAULT_RATE, event_rates
from tests.test_theta import PRE, Resp, _client, _handler


def _with_rates(url, params):
    if "interest_rate" in url:
        # incluye fin de semana, como el servicio real, y un día posterior al pedido que no debe usarse
        return Resp(200, 'created,rate\n"2026-10-27",4.00\n"2026-10-28",4.10\n"2026-10-29",4.20\n"2026-10-30",9.99\n')
    return _handler(url, params)


def test_event_rates_fallback():
    assert event_rates({}) == (DEFAULT_RATE, DEFAULT_RATE)
    assert event_rates({"r_pre": 0.05}, default=0.03) == (0.05, 0.03)


def test_rate_takes_last_day_not_after_request(tmp_path):
    client, session, _ = _client(tmp_path, _with_rates)
    src = ThetaSource(client)
    assert src.rate("2026-10-29") == pytest.approx(0.042)  # 4.20 %, no el 9.99 del día siguiente
    assert src.rate("2026-10-28") == pytest.approx(0.041)


def test_rate_none_when_plan_has_no_access(tmp_path):
    client, _, _ = _client(tmp_path, lambda u, p: Resp(400, "Requesting Interest Rate data before 2024-01-01 with FREE subscription."))
    assert ThetaSource(client).rate("2023-08-01") is None


def test_event_meta_has_rates_and_market_uses_them(tmp_path):
    client, _, _ = _client(tmp_path / "api", _with_rates)
    path, _ = download_and_cache(ThetaSource(client), "XYZ", PRE, "amc", root=tmp_path / "raw", workers=1)
    ev = load_event(path)
    assert ev.meta["r_pre"] == pytest.approx(0.042)
    t_default = slice_table(ev, 0.04)
    ev.meta["r_pre"] = ev.meta["r_post"] = None  # sin tasas en la meta: se usa la de respaldo
    t_fallback = slice_table(ev, 0.0)
    assert (t_default["atm_iv"] != t_fallback["atm_iv"]).all()  # la tasa entra en el precio de Black-Scholes


def test_source_without_rates_keeps_working(tmp_path):
    client, _, _ = _client(tmp_path / "api", _handler)  # el servidor falso no tiene endpoint de tasas: responde con opciones
    path, _ = download_and_cache(ThetaSource(client), "XYZ", PRE, "amc", root=tmp_path / "raw", workers=1)
    assert load_event(path).meta["n_contracts"] > 0
