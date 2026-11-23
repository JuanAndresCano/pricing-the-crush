"""ThetaData: control de ritmo, guardado de respuestas crudas y descarga de un evento con un servidor falso."""

import json

import numpy as np
import pandas as pd
import pytest
import requests

from crush.data.rawcache import DefinitiveError, RawCache
from crush.data.source import download_and_cache, load_event
from crush.data.theta_client import ThetaClient, ThetaError, ThetaSource, occ_ticker
from crush.data.throttle import RateLimiter


class Clock:
    def __init__(self):
        self.t = 0.0
        self.sleeps = []

    def now(self):
        return self.t

    def sleep(self, s):
        self.sleeps.append(s)
        self.t += s


class Resp:
    def __init__(self, status=200, text=""):
        self.status_code, self.text = status, text


class FakeSession:
    """Responde según el endpoint y cuenta los pedidos."""

    def __init__(self, handler):
        self.handler, self.calls = handler, []

    def get(self, url, params=None, timeout=None):
        self.calls.append((url, dict(params)))
        out = self.handler(url, params)
        if isinstance(out, Exception):
            raise out
        return out


def _client(tmp_path, handler, clock=None, **kw):
    clock = clock or Clock()
    cache = RawCache(tmp_path, "theta")
    limiter = RateLimiter(3.0, clock=clock.now, sleep=clock.sleep)
    session = FakeSession(handler)
    return ThetaClient("http://x/v3", cache, limiter, session, sleep=clock.sleep, **kw), session, clock


def test_rate_limiter_spaces_requests():
    c = Clock()
    lim = RateLimiter(3.0, clock=c.now, sleep=c.sleep)
    for _ in range(4):
        lim.wait()
    assert c.t == pytest.approx(9.0) and len(c.sleeps) == 3


def test_raw_cache_hits_and_errors(tmp_path):
    cache = RawCache(tmp_path, "t")
    n = {"calls": 0}

    def ok():
        n["calls"] += 1
        return "a,b\n1,2\n"

    assert cache.fetch("e", {"symbol": "X", "start_date": "20260101"}, ok) == "a,b\n1,2\n"
    assert cache.fetch("e", {"symbol": "X", "start_date": "20260101"}, ok) == "a,b\n1,2\n"
    assert n["calls"] == 1
    lines = [json.loads(x) for x in (tmp_path / "t" / "index.jsonl").read_text().splitlines()]
    assert lines[0]["status"] == "ok" and lines[0]["bytes"] > 0

    def nodata():
        n["calls"] += 1
        raise DefinitiveError(472, "no data")

    for _ in range(2):
        with pytest.raises(DefinitiveError):
            cache.fetch("e", {"symbol": "Y"}, nodata)
    assert n["calls"] == 2  # el segundo error salió del disco
    assert cache.clear_errors() == 1


def test_raw_cache_does_not_store_transient_failures(tmp_path):
    cache = RawCache(tmp_path, "t")

    def boom():
        raise ThetaError("caída")

    with pytest.raises(ThetaError):
        cache.fetch("e", {"symbol": "Z"}, boom)
    assert cache.fetch("e", {"symbol": "Z"}, lambda: "x\n1\n") == "x\n1\n"


def test_client_retries_429_then_succeeds(tmp_path):
    seq = iter([Resp(429, "slow"), Resp(503, "busy"), Resp(200, "close\n10\n")])
    client, session, clock = _client(tmp_path, lambda u, p: next(seq))
    df = client.stock_eod("AAPL", "2026-10-29")
    assert df["close"].iloc[0] == 10 and len(session.calls) == 3
    assert clock.sleeps  # hubo esperas entre reintentos


def test_client_connection_error_message(tmp_path):
    client, _, _ = _client(tmp_path, lambda u, p: requests.ConnectionError("refused"))
    with pytest.raises(ThetaError, match="Theta Terminal"):
        client.stock_eod("AAPL", "2026-10-29")


def test_client_definitive_error_is_cached(tmp_path):
    client, session, _ = _client(tmp_path, lambda u, p: Resp(472, "No data found"))
    for _ in range(2):
        with pytest.raises(DefinitiveError):
            client.stock_eod("AAPL", "2026-10-29")
    assert len(session.calls) == 1


def test_occ_ticker():
    assert occ_ticker("AAPL", "2026-11-06", "c", 270.0) == "AAPL261106C00270000"


# --- un evento completo -----------------------------------------------------------------------------------

PRE, POST = "2026-10-29", "2026-10-30"
EXPIRIES = ["2026-11-06", "2026-11-13", "2026-12-04"]


def _option_csv(day_shift: float) -> str:
    rows = []
    for exp in EXPIRIES:
        for k in np.arange(70.0, 130.1, 2.5):
            for right in ("call", "put"):
                mid = max(abs(100 - k) * 0.5, 0.5) + 1.0 + day_shift
                rows.append({"symbol": "XYZ", "expiration": exp, "strike": k, "right": right, "bid": mid - 0.05, "ask": mid + 0.05, "bid_size": 5, "ask_size": 5})
    return pd.DataFrame(rows).to_csv(index=False)


def _handler(url, params):
    if url.endswith("/stock/history/eod"):
        return Resp(200, "close\n100.0\n")
    return Resp(200, _option_csv(0.0 if params["start_date"] == "20261029" else -0.3))


def test_download_event_with_theta_uses_one_request_per_day_and_caches(tmp_path):
    client, session, _ = _client(tmp_path / "api", _handler)
    src = ThetaSource(client)
    path, status = download_and_cache(src, "XYZ", PRE, "amc", root=tmp_path / "raw", workers=4)
    assert status == "downloaded"
    assert len(session.calls) == 6  # acción, tasa y cadena, antes y después (el servidor falso no tiene tasas: la tasa queda en None)
    ev = load_event(path)
    assert ev.meta["n_contracts"] > 0 and ev.meta["n_quotes_pre"] == ev.meta["n_contracts"]
    assert ev.pre["quote_age_s"].isna().all() and ev.pre["has_quote"].all()
    assert (ev.post["bid"] < ev.pre["bid"]).all()

    # una segunda fuente nueva, sin tocar la red: todo sale de las respuestas crudas guardadas
    client2, session2, _ = _client(tmp_path / "api", lambda u, p: pytest.fail("no debía pedir nada"))
    path2, status2 = download_and_cache(ThetaSource(client2), "XYZ", PRE, "amc", root=tmp_path / "raw2", workers=4)
    assert status2 == "downloaded" and not session2.calls
    pd.testing.assert_frame_equal(load_event(path2).pre, ev.pre)


def test_pipeline_runs_on_theta_event(tmp_path):
    """Lo descargado con ThetaSource entra directo a la fase de mercado (sin hora de cotización)."""
    from crush.market.chain import build_slices

    client, _, _ = _client(tmp_path / "api", _handler)
    path, _ = download_and_cache(ThetaSource(client), "XYZ", PRE, "amc", root=tmp_path / "raw", workers=2)
    ev = load_event(path)
    slices = build_slices(ev.pre, ev.meta["pre_day"], ev.meta["spot_pre"], 0.04)
    assert slices and all(np.isfinite(s.F) for s in slices)
