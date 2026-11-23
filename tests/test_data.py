import json
from datetime import date

import pandas as pd
import pytest

from crush.data import calendar as cal
from crush.data.massive_client import MassiveClient, MassiveError
from crush.data.selection import SelectionParams, select_contracts
from crush.data.source import DataError, download_and_cache, download_event, load_event

NS_PER_H = 3_600_000_000_000


# ---------- calendario ----------


def test_pre_post_days_before_open_and_after_close():
    # bmo: la reacción está en el cierre del mismo día. amc: en el cierre del día siguiente
    assert cal.pre_post_days("2026-10-27", "bmo") == (date(2026, 10, 26), date(2026, 10, 27))
    assert cal.pre_post_days("2026-10-27", "amc") == (date(2026, 10, 27), date(2026, 10, 28))


def test_pre_post_days_skip_weekends_and_holidays():
    assert cal.pre_post_days("2026-10-26", "bmo")[0] == date(2026, 10, 23)  # lunes -> viernes anterior
    assert cal.pre_post_days("2026-10-30", "amc")[1] == date(2026, 11, 2)  # viernes -> lunes
    assert cal.pre_post_days("2026-11-25", "amc")[1] == date(2026, 11, 27)  # Acción de Gracias el 26: cerrado
    assert cal.pre_post_days("2026-11-27", "bmo")[0] == date(2026, 11, 25)


def test_announcement_on_a_non_trading_day_is_rejected():
    with pytest.raises(ValueError):
        cal.pre_post_days("2026-11-26", "bmo")
    with pytest.raises(ValueError):
        cal.pre_post_days("2026-10-31", "amc")  # sábado
    with pytest.raises(ValueError):
        cal.pre_post_days("2026-10-27", "dmh")


def test_trading_days_between_counts_sessions_after_the_start():
    assert cal.trading_days_between("2026-10-26", "2026-10-27") == 1
    assert cal.trading_days_between("2026-10-23", "2026-10-26") == 1  # viernes a lunes
    assert cal.trading_days_between("2026-11-25", "2026-11-27") == 1  # salta el feriado
    assert cal.trading_days_between("2026-10-26", "2026-10-26") == 0


def test_market_close_in_utc_follows_daylight_saving_and_half_days():
    assert cal.market_close_ns("2026-11-20") == pd.Timestamp("2026-11-20 21:00", tz="UTC").value  # EST
    assert cal.market_close_ns("2026-07-02") == pd.Timestamp("2026-07-02 20:00", tz="UTC").value  # EDT
    assert cal.market_close_ns("2026-11-27") == pd.Timestamp("2026-11-27 18:00", tz="UTC").value  # medio día


def test_load_events_validates_the_calendar(tmp_path):
    good = tmp_path / "ok.csv"
    good.write_text(
        "ticker,announcement_date,timing,source_a,source_b,verified\naapl,2026-10-29,amc,a,b,true\n", encoding="utf-8"
    )
    df = cal.load_events(good)
    assert df.loc[0, "ticker"] == "AAPL" and df.loc[0, "announcement_date"] == date(2026, 10, 29)

    bad = tmp_path / "bad.csv"
    bad.write_text(
        "ticker,announcement_date,timing,source_a,source_b,verified\n"
        "AAPL,2026-10-31,amc,,,\nMSFT,2026-10-28,xyz,,,\nMSFT,2026-10-28,bmo,,,\nMSFT,2026-10-28,bmo,,,\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError) as exc:
        cal.load_events(bad)
    msg = str(exc.value)
    assert "no es día hábil" in msg and "no es bmo ni amc" in msg and "repetido" in msg


def test_repo_events_file_has_the_expected_header():
    from crush.config import ROOT

    assert (ROOT / "events" / "earnings_events.csv").read_text(encoding="utf-8").startswith(",".join(cal.EVENT_COLUMNS))


# ---------- selección de contratos ----------

PRE, POST = date(2026, 10, 26), date(2026, 10, 27)  # anuncio bmo el 27 de octubre
EXPIRIES = ["2026-10-30", "2026-11-06", "2026-11-13", "2026-11-20", "2026-11-27", "2026-12-18"]


def fake_contracts():
    rows = []
    for e in EXPIRIES + ["2026-10-27"]:  # un vencimiento que vence el mismo día del anuncio
        for k in range(60, 141, 5):
            for t in ("call", "put"):
                rows.append({
                    "ticker": f"O:XYZ{e.replace('-', '')}{t[0].upper()}{k:05d}",
                    "underlying_ticker": "XYZ", "contract_type": t, "expiration_date": e,
                    "strike_price": float(k), "shares_per_contract": 100,
                })
    rows.append({"ticker": "O:XYZ-ADJ", "underlying_ticker": "XYZ", "contract_type": "call",
                 "expiration_date": "2026-11-06", "strike_price": 100.0, "shares_per_contract": 50})
    return rows


def test_selection_picks_near_expiries_plus_one_far_expiry():
    sel = select_contracts(fake_contracts(), 100.0, PRE, POST, SelectionParams())
    expiries = sorted(str(e) for e in sel["expiry"].unique())
    # Los 2 más cercanos que cruzan el anuncio con al menos 3 días hábiles, y el primero de 21 a 60 días
    assert expiries[:2] == ["2026-10-30", "2026-11-06"]
    assert len(expiries) == 3 and expiries[2] in EXPIRIES[3:]
    assert cal.trading_days_between(PRE, date.fromisoformat(expiries[2])) >= 21
    assert "2026-10-27" not in expiries  # vence con el anuncio: demasiado corto
    assert "O:XYZ-ADJ" not in set(sel["ticker"])  # contratos no estándar fuera


def test_selection_keeps_otm_wings_and_both_sides_near_the_money():
    sel = select_contracts(fake_contracts(), 100.0, PRE, POST, SelectionParams(max_abs_logm=0.30, atm_band=0.05))
    assert sel["log_moneyness"].abs().max() <= 0.30
    wings = sel[sel["log_moneyness"].abs() > 0.05]
    assert ((wings["contract_type"] == "put") == (wings["log_moneyness"] < 0)).all()  # solo OTM
    atm = sel[sel["log_moneyness"].abs() <= 0.05]
    assert set(atm["contract_type"]) == {"call", "put"}


def test_selection_is_empty_when_nothing_crosses_the_announcement():
    contracts = [c for c in fake_contracts() if c["expiration_date"] == "2026-10-27"]
    assert select_contracts(contracts, 100.0, PRE, POST).empty


# ---------- descarga con una fuente simulada ----------


class FakeSource:
    name = "fake"

    def __init__(self):
        self.calls = {"spot": 0, "contracts": 0, "quote": 0}
        self.no_quote = set()

    def spot(self, ticker, day):
        self.calls["spot"] += 1
        return 100.0 if day == PRE else 92.0

    def list_contracts(self, ticker, as_of):
        self.calls["contracts"] += 1
        assert as_of == PRE  # los contratos se eligen con la cadena del cierre previo
        return fake_contracts()

    def last_quote(self, option_ticker, day):
        self.calls["quote"] += 1
        if option_ticker in self.no_quote:
            return None
        close = cal.market_close_ns(day)
        return {"bid_price": 1.0, "ask_price": 1.2, "bid_size": 10, "ask_size": 12, "sip_timestamp": close - 30 * 10**9}


def test_download_event_builds_pre_and_post_chains_over_the_same_contracts():
    src = FakeSource()
    data = download_event(src, "XYZ", date(2026, 10, 27), "bmo", workers=2)
    assert list(data.pre["option_ticker"]) == list(data.post["option_ticker"])
    assert data.meta["pre_day"] == "2026-10-26" and data.meta["post_day"] == "2026-10-27"
    assert data.meta["spot_pre"] == 100.0 and data.meta["spot_post"] == 92.0
    assert data.pre["has_quote"].all() and (data.pre["quote_age_s"] == 30).all()
    assert data.meta["n_quotes_pre"] == data.meta["n_quotes_post"] == data.meta["n_contracts"]
    assert src.calls["contracts"] == 1 and src.calls["quote"] == 2 * len(data.pre)


def test_missing_quotes_are_kept_but_flagged():
    src = FakeSource()
    first = download_event(src, "XYZ", date(2026, 10, 27), "bmo").pre["option_ticker"].iloc[0]
    src.no_quote = {first}
    data = download_event(src, "XYZ", date(2026, 10, 27), "bmo")
    row = data.pre[data.pre["option_ticker"] == first].iloc[0]
    assert not row["has_quote"] and pd.isna(row["bid"])
    assert data.meta["n_quotes_pre"] == data.meta["n_contracts"] - 1


def test_download_fails_clearly_without_stock_price():
    src = FakeSource()
    src.spot = lambda t, d: None
    with pytest.raises(DataError):
        download_event(src, "XYZ", date(2026, 10, 27), "bmo")


def test_cache_does_not_call_the_api_twice(tmp_path):
    src = FakeSource()
    path, status = download_and_cache(src, "XYZ", date(2026, 10, 27), "bmo", root=tmp_path)
    assert status == "downloaded" and (path / "pre.parquet").exists() and (path / "meta.json").exists()
    calls = dict(src.calls)
    _, status2 = download_and_cache(src, "XYZ", date(2026, 10, 27), "bmo", root=tmp_path)
    assert status2 == "cached" and src.calls == calls
    _, status3 = download_and_cache(src, "XYZ", date(2026, 10, 27), "bmo", root=tmp_path, force=True)
    assert status3 == "downloaded" and src.calls["contracts"] == 2
    loaded = load_event(path)
    assert len(loaded.pre) == len(loaded.post) == loaded.meta["n_contracts"]
    assert json.loads((path / "meta.json").read_text(encoding="utf-8"))["source"] == "fake"


# ---------- cliente de Massive ----------


class FakeResponse:
    def __init__(self, status_code, payload):
        self.status_code, self._payload, self.text = status_code, payload, str(payload)

    def json(self):
        return self._payload


class FakeSession:
    def __init__(self, responses):
        self.responses, self.calls = list(responses), []

    def get(self, url, params=None, headers=None, timeout=None):
        self.calls.append({"url": url, "params": params, "headers": headers})
        return self.responses.pop(0)


def test_client_retries_on_rate_limit_then_succeeds(monkeypatch):
    sleeps = []
    monkeypatch.setattr("crush.data.massive_client.time.sleep", sleeps.append)
    session = FakeSession([FakeResponse(429, {}), FakeResponse(503, {}), FakeResponse(200, {"results": [{"c": 101.5}]})])
    client = MassiveClient(api_key="k", base_url="https://x.test", session=session, backoff=0.5)
    assert client.stock_close("XYZ", "2026-10-26") == 101.5
    assert sleeps == [0.5, 1.0]  # espera exponencial
    assert "adjusted" in session.calls[-1]["params"] and session.calls[-1]["params"]["adjusted"] == "false"


def test_client_does_not_retry_on_forbidden_plan_error():
    session = FakeSession([FakeResponse(403, {"message": "plan"})])
    client = MassiveClient(api_key="SECRET", base_url="https://x.test", session=session)
    with pytest.raises(MassiveError) as exc:
        client.option_last_quote("O:XYZ261120C00100000", date(2026, 10, 26))
    assert len(session.calls) == 1 and "SECRET" not in str(exc.value)


def test_last_quote_asks_for_one_result_up_to_the_real_close():
    session = FakeSession([FakeResponse(200, {"results": [{"bid_price": 1.0, "ask_price": 1.1}]})])
    client = MassiveClient(api_key="k", base_url="https://x.test", session=session)
    q = client.option_last_quote("O:XYZ261120C00100000", date(2026, 11, 27))  # medio día
    p = session.calls[0]["params"]
    assert q["bid_price"] == 1.0
    assert (p["order"], p["sort"], p["limit"]) == ("desc", "timestamp", 1)
    assert p["timestamp.lte"] == pd.Timestamp("2026-11-27 18:00", tz="UTC").value
    assert p["timestamp.gte"] < p["timestamp.lte"]
    assert session.calls[0]["headers"] == {"Authorization": "Bearer k"}


def test_contracts_include_expired_ones_without_duplicates():
    session = FakeSession([
        FakeResponse(200, {"results": [{"ticker": "A"}, {"ticker": "B"}]}),
        FakeResponse(200, {"results": [{"ticker": "B"}, {"ticker": "C"}]}),
    ])
    client = MassiveClient(api_key="k", base_url="https://x.test", session=session)
    out = client.option_contracts_incl_expired("XYZ", "2026-10-26")
    assert sorted(c["ticker"] for c in out) == ["A", "B", "C"]
    assert [c["params"]["expired"] for c in session.calls] == ["false", "true"]
