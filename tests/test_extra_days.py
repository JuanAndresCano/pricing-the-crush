"""Días posteriores al anuncio: descarga reanudable y trayectoria de la volatilidad contra el modelo."""

import json
from datetime import date

import numpy as np
import pandas as pd
import pytest

from crush.analysis.crush import evolve_state, path_table, summarize_path
from crush.data.extra_days import download_extra_days, load_extra_days
from crush.data.source import EventData, save_event
from crush.models.event_models import BatesEvent, MertonEvent
from crush.models.fourier import CallPricer
from tests.test_market import EXPIRIES, POST, PRE, R, S0

MERTON = np.array([0.25, 1.0, -0.01, 0.05])
BATES = np.array([0.16, 20.0, 0.04, 0.3, -0.3, 1.0, -0.01, 0.05])  # v0 muy sobre theta: la varianza baja sola
OFFSETS = {0: 1, 1: 2, 2: 3, 4: 5}  # día extra k -> días hábiles desde PRE (post = 1)


def _chain(model, theta, d: int, crosses: bool, spread: float = 0.02) -> pd.DataFrame:
    """Cadena de mercado `d` días hábiles después de PRE, con el estado del modelo evolucionado d días."""
    th = evolve_state(model, theta, d / 252.0) if d else np.array(theta)
    rows = []
    for exp, days in EXPIRIES.items():
        T = (days - d) / 252.0
        F = S0 * np.exp(R * T)
        K = np.round(F * np.exp(np.linspace(-0.25, 0.25, 41)), 1)
        calls = model.call_prices(th, F, T, R, crosses, CallPricer(K))
        puts = calls - np.exp(-R * T) * (F - K)
        for k, c, p in zip(K, calls, puts):
            for typ, px in (("call", c), ("put", p)):
                rows.append({
                    "option_ticker": f"X{exp}{typ}{k}", "underlying": "XYZ", "expiration_date": exp, "strike": float(k), "option_type": typ,
                    "days_to_expiry": days, "bid": px * (1 - spread / 2), "ask": px * (1 + spread / 2), "bid_size": 5, "ask_size": 5,
                    "quote_ts": None, "quote_age_s": None, "has_quote": True,
                })
    return pd.DataFrame(rows)


def _world(model, theta):
    meta = {"ticker": "XYZ", "announcement_date": PRE, "timing": "amc", "pre_day": PRE, "post_day": POST, "spot_pre": S0, "spot_post": S0}
    data = EventData(_chain(model, theta, 0, True), _chain(model, theta, 1, False), meta)
    extra = {k: (_chain(model, theta, d, False), f"2026-11-{d:02d}", S0, None) for k, d in {1: 2, 2: 3, 4: 5}.items()}
    return data, extra


def _row():
    row = {f"merton_{n}": v for n, v in zip(MertonEvent().names, MERTON)}
    row.update({f"bates_{n}": v for n, v in zip(BatesEvent().names, BATES)})
    row.update(ticker="XYZ", announcement_date=PRE)
    return row


def test_path_table_covers_days_and_drops_short_expiries():
    data, extra = _world(MertonEvent(), MERTON)
    t = path_table(data, extra, _row(), R)
    assert set(t["k"]) == {0, 1, 2, 4}
    assert (t["days_left"] >= 3).all()
    near = t[t["expiry"] == "2026-11-06"]
    assert set(near["k"]) == {0, 1, 2}  # el 5-nov (k=4) al vencimiento más cercano le queda 1 día: se descarta
    assert set(t[t["k"] == 4]["expiry"]) == {"2026-11-13", "2026-12-04"}
    assert set(t["model"]) == {"merton", "bates"}


def test_instantaneous_world_has_flat_path_in_merton():
    """Si el mercado lo generó Merton-evento, la volatilidad no baja después del día 0 y Merton acierta."""
    data, extra = _world(MertonEvent(), MERTON)
    t = path_table(data, extra, _row(), R)
    m = t[t["model"] == "merton"]
    assert m["g_gap"].abs().max() < 0.01
    assert (m["cr_obs"] - m["cr_model"]).abs().max() < 0.01


def test_gradual_world_is_missed_by_merton_and_caught_by_bates():
    """Si el mercado lo generó Bates con v0 >> theta, la volatilidad sigue bajando: Merton se equivoca y Bates acierta."""
    data, extra = _world(BatesEvent(), BATES)
    t = path_table(data, extra, _row(), R)
    far = t[(t["k"] > 0) & (t["expiry"] == "2026-12-04")]
    assert (far["g_obs"] < 1.0).all()  # sigue bajando después del día 0
    bates = far[far["model"] == "bates"]
    merton = far[far["model"] == "merton"]
    assert bates["g_gap"].abs().max() < 0.005
    assert (merton["g_gap"] < -0.005).all()  # Merton no espera esa caída
    s = summarize_path(t)
    assert set(s["model"]) == {"merton", "bates"}


# --- descarga ---------------------------------------------------------------------------------------------


class FakeSource:
    name = "fake"

    def __init__(self):
        self.spot_calls = self.quote_calls = 0

    def spot(self, ticker, day):
        self.spot_calls += 1
        return 100.0

    def last_quote(self, option_ticker, day):
        self.quote_calls += 1
        return {"bid_price": 1.0, "ask_price": 1.1, "bid_size": 1, "ask_size": 1, "sip_timestamp": None}

    def rate(self, day):
        return 0.04


def _saved_event(tmp_path):
    meta = {"ticker": "XYZ", "announcement_date": PRE, "timing": "amc", "pre_day": PRE, "post_day": POST, "spot_pre": S0, "spot_post": S0}
    path = tmp_path / "XYZ_2026-10-29"
    save_event(path, EventData(_chain(MertonEvent(), MERTON, 0, True), _chain(MertonEvent(), MERTON, 1, False), meta))
    return path


def test_download_extra_days_is_additive_resumable_and_skips_future(tmp_path):
    path = _saved_event(tmp_path)
    pre_before = pd.read_parquet(path / "pre.parquet")
    src = FakeSource()
    # PRE = jueves 29-oct, POST = viernes 30-oct; k=1 -> lunes 2-nov, k=2 -> martes 3-nov, k=4 -> jueves 5-nov
    extra = download_extra_days(src, path, offsets=(1, 2, 4), today=date(2026, 11, 4))
    assert sorted(extra) == ["1", "2"]  # el 5-nov todavía no existe a fecha de 4-nov
    assert (path / "post_p1.parquet").exists() and not (path / "post_p4.parquet").exists()
    n_quotes = src.quote_calls

    download_extra_days(src, path, offsets=(1, 2), today=date(2026, 11, 4))  # ya está todo: sin pedidos nuevos
    assert src.quote_calls == n_quotes

    extra = download_extra_days(src, path, offsets=(1, 2, 4), today=date(2026, 11, 30))  # ahora sí existe el día 4
    assert sorted(extra) == ["1", "2", "4"] and src.quote_calls > n_quotes

    loaded = load_extra_days(path)
    assert list(loaded) == [1, 2, 4]
    df, day, spot, r = loaded[1]
    assert day == "2026-11-02" and spot == 100.0 and r == 0.04 and len(df) == len(pre_before)
    pd.testing.assert_frame_equal(pd.read_parquet(path / "pre.parquet"), pre_before)  # lo previo no se tocó
    assert json.loads((path / "meta.json").read_text())["extra_days"]["4"]["day"] == "2026-11-05"
