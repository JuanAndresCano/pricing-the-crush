import numpy as np
import pandas as pd
import pytest

from crush.analysis.same_recipe import RECIPES, model_smile
from crush.analysis.vrp import closes_from_eod, realized_to_expiry
from crush.models.event_models import MertonEvent


def test_realized_to_expiry_uses_all_days_and_rejects_gaps():
    days = ["2026-01-05", "2026-01-06", "2026-01-07", "2026-01-08"]
    closes = pd.Series([100.0, 110.0, 110.0, 99.0], index=days)
    out = realized_to_expiry(closes, days[0], days[-1])
    r = np.diff(np.log(closes.to_numpy()))
    assert out["n_days"] == 3 and out["rv"] == pytest.approx(252 / 3 * np.sum(r**2)) and not out["split_suspect"]
    assert out["event_share"] == pytest.approx(r[0] ** 2 / np.sum(r**2))
    assert realized_to_expiry(closes.drop(days[2]), days[0], days[-1]) is None  # falta un cierre
    assert realized_to_expiry(closes, days[0], "2026-01-09") is None  # el vencimiento aún no está


def test_closes_from_eod_indexes_by_day():
    frame = pd.DataFrame({"created": ["2026-01-02T17:15:57.229", "2026-01-05T17:15:06.709"], "close": [271.01, 267.26]})
    assert list(closes_from_eod(frame).index) == ["2026-01-02", "2026-01-05"]


def test_same_recipe_recovers_closed_form_with_dense_wide_strikes():
    model, theta = MertonEvent(), np.array([0.30, 1.0, -0.01, 0.05])
    F, T, r = 100.0, 8 / 252, 0.04
    K = np.linspace(70.0, 140.0, 141)
    iv = model_smile(model, theta, K, F, T, r)
    s = type("S", (), {"T": T, "r": r, "F": F})
    ok = np.isfinite(iv)
    k_closed = model.k_repl(theta, T, True)
    assert RECIPES["extrap"](K[ok], iv[ok], s) == pytest.approx(k_closed, rel=0.01)
