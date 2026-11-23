"""Razón de crush del modelo contra la observada, con una cadena de mercado generada por Merton-evento."""

import numpy as np
import pytest

from crush.analysis.crush import crush_table, evolve_state, model_crush, summarize_crush
from crush.market.chain import build_slices
from crush.models.event_models import BatesEvent, MertonEvent
from tests.test_market import EXPIRIES, POST, PRE, R, S0, THETA, event  # noqa: F401


def _row():
    row = {}
    for n, v in zip(MertonEvent().names, THETA):
        row[f"merton_{n}"] = v
    for n, v in zip(BatesEvent().names, [0.0625, 3.0, 0.0625, 0.5, -0.5, THETA[1], THETA[2], THETA[3]]):
        row[f"bates_{n}"] = v
    row.update(ticker="XYZ", announcement_date=PRE)
    return row


def test_evolve_state_only_moves_bates_variance():
    b = np.array([0.16, 5.0, 0.04, 0.5, -0.5, 1.0, -0.01, 0.05])
    assert evolve_state(BatesEvent(), b, 1 / 252)[0] < b[0]  # v0 > theta: baja hacia theta
    flat = b.copy(); flat[0] = flat[2]
    assert evolve_state(BatesEvent(), flat, 1 / 252)[0] == pytest.approx(flat[2])
    m = THETA.copy()
    assert np.array_equal(evolve_state(MertonEvent(), m, 1 / 252), m)


def test_merton_crush_matches_market_made_by_merton(event):  # noqa: F811
    pre = build_slices(event.pre, PRE, S0, R)[0]
    post = build_slices(event.post, POST, S0, R)[0]
    res = model_crush(MertonEvent(), THETA, pre, post)
    assert res["cr_revert"] == pytest.approx(1.0)  # Merton no tiene reversión
    assert res["cr_model"] == pytest.approx(post.atm_iv / pre.atm_iv, rel=5e-3)
    assert res["cr_model"] < 1.0


def test_bates_reversion_piece_follows_variance_gap(event):  # noqa: F811
    pre = build_slices(event.pre, PRE, S0, R)[0]
    post = build_slices(event.post, POST, S0, R)[0]
    base = np.array([0.0625, 3.0, 0.0625, 0.5, -0.5, 1.0, -0.01, 0.05])
    above, below = base.copy(), base.copy()
    above[0], below[0] = 0.16, 0.02
    r_up = model_crush(BatesEvent(), above, pre, post)
    r_dn = model_crush(BatesEvent(), below, pre, post)
    assert r_up["cr_revert"] < 1.0 < r_dn["cr_revert"]
    assert r_up["cr_model"] == pytest.approx(r_up["cr_event"] * r_up["cr_revert"])


def test_crush_table_and_summary(event):  # noqa: F811
    t = crush_table(event, _row(), R)
    assert set(t["model"]) == {"merton", "bates"} and len(t) == 2 * len(EXPIRIES)
    merton = t[t["model"] == "merton"]
    assert (merton["gap"].abs() < 0.01).all()  # el mercado lo generó Merton
    s = summarize_crush(t)
    assert set(s["model"]) == {"merton", "bates"} and (s["cr_obs"] < 1).all()
