"""Movimiento realizado contra implícito: fórmulas, fila por evento, historia y resumen."""

import numpy as np
import pandas as pd
import pytest

from crush.analysis.realized import add_history, event_swap_value, merge_models, realized_move, realized_row, summarize_realized
from crush.data.source import EventData
from crush.simulation.rv import swap_value
from tests.test_market import EXPIRIES, PRE, POST, S0, THETA, _chain


def _event(spot_post: float) -> EventData:
    meta = {"ticker": "XYZ", "announcement_date": PRE, "timing": "amc", "pre_day": PRE, "post_day": POST, "spot_pre": S0, "spot_post": spot_post}
    return EventData(_chain(0, True), _chain(1, False), meta)


def test_realized_move_is_log_return():
    assert realized_move({"spot_pre": 100.0, "spot_post": 105.0}) == pytest.approx(np.log(1.05))


def test_event_swap_value_matches_equation_35():
    """V exacta = ecuación (35) con RV_{0,t} = R^2/t, K_{t,T} = sigma_b^2 (sin evento) y K_var = sigma_b^2 + eps/T."""
    R, eps, T, sb, t = 0.06, 0.0025, 6 / 252, 0.25, 1 / 252
    k_var = sb**2 + eps / T
    expected = swap_value(R**2 / t, t, T, sb**2, k_var, notional=3.0)
    assert event_swap_value(R, eps, T, sb, t, notional=3.0) == pytest.approx(expected)


def test_swap_value_sign_depends_on_realized_vs_implied_not_on_crush():
    eps, T = 0.0025, 6 / 252
    assert event_swap_value(np.sqrt(eps) * 1.5, eps, T) > 0  # se movió más de lo cobrado: gana el comprador
    assert event_swap_value(np.sqrt(eps) * 0.5, eps, T) < 0  # se movió menos: pierde


def test_realized_row_fields_and_signs():
    big = realized_row(_event(S0 * np.exp(0.09)), 0.04)
    small = realized_row(_event(S0 * np.exp(0.005)), 0.04)
    assert big["R2"] == pytest.approx(0.09**2) and big["eps_atm"] > 0
    assert big["implied_move_atm"] == pytest.approx(np.sqrt(big["eps_atm"]))
    assert big["vrp_atm"] < 0 < small["vrp_atm"]  # el movimiento grande supera lo cobrado; el chico, no
    assert big["swap_value_atm"] > 0 > small["swap_value_atm"]
    assert big["z_atm"] > 1 > small["z_atm"]
    assert not big["split_suspect"]


def test_split_is_flagged():
    row = realized_row(_event(S0 * 0.5), 0.04)  # -69 % entre dos cierres seguidos: casi seguro un split
    assert row["split_suspect"]


def _table():
    rows = []
    for i, (tk, d, r2, eps) in enumerate([
        ("A", "2026-01-10", 0.0010, 0.0020), ("A", "2026-04-10", 0.0030, 0.0020), ("A", "2026-07-10", 0.0020, 0.0020), ("A", "2026-10-10", 0.0040, 0.0020),
        ("B", "2026-01-10", 0.0005, 0.0015), ("B", "2026-04-11", 0.0010, 0.0015),
    ]):
        rows.append({"ticker": tk, "announcement_date": d, "R2": r2, "abs_move": np.sqrt(r2), "split_suspect": False,
                     "eps_atm": eps, "eps_strike": eps, "implied_move_atm": np.sqrt(eps), "implied_move_strike": np.sqrt(eps),
                     "r2_over_eps_atm": r2 / eps, "r2_over_eps_strike": r2 / eps, "vrp_atm": eps - r2, "vrp_strike": eps - r2})
    return pd.DataFrame(rows)


def test_history_uses_only_prior_events_of_same_ticker():
    h = add_history(_table(), min_prior=2)
    a = h[h["ticker"] == "A"].sort_values("announcement_date")
    assert a["hist_mean_r2"].isna().tolist() == [True, True, False, False]  # hacen falta 2 previos
    assert a["hist_mean_r2"].iloc[2] == pytest.approx((0.0010 + 0.0030) / 2)  # no incluye el propio
    assert a["hist_mean_r2"].iloc[3] == pytest.approx((0.0010 + 0.0030 + 0.0020) / 3)
    assert h[h["ticker"] == "B"]["hist_mean_r2"].isna().all()


def test_summary_counts_and_clusters_by_date():
    s = summarize_realized(_table(), "atm")
    assert s["n_events"] == 6 and s["n_excluded"] == 0
    assert s["n_dates"] == 5  # 2026-01-10 se repite en dos empresas: un solo cluster
    assert s["share_implied_gt_realized"] == pytest.approx(3 / 6)
    assert np.isnan(s["pvalue_wilcoxon"])  # menos de 6 fechas


def test_summary_excludes_split_suspects():
    t = _table()
    t.loc[0, "split_suspect"] = True
    s = summarize_realized(t, "atm")
    assert s["n_events"] == 5 and s["n_excluded"] == 1


def test_merge_models_adds_ratio():
    cal = pd.DataFrame({"ticker": ["A"], "announcement_date": ["2026-01-10"], "eps_repl_merton": [0.002], "eps_qv_merton": [0.0021]})
    m = merge_models(_table(), cal)
    row = m[(m["ticker"] == "A") & (m["announcement_date"] == "2026-01-10")].iloc[0]
    assert row["r2_over_eps_repl_merton"] == pytest.approx(0.0010 / 0.002)
    assert m[m["ticker"] == "B"]["eps_repl_merton"].isna().all()
