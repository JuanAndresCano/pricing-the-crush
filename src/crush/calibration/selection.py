"""Medidas de ajuste y comparación de modelos.

  - RMSE en volatilidad implícita, ecuación (53)
  - BIC, ecuación (42):  BIC = M ln(RSS_IV / M) + p ln M
"""

from __future__ import annotations

import numpy as np

from crush.calibration.quotes import OptionChain, model_prices
from crush.models.black_scholes import implied_vol_array
from crush.models.event_models import EventModel


def iv_errors(model: EventModel, theta, chain: OptionChain) -> np.ndarray:
    """Error en volatilidad implícita (modelo - mercado) para las opciones donde ambas existen."""
    mod_iv = implied_vol_array(model_prices(model, theta, chain), chain.F, chain.K, chain.T, chain.r, chain.is_call)
    err = mod_iv - chain.market_iv()
    return err[np.isfinite(err)]


def rmse_iv(model: EventModel, theta, chain: OptionChain) -> float:
    e = iv_errors(model, theta, chain)
    return float(np.sqrt(np.mean(e**2)))


def bic(model: EventModel, theta, chain: OptionChain) -> float:
    e = iv_errors(model, theta, chain)
    m = e.size
    rss = max(float(np.sum(e**2)), 1e-300)
    return float(m * np.log(rss / m) + model.n_params * np.log(m))
