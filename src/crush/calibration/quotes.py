"""Cadena de opciones para calibrar: vencimientos, strikes, precios y pesos.

Una cadena agrupa opciones OTM de varios vencimientos. Cada vencimiento tiene su forward F, su tasa r y una
marca `crosses` que dice si el vencimiento cruza el anuncio (T >= tau). Los precios son puntos medios.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from crush.models.black_scholes import bs_vega, implied_vol_array
from crush.models.event_models import EventModel
from crush.models.fourier import CallPricer


@dataclass
class OptionChain:
    T: np.ndarray  # plazo en años de negociación, por opción
    F: np.ndarray  # forward del vencimiento, por opción
    r: np.ndarray  # tasa libre de riesgo, por opción
    K: np.ndarray  # strike
    is_call: np.ndarray  # True si la opción cotizada es un call, False si es un put
    price: np.ndarray  # precio medio de mercado
    crosses: np.ndarray  # True si el vencimiento cruza el anuncio
    _groups: list = field(default_factory=list, repr=False)
    _pricers: dict = field(default_factory=dict, repr=False)

    def __post_init__(self):
        for name in ("T", "F", "r", "K", "price"):
            setattr(self, name, np.asarray(getattr(self, name), dtype=float))
        self.is_call = np.asarray(self.is_call, dtype=bool)
        self.crosses = np.asarray(self.crosses, dtype=bool)
        self._groups = []
        for t in np.unique(self.T):
            idx = np.flatnonzero(self.T == t)
            self._groups.append((float(t), float(self.F[idx[0]]), float(self.r[idx[0]]), bool(self.crosses[idx[0]]), idx))

    def __len__(self) -> int:
        return int(self.K.size)

    @property
    def groups(self):
        return self._groups

    def pricer(self, idx: np.ndarray, **kw) -> CallPricer:
        key = (idx.tobytes(), tuple(sorted(kw.items())))
        if key not in self._pricers:
            self._pricers[key] = CallPricer(self.K[idx], **kw)
        return self._pricers[key]

    def market_iv(self) -> np.ndarray:
        return implied_vol_array(self.price, self.F, self.K, self.T, self.r, self.is_call)

    def vega(self) -> np.ndarray:
        """Vega de Black-Scholes a la volatilidad implícita de mercado. nan donde no hay volatilidad implícita."""
        iv = self.market_iv()
        return bs_vega(self.F, self.K, self.T, iv, self.r)

    def subset(self, mask) -> "OptionChain":
        m = np.asarray(mask)
        return OptionChain(self.T[m], self.F[m], self.r[m], self.K[m], self.is_call[m], self.price[m], self.crosses[m])


def model_prices(model: EventModel, theta, chain: OptionChain, **pricer_kw) -> np.ndarray:
    """Precio del modelo para cada opción de la cadena. Los puts se obtienen por paridad put-call."""
    out = np.empty(len(chain))
    for T, F, r, crosses, idx in chain.groups:
        calls = model.call_prices(theta, F, T, r, crosses, chain.pricer(idx, **pricer_kw))
        puts = calls - np.exp(-r * T) * (F - chain.K[idx])
        out[idx] = np.where(chain.is_call[idx], calls, puts)
    return out


def synthetic_chain(
    model: EventModel,
    theta,
    days=(5, 10, 21),
    event_day: int = 2,
    S0: float = 100.0,
    r: float = 0.03,
    n_strikes: int = 25,
    half_width: float = 0.20,
) -> OptionChain:
    """Cadena sin ruido generada con el propio modelo: puts OTM bajo el forward y calls OTM sobre él."""
    T, F, rr, K, is_call, crosses = [], [], [], [], [], []
    for d in days:
        t = d / 252.0
        fwd = S0 * np.exp(r * t)
        strikes = fwd * np.exp(np.linspace(-half_width, half_width, n_strikes))
        T += [t] * n_strikes
        F += [fwd] * n_strikes
        rr += [r] * n_strikes
        K += list(strikes)
        is_call += list(strikes >= fwd)
        crosses += [d >= event_day] * n_strikes
    chain = OptionChain(np.array(T), np.array(F), np.array(rr), np.array(K), np.array(is_call), np.zeros(len(T)), np.array(crosses))
    chain.price = model_prices(model, theta, chain)
    return chain


def perturb_iv(chain: OptionChain, sd: float, rng: np.random.Generator) -> OptionChain:
    """Copia de la cadena con ruido gaussiano de desviación `sd` (en volatilidad) sobre la volatilidad implícita."""
    from crush.models.black_scholes import bs_price

    iv = chain.market_iv()
    ok = np.isfinite(iv)
    noisy = np.where(ok, iv + sd * rng.standard_normal(len(chain)), np.nan)
    new_price = np.where(
        ok,
        [float(bs_price(f, k, t, s, r, c)) if np.isfinite(s) else np.nan for f, k, t, s, r, c in zip(chain.F, chain.K, chain.T, noisy, chain.r, chain.is_call)],
        np.nan,
    )
    out = OptionChain(chain.T, chain.F, chain.r, chain.K, chain.is_call, new_price, chain.crosses)
    return out.subset(ok)
