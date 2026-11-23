"""De cotizaciones bid/ask de un día a una curva de volatilidad implícita por vencimiento.

Pasos (sección 5 del marco teórico):

  1. Limpieza: se descartan cotizaciones sin dato, con bid cero, cruzadas, con spread relativo muy ancho o
     demasiado viejas respecto al cierre. El precio de cada opción es el punto medio.
  2. Forward por paridad put-call, ecuación (39): F = K + e^{rT}(C - P), con los strikes más cercanos a la
     acción y la mediana de los resultados. Incorpora dividendos y costo de préstamo.
  3. Opciones OTM (put bajo el forward, call sobre él) y volatilidad implícita de Black-Scholes sobre el forward.

El plazo T se mide en días de negociación / 252, igual que en la calibración. La tasa `r` se pasa como constante:
afecta poco a vencimientos de semanas, y el forward ya no depende de dividendos.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from crush.config import TRADING_DAYS_PER_YEAR
from crush.data.calendar import trading_days_between
from crush.models.black_scholes import implied_vol_array


@dataclass(frozen=True)
class CleanParams:
    max_rel_spread: float = 0.5  # (ask - bid) / mid
    max_age_s: float = 3600.0  # antigüedad máxima de la cotización respecto al cierre
    n_parity: int = 5  # strikes cercanos a la acción para el forward


def clean_quotes(df: pd.DataFrame, params: CleanParams | None = None) -> pd.DataFrame:
    """Agrega `mid`, `rel_spread` y `valid`. No elimina filas, para poder contar cuántas se perdieron."""
    p = params or CleanParams()
    out = df.copy()
    bid = pd.to_numeric(out["bid"], errors="coerce")
    ask = pd.to_numeric(out["ask"], errors="coerce")
    out["mid"] = (bid + ask) / 2.0
    out["rel_spread"] = (ask - bid) / out["mid"]
    age = pd.to_numeric(out["quote_age_s"], errors="coerce")
    out["valid"] = (
        out["has_quote"].astype(bool)
        & (bid > 0)
        & (ask >= bid)
        & (out["rel_spread"] <= p.max_rel_spread)
        & (age.isna() | (age <= p.max_age_s))  # sin hora (reporte de fin de día) no se puede juzgar la antigüedad
    )
    return out


def years_to_expiry(day, expiry) -> float:
    return trading_days_between(day, expiry) / TRADING_DAYS_PER_YEAR


def parity_forward(calls: pd.DataFrame, puts: pd.DataFrame, spot: float, T: float, r: float, n: int = 5) -> tuple[float, int]:
    """Forward por (39) con los `n` strikes más cercanos a `spot` que tienen call y put válidos. Devuelve (F, pares usados)."""
    m = calls.merge(puts, on="strike", suffixes=("_c", "_p"))
    if m.empty:
        return float("nan"), 0
    m = m.assign(dist=(m["strike"] - spot).abs()).nsmallest(n, "dist")
    f = m["strike"] + np.exp(r * T) * (m["mid_c"] - m["mid_p"])
    return float(np.median(f)), int(len(m))


@dataclass
class Slice:
    """Un vencimiento en un día: forward, tasa, plazo y la curva de opciones OTM con su volatilidad implícita."""

    expiry: object
    T: float
    r: float
    F: float
    K: np.ndarray
    price: np.ndarray
    is_call: np.ndarray
    iv: np.ndarray
    n_pairs: int

    @property
    def logm(self) -> np.ndarray:
        return np.log(self.K / self.F)

    @property
    def atm_iv(self) -> float:
        """Volatilidad implícita en K = F, interpolada linealmente en la moneyness logarítmica."""
        x = self.logm
        if x.size < 2 or x.min() > 0 or x.max() < 0:
            return float("nan")
        return float(np.interp(0.0, x, self.iv))


def build_slices(df: pd.DataFrame, day, spot: float, r: float, params: CleanParams | None = None) -> list[Slice]:
    """Un `Slice` por vencimiento de la cadena `df` (formato de `source.CHAIN_COLUMNS`). Ordenados por plazo.

    Se omiten los vencimientos sin pares call/put válidos (no hay forward confiable) o con menos de 5 puntos de smile.
    """
    p = params or CleanParams()
    d = clean_quotes(df, p)
    d = d[d["valid"]]
    slices: list[Slice] = []
    for expiry, g in d.groupby("expiration_date"):
        T = years_to_expiry(day, expiry)
        if T <= 0:
            continue
        calls = g[g["option_type"] == "call"][["strike", "mid"]]
        puts = g[g["option_type"] == "put"][["strike", "mid"]]
        F, n_pairs = parity_forward(calls, puts, spot, T, r, p.n_parity)
        if not np.isfinite(F):
            continue
        is_call = np.where(g["strike"].to_numpy() >= F, True, False)
        want = np.where(is_call, g["option_type"].to_numpy() == "call", g["option_type"].to_numpy() == "put")
        k = g["strike"].to_numpy(float)[want]
        price = g["mid"].to_numpy(float)[want]
        ic = is_call[want]
        iv = implied_vol_array(price, F, k, T, r, ic)
        ok = np.isfinite(iv)
        if ok.sum() < 5:
            continue
        order = np.argsort(k[ok])
        slices.append(Slice(expiry, T, r, F, k[ok][order], price[ok][order], ic[ok][order], iv[ok][order], n_pairs))
    return sorted(slices, key=lambda s: s.T)
