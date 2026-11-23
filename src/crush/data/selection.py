"""Qué contratos descargar para un anuncio.

Reglas (documento de plan, decisión 10 y sección 5 del marco teórico):

  - Vencimientos que cruzan el anuncio: a partir del cierre posterior, y con al menos `min_days` días hábiles
    desde el cierre previo (el README usa vencimientos de 5, 7 o 10 días).
  - Los `n_near` más cercanos, más un vencimiento lejano de 21 a 60 días hábiles, que sirve para identificar
    kappa y theta en la calibración de Bates.
  - Strikes dentro de |ln(K/S)| <= max_abs_logm. Cerca del dinero (banda `atm_band`) se piden calls y puts,
    porque el forward sale de la paridad put-call. Lejos del dinero solo las opciones OTM.
  - Solo contratos estándar de 100 acciones.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from crush.data.calendar import trading_days_between


@dataclass(frozen=True)
class SelectionParams:
    n_near: int = 2
    min_days: int = 3
    far_min_days: int = 21
    far_max_days: int = 60
    max_abs_logm: float = 0.30
    atm_band: float = 0.05


def select_contracts(contracts, spot: float, pre_day, post_day, params: SelectionParams | None = None) -> pd.DataFrame:
    p = params or SelectionParams()
    df = pd.DataFrame(contracts)
    if df.empty:
        return df
    df = df[df["contract_type"].isin(["call", "put"])]
    if "shares_per_contract" in df:
        df = df[df["shares_per_contract"] == 100]
    df = df.assign(expiry=pd.to_datetime(df["expiration_date"]).dt.date)

    post = pd.Timestamp(post_day).date()
    dist = {e: trading_days_between(pre_day, e) for e in df["expiry"].unique()}
    eligible = sorted(e for e, d in dist.items() if e >= post and d >= p.min_days)
    near = eligible[: p.n_near]
    far = [e for e in eligible if e not in near and p.far_min_days <= dist[e] <= p.far_max_days][:1]
    chosen = near + far

    sel = df[df["expiry"].isin(chosen)].copy()
    if sel.empty:
        return sel
    lm = np.log(sel["strike_price"] / spot)
    otm = ((sel["contract_type"] == "put") & (lm < 0)) | ((sel["contract_type"] == "call") & (lm >= 0))
    keep = (lm.abs() <= p.max_abs_logm) & ((lm.abs() <= p.atm_band) | otm)
    sel = sel[keep].assign(days_to_expiry=lambda x: x["expiry"].map(dist), log_moneyness=lm[keep])
    return sel.sort_values(["expiry", "strike_price", "contract_type"]).reset_index(drop=True)
