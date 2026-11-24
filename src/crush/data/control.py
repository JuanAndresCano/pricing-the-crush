"""Cadenas de un activo de control (por defecto SPY) para descontar el movimiento general del mercado.

La volatilidad de una acción cambia día a día por la volatilidad del mercado entero, no solo por su anuncio. Para
aislar el efecto del anuncio se guarda, para cada día que usa un evento, la parte cercana al dinero de la cadena
de SPY. Los días se comparten entre eventos: un día ya guardado no se vuelve a pedir.

    <raíz>/<fuente>/<símbolo>/<día>.parquet    cadena (mismo formato que pre.parquet)
    <raíz>/<fuente>/<símbolo>/<día>.json       precio del activo y tasa de ese día

Para el ATM interesan pocos strikes, así que la fuente de control se crea con `strike_range` (ThetaSource). Ojo:
ThetaData centra esa ventana en el precio del día anterior, no en el cierre del día pedido; si el activo se mueve más
que la ventana, el dinero queda afuera. Por eso se usa una ventana ancha y `ensure_control_day` rechaza (y no guarda)
cualquier cadena que no cubra el precio de cierre.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from crush.data.calendar import trading_days_between
from crush.data.source import ChainSource, _chain_frame


def control_dir(root: Path, source: str, symbol: str) -> Path:
    return Path(root) / source / symbol


def event_days(meta: dict) -> list[str]:
    """Días que usa un evento: cierre previo, cierre posterior y los días extra descargados."""
    days = [str(meta["pre_day"]), str(meta["post_day"])]
    days += [info["day"] for _, info in sorted(meta.get("extra_days", {}).items(), key=lambda kv: int(kv[0]))]
    return days


def covers_spot(df: pd.DataFrame, spot: float, tol: float = 0.015) -> bool:
    """True si la cadena tiene strikes a ambos lados del precio y uno a menos de `tol` (relativo) de él."""
    k = df["strike"].astype(float)
    return bool((k < spot).any() and (k > spot).any() and (k - spot).abs().min() <= tol * spot)


def ensure_control_day(src: ChainSource, symbol: str, day, root: Path, force: bool = False) -> Path | None:
    """Guarda la cadena de control de un día si no está. Devuelve la ruta del parquet, o None si no hay datos."""
    day = str(pd.Timestamp(day).date())
    d = control_dir(root, src.name, symbol)
    f = d / f"{day}.parquet"
    if f.exists() and not force:
        return f
    spot = src.spot(symbol, day)
    contracts = src.list_contracts(symbol, day)
    if spot is None or not contracts:
        return None
    sel = pd.DataFrame(contracts)
    sel = sel.assign(expiry=pd.to_datetime(sel["expiration_date"]).dt.date)
    sel["days_to_expiry"] = [trading_days_between(day, e) for e in sel["expiry"]]
    sel = sel[sel["days_to_expiry"] > 0].reset_index(drop=True)
    quotes = [src.last_quote(t, day) for t in sel["ticker"]]
    frame = _chain_frame(sel, quotes, day, symbol)
    if not covers_spot(frame, spot):
        return None  # la ventana de strikes no cubre el cierre: mejor no guardar nada que guardar una cadena sin ATM
    d.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(f, index=False)
    info = {"spot": spot}
    rate = getattr(src, "rate", None)
    r = rate(day) if rate else None
    if r is not None:
        info["r"] = r
    (d / f"{day}.json").write_text(json.dumps(info), encoding="utf-8")
    return f


def download_control(src: ChainSource, meta: dict, symbol: str, root: Path, force: bool = False) -> dict[str, bool]:
    """Guarda los días de control que necesita un evento. Devuelve {día: disponible}."""
    return {d: ensure_control_day(src, symbol, d, root, force) is not None for d in event_days(meta)}


def load_control(root: Path, source: str, symbol: str, days: list[str]) -> dict[str, tuple[pd.DataFrame, float, float | None]]:
    """{día: (cadena, precio, tasa o None)} de los días de control disponibles."""
    d = control_dir(root, source, symbol)
    out = {}
    for day in days:
        f = d / f"{day}.parquet"
        if f.exists():
            info = json.loads((d / f"{day}.json").read_text(encoding="utf-8"))
            out[day] = (pd.read_parquet(f), float(info["spot"]), info.get("r"))
    return out
