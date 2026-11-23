"""Descarga de la cadena de opciones de un evento, independiente de la fuente de datos.

Una fuente es cualquier objeto con tres métodos (Massive hoy; Databento, ThetaData u otra mañana):

    name: str
    spot(ticker, day) -> float | None                  cierre de la acción sin ajustar
    list_contracts(ticker, as_of) -> list[dict]        contratos vigentes ese día (ticker, contract_type,
                                                       expiration_date, strike_price, shares_per_contract)
    last_quote(option_ticker, day) -> dict | None      última cotización hasta el cierre (bid_price, ask_price,
                                                       bid_size, ask_size, sip_timestamp en ns)

`download_event` elige los contratos con la cadena del cierre previo y pide, para ese mismo conjunto, las
cotizaciones del cierre previo y del posterior. Así los dos días son comparables contrato a contrato.
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol

import pandas as pd

from crush.config import RAW_DIR
from crush.data.calendar import market_close_ns, pre_post_days
from crush.data.massive_client import MassiveClient
from crush.data.selection import SelectionParams, select_contracts


class ChainSource(Protocol):
    name: str

    def spot(self, ticker: str, day) -> float | None: ...
    def list_contracts(self, ticker: str, as_of) -> list[dict]: ...
    def last_quote(self, option_ticker: str, day) -> dict | None: ...


class MassiveSource:
    name = "massive"

    def __init__(self, client: MassiveClient | None = None):
        self.client = client or MassiveClient()

    def spot(self, ticker, day):
        return self.client.stock_close(ticker, day)

    def list_contracts(self, ticker, as_of):
        return self.client.option_contracts_incl_expired(ticker, str(as_of))

    def last_quote(self, option_ticker, day):
        return self.client.option_last_quote(option_ticker, day)


class DataError(RuntimeError):
    """Falta algo imprescindible del evento (por ejemplo, el precio de la acción)."""


@dataclass
class EventData:
    pre: pd.DataFrame
    post: pd.DataFrame
    meta: dict


CHAIN_COLUMNS = [
    "option_ticker", "underlying", "expiration_date", "strike", "option_type", "days_to_expiry",
    "bid", "ask", "bid_size", "ask_size", "quote_ts", "quote_age_s", "has_quote",
]


def _chain_frame(selected: pd.DataFrame, quotes: list[dict | None], day, ticker: str) -> pd.DataFrame:
    close = market_close_ns(day)
    rows = []
    for (_, c), q in zip(selected.iterrows(), quotes):
        q = q or {}
        ts = q.get("sip_timestamp")
        rows.append({
            "option_ticker": c["ticker"],
            "underlying": ticker,
            "expiration_date": c["expiry"],
            "strike": float(c["strike_price"]),
            "option_type": c["contract_type"],
            "days_to_expiry": int(c["days_to_expiry"]),
            "bid": q.get("bid_price"),
            "ask": q.get("ask_price"),
            "bid_size": q.get("bid_size"),
            "ask_size": q.get("ask_size"),
            "quote_ts": ts,
            "quote_age_s": (close - ts) / 1e9 if ts else None,
            "has_quote": bool(q),
        })
    return pd.DataFrame(rows, columns=CHAIN_COLUMNS)


def download_event(
    src: ChainSource,
    ticker: str,
    announcement_date,
    timing: str,
    params: SelectionParams | None = None,
    workers: int = 8,
) -> EventData:
    pre, post = pre_post_days(announcement_date, timing)
    spot_pre, spot_post = src.spot(ticker, pre), src.spot(ticker, post)
    if spot_pre is None or spot_post is None:
        raise DataError(f"{ticker}: sin precio de cierre de la acción el {pre if spot_pre is None else post}")
    rate = getattr(src, "rate", None)  # opcional: no todas las fuentes entregan tasas
    r_pre, r_post = (rate(pre), rate(post)) if rate else (None, None)
    selected = select_contracts(src.list_contracts(ticker, pre), spot_pre, pre, post, params)
    if selected.empty:
        raise DataError(f"{ticker}: ningún contrato cumple las reglas de selección para el anuncio del {announcement_date}")

    def fetch(day):
        with ThreadPoolExecutor(max_workers=workers) as pool:
            return list(pool.map(lambda t: src.last_quote(t, day), selected["ticker"]))

    pre_df = _chain_frame(selected, fetch(pre), pre, ticker)
    post_df = _chain_frame(selected, fetch(post), post, ticker)
    meta = {
        "source": src.name,
        "ticker": ticker,
        "announcement_date": str(announcement_date),
        "timing": timing,
        "pre_day": str(pre),
        "post_day": str(post),
        "spot_pre": spot_pre,
        "spot_post": spot_post,
        **({"r_pre": r_pre} if r_pre is not None else {}),
        **({"r_post": r_post} if r_post is not None else {}),
        "expirations": sorted(str(e) for e in selected["expiry"].unique()),
        "n_contracts": int(len(selected)),
        "n_quotes_pre": int(pre_df["has_quote"].sum()),
        "n_quotes_post": int(post_df["has_quote"].sum()),
        "downloaded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    return EventData(pre_df, post_df, meta)


# --- caché en disco ---------------------------------------------------------------------------------


def event_dir(root: Path, source: str, ticker: str, announcement_date) -> Path:
    return Path(root) / source / f"{ticker}_{announcement_date}"


def save_event(path: Path, data: EventData) -> None:
    path.mkdir(parents=True, exist_ok=True)
    data.pre.to_parquet(path / "pre.parquet", index=False)
    data.post.to_parquet(path / "post.parquet", index=False)
    (path / "meta.json").write_text(json.dumps(data.meta, indent=2), encoding="utf-8")


def load_event(path: Path) -> EventData:
    return EventData(
        pd.read_parquet(path / "pre.parquet"),
        pd.read_parquet(path / "post.parquet"),
        json.loads((path / "meta.json").read_text(encoding="utf-8")),
    )


def download_and_cache(
    src: ChainSource,
    ticker: str,
    announcement_date,
    timing: str,
    root: Path = RAW_DIR,
    force: bool = False,
    **kw,
) -> tuple[Path, str]:
    """Descarga un evento y lo guarda. Si ya está en disco no vuelve a llamar a la API, salvo con `force`."""
    path = event_dir(root, src.name, ticker, announcement_date)
    if not force and (path / "meta.json").exists():
        return path, "cached"
    save_event(path, download_event(src, ticker, announcement_date, timing, **kw))
    return path, "downloaded"
