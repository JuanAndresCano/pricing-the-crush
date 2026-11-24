"""Descarga de la ventana placebo de un evento (ver analysis/placebo.py).

Para cada evento ya descargado se toma un día 0 de placebo, `offset` días hábiles después del día 0 real, y se
descargan las cadenas de ese día y de los días +1, +2 y +4 con la misma regla de selección de contratos del evento
(`data.selection.select_contracts`: dos vencimientos cercanos y uno lejano). Se guarda aparte del evento, en

    <raíz>/<TICKER>_<fecha del anuncio>/meta.json, p0.parquet, p1.parquet, p2.parquet, p4.parquet

para no tocar nada del pipeline principal. Es reanudable: lo que ya está en disco no se vuelve a pedir, y los días que
la fuente todavía no entrega se omiten. Cuesta unos 3 pedidos por día (precio, tasa y cadena completa).
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pandas as pd

from crush.data.calendar import next_trading_day, prev_trading_day
from crush.data.selection import SelectionParams, select_contracts
from crush.data.source import ChainSource, _chain_frame

OFFSETS = (1, 2, 4)


def placebo_dir(root: Path, ticker: str, announcement_date) -> Path:
    return Path(root) / f"{ticker}_{announcement_date}"


def placebo_day0(post_day, offset: int) -> date:
    return next_trading_day(pd.Timestamp(post_day).date(), offset)


def download_placebo(src: ChainSource, event_meta: dict, root: Path, offset: int, offsets=OFFSETS, today: date | None = None,
                     params: SelectionParams | None = None) -> dict:
    """Descarga el placebo de un evento. Devuelve el `meta` resultante (con los días disponibles en `days`)."""
    today = today or date.today()
    ticker, ann = event_meta["ticker"], event_meta["announcement_date"]
    d0 = placebo_day0(event_meta["post_day"], offset)
    out_dir = placebo_dir(root, ticker, ann)
    meta_path = out_dir / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {
        "ticker": ticker, "announcement_date": ann, "offset": offset, "day0": str(d0), "days": {}}
    if d0 >= today:
        return meta  # el día 0 del placebo todavía no existe
    out_dir.mkdir(parents=True, exist_ok=True)
    rate = getattr(src, "rate", None)

    def save():
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    if "contracts" not in meta:
        spot0 = src.spot(ticker, d0)
        if spot0 is None:
            return meta
        pre_p = prev_trading_day(d0)
        selected = select_contracts(src.list_contracts(ticker, d0), spot0, pre_p, d0, params)
        if selected.empty:
            return meta
        # días hasta el vencimiento se miden desde el día previo al día 0 del placebo, como en el evento real
        selected.to_parquet(out_dir / "contracts.parquet", index=False)
        meta["contracts"] = int(len(selected))
    selected = pd.read_parquet(out_dir / "contracts.parquet")

    for k in (0, *offsets):
        f = out_dir / f"p{k}.parquet"
        if f.exists() and str(k) in meta["days"]:
            continue
        day = d0 if k == 0 else next_trading_day(d0, k)
        if day >= today:
            continue
        spot = src.spot(ticker, day)
        if spot is None:
            continue
        quotes = [src.last_quote(t, day) for t in selected["ticker"]]
        _chain_frame(selected, quotes, day, ticker).to_parquet(f, index=False)
        info = {"day": str(day), "spot": spot}
        r = rate(day) if rate else None
        if r is not None:
            info["r"] = r
        meta["days"][str(k)] = info
        save()
    save()
    return meta


def placebo_days(meta: dict) -> list[str]:
    return [info["day"] for _, info in sorted(meta.get("days", {}).items(), key=lambda kv: int(kv[0]))]


def load_placebo(folder: Path) -> tuple[dict, dict]:
    """(meta, {k: (cadena, día, precio, tasa o None)}) de lo ya descargado."""
    folder = Path(folder)
    meta = json.loads((folder / "meta.json").read_text(encoding="utf-8"))
    days = {}
    for k, info in meta.get("days", {}).items():
        f = folder / f"p{k}.parquet"
        if f.exists():
            days[int(k)] = (pd.read_parquet(f), info["day"], float(info["spot"]), info.get("r"))
    return meta, dict(sorted(days.items()))
