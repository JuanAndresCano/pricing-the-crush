"""Cadenas de los días posteriores al anuncio, para ver si el vol crush es instantáneo o gradual (Pregunta 2).

Un evento descargado trae el cierre previo (`pre`) y el primero posterior (`post`, día 0). Aquí se agregan, para
los mismos contratos, los cierres `offsets` días hábiles después de `post`: por defecto 1, 2 y 4. Se guardan en la
carpeta del evento como `post_p<k>.parquet` y su información (día, precio de la acción, tasa) en `meta.json` bajo
`extra_days`. Es aditivo: no cambia `pre.parquet`, `post.parquet` ni la lectura normal del evento.

Es reanudable: lo que ya está en disco no se vuelve a pedir, y los días que todavía no existen (o que la fuente aún
no entrega) se omiten y se piden en la siguiente corrida.
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from pathlib import Path

import pandas as pd

from crush.data.calendar import next_trading_day
from crush.data.source import ChainSource, _chain_frame

DEFAULT_OFFSETS = (1, 2, 4)


def _selected_from_pre(pre: pd.DataFrame) -> pd.DataFrame:
    """Reconstruye la tabla de contratos elegidos a partir de la cadena previa ya guardada."""
    return pd.DataFrame({
        "ticker": pre["option_ticker"], "expiry": pre["expiration_date"], "strike_price": pre["strike"],
        "contract_type": pre["option_type"], "days_to_expiry": pre["days_to_expiry"],
    })


def extra_day(post_day, k: int) -> date:
    return next_trading_day(post_day, k)


def download_extra_days(
    src: ChainSource,
    event_path: Path,
    offsets=DEFAULT_OFFSETS,
    workers: int = 1,
    force: bool = False,
    today: date | None = None,
) -> dict:
    """Descarga los días extra de un evento ya guardado. Devuelve el diccionario `extra_days` resultante."""
    event_path = Path(event_path)
    meta = json.loads((event_path / "meta.json").read_text(encoding="utf-8"))
    pre = pd.read_parquet(event_path / "pre.parquet")
    selected = _selected_from_pre(pre)
    post_day = pd.Timestamp(meta["post_day"]).date()
    today = today or date.today()
    extra = meta.setdefault("extra_days", {})
    for k in offsets:
        f = event_path / f"post_p{k}.parquet"
        if f.exists() and not force and str(k) in extra:
            continue
        day = extra_day(post_day, k)
        if day >= today:  # el reporte de fin de día de ese día todavía no está disponible
            continue
        spot = src.spot(meta["ticker"], day)
        if spot is None:
            continue

        def fetch(t, d=day):
            return src.last_quote(t, d)

        with ThreadPoolExecutor(max_workers=workers) as pool:
            quotes = list(pool.map(fetch, selected["ticker"]))
        _chain_frame(selected, quotes, day, meta["ticker"]).to_parquet(f, index=False)
        info = {"day": str(day), "spot": spot}
        rate = getattr(src, "rate", None)
        r = rate(day) if rate else None
        if r is not None:
            info["r"] = r
        extra[str(k)] = info
        (event_path / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")  # se guarda tras cada día
    return extra


def load_extra_days(event_path: Path) -> dict[int, tuple[pd.DataFrame, str, float, float | None]]:
    """{k: (cadena, día, precio de la acción, tasa o None)} de los días extra ya descargados."""
    event_path = Path(event_path)
    meta = json.loads((event_path / "meta.json").read_text(encoding="utf-8"))
    out = {}
    for k, info in meta.get("extra_days", {}).items():
        f = event_path / f"post_p{k}.parquet"
        if f.exists():
            out[int(k)] = (pd.read_parquet(f), info["day"], float(info["spot"]), info.get("r"))
    return dict(sorted(out.items()))
