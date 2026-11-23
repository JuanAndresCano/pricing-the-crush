"""Calendario de negociación de la NYSE y fechas de los anuncios de resultados.

El salto de earnings cae en el retorno que cruza el anuncio, así que importa si el anuncio fue antes de la
apertura (bmo) o después del cierre (amc):

  - bmo, fecha tau:  cierre previo = último día hábil antes de tau,  cierre posterior = tau
  - amc, fecha tau:  cierre previo = tau,  cierre posterior = siguiente día hábil

Las cadenas de opciones se descargan en esos dos cierres. El cierre real de cada día (incluidos los
medios días, como el viernes después de Acción de Gracias) viene del calendario de la NYSE.
"""

from __future__ import annotations

from datetime import date
from functools import lru_cache
from pathlib import Path

import pandas as pd
import pandas_market_calendars as mcal

TIMINGS = ("bmo", "amc")  # antes de la apertura, después del cierre
EVENT_COLUMNS = ["ticker", "announcement_date", "timing", "source_a", "source_b", "verified"]


@lru_cache(maxsize=1)
def _schedule() -> pd.DataFrame:
    return mcal.get_calendar("NYSE").schedule("2020-01-01", "2030-12-31")


def _ts(d) -> pd.Timestamp:
    return pd.Timestamp(d).normalize()


def is_trading_day(d) -> bool:
    return _ts(d) in _schedule().index


def trading_days(start, end) -> pd.DatetimeIndex:
    idx = _schedule().index
    return idx[(idx >= _ts(start)) & (idx <= _ts(end))]


def next_trading_day(d, n: int = 1) -> date:
    idx = _schedule().index
    pos = idx.searchsorted(_ts(d), side="right") - 1 + n if is_trading_day(d) else idx.searchsorted(_ts(d)) + n - 1
    return idx[pos].date()


def prev_trading_day(d, n: int = 1) -> date:
    idx = _schedule().index
    pos = idx.searchsorted(_ts(d), side="left") - n
    return idx[pos].date()


def trading_days_between(d0, d1) -> int:
    """Número de días hábiles en (d0, d1]. Negativo si d1 < d0."""
    if _ts(d1) < _ts(d0):
        return -trading_days_between(d1, d0)
    return int(len(trading_days(_ts(d0) + pd.Timedelta(days=1), d1)))


def market_open_ns(d) -> int:
    """Apertura de la sesión, en nanosegundos UTC desde el epoch."""
    return int(_schedule().loc[_ts(d), "market_open"].value)


def market_close_ns(d) -> int:
    """Cierre real de la sesión (13:00 ET en los medios días), en nanosegundos UTC."""
    return int(_schedule().loc[_ts(d), "market_close"].value)


def pre_post_days(announcement_date, timing: str) -> tuple[date, date]:
    """Cierre previo y cierre posterior al anuncio."""
    if timing not in TIMINGS:
        raise ValueError(f"timing debe ser uno de {TIMINGS}, no {timing!r}")
    if not is_trading_day(announcement_date):
        raise ValueError(f"{announcement_date} no es día hábil de la NYSE")
    ann = _ts(announcement_date).date()
    if timing == "bmo":
        return prev_trading_day(ann), ann
    return ann, next_trading_day(ann)


def load_events(path: str | Path) -> pd.DataFrame:
    """Lee y valida el CSV de anuncios. Lanza ValueError con la lista de problemas encontrados."""
    df = pd.read_csv(path, dtype=str).fillna("")
    missing = [c for c in EVENT_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas en {path}: {missing}")
    problems = []
    df["ticker"] = df["ticker"].str.strip().str.upper()
    df["timing"] = df["timing"].str.strip().str.lower()
    dates = pd.to_datetime(df["announcement_date"], errors="coerce")
    for i, row in df.iterrows():
        tag = f"fila {i + 2} ({row['ticker']})"
        if not row["ticker"]:
            problems.append(f"{tag}: ticker vacío")
        if pd.isna(dates[i]):
            problems.append(f"{tag}: fecha inválida {row['announcement_date']!r}")
        elif not is_trading_day(dates[i]):
            problems.append(f"{tag}: {row['announcement_date']} no es día hábil")
        if row["timing"] not in TIMINGS:
            problems.append(f"{tag}: timing {row['timing']!r} no es bmo ni amc")
    df["announcement_date"] = dates.dt.date
    dup = df.duplicated(["ticker", "announcement_date"])
    problems += [f"fila {i + 2}: evento repetido" for i in df.index[dup]]
    if problems:
        raise ValueError("Problemas en el calendario de anuncios:\n  " + "\n  ".join(problems))
    return df
