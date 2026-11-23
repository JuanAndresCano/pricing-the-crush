"""Cliente de ThetaData v3 (plan gratis: reporte de fin de día) y su fuente para `download_event`.

Requiere el Theta Terminal corriendo en el computador (por defecto http://127.0.0.1:25503/v3).

  - Un pedido por día y empresa trae toda la cadena (`expiration=*`) con bid y ask de cierre. El reporte de fin de día
    lo genera ThetaData a las 17:15 ET con el último NBBO de OPRA. No trae hora de cada cotización.
  - Ritmo controlado (plan gratis: 20 pedidos por minuto) y respuestas crudas guardadas en `data/raw_api/theta/`.
  - Errores definitivos (4xx salvo 408 y 429) se guardan; los transitorios se reintentan con espera creciente.

Detalles del formato tomados de la documentación oficial; se confirman con la primera llamada real.
"""

from __future__ import annotations

import io
import os
import threading
import time
from datetime import date

import pandas as pd
import requests

from crush.config import RAW_API_DIR
from crush.data.rawcache import DefinitiveError, RawCache
from crush.data.throttle import RateLimiter

DEFAULT_BASE_URL = "http://127.0.0.1:25503/v3"
TRANSIENT = {408, 429}


class ThetaError(RuntimeError):
    pass


def _ymd(day) -> str:
    return pd.Timestamp(day).strftime("%Y%m%d")


class ThetaClient:
    def __init__(
        self,
        base_url: str | None = None,
        cache: RawCache | None = None,
        limiter: RateLimiter | None = None,
        session=None,
        retries: int = 5,
        backoff: float = 5.0,
        sleep=time.sleep,
    ):
        self.base_url = (base_url or os.environ.get("THETA_BASE_URL") or DEFAULT_BASE_URL).rstrip("/")
        self.cache = cache or RawCache(RAW_API_DIR, "theta")
        self.limiter = limiter or RateLimiter(float(os.environ.get("THETA_MIN_INTERVAL", "3.1")))
        self.session = session or requests.Session()
        self.retries, self.backoff, self._sleep = retries, backoff, sleep

    def _request(self, endpoint: str, params: dict) -> str:
        url = f"{self.base_url}/{endpoint.strip('/')}"
        for attempt in range(self.retries + 1):
            self.limiter.wait()
            try:
                resp = self.session.get(url, params=params, timeout=120)
            except requests.ConnectionError as exc:
                raise ThetaError("No hay conexión con el Theta Terminal. ¿Está abierto y con la sesión iniciada?") from exc
            code = resp.status_code
            if code == 200:
                return resp.text
            if code in TRANSIENT or code >= 500:
                if attempt < self.retries:
                    self._sleep(self.backoff * 2**attempt)
                    continue
                raise ThetaError(f"{endpoint}: {code} tras {self.retries + 1} intentos: {resp.text[:200]}")
            raise DefinitiveError(code, resp.text[:300])
        raise ThetaError(f"{endpoint}: sin respuesta")  # pragma: no cover

    def get_csv(self, endpoint: str, params: dict) -> pd.DataFrame:
        params = {**params, "format": "csv"}
        text = self.cache.fetch(endpoint, params, lambda: self._request(endpoint, params))
        return pd.read_csv(io.StringIO(text)) if text.strip() else pd.DataFrame()

    def stock_eod(self, symbol: str, day) -> pd.DataFrame:
        d = _ymd(day)
        return self.get_csv("stock/history/eod", {"symbol": symbol, "start_date": d, "end_date": d})

    def interest_rate(self, symbol: str, start_day, end_day) -> pd.DataFrame:
        """Tasa diaria en porcentaje (columnas created, rate). Gratis desde 2024-01-01."""
        return self.get_csv("interest_rate/history/eod", {"symbol": symbol, "start_date": _ymd(start_day), "end_date": _ymd(end_day)})

    def option_eod(self, symbol: str, day, max_dte: int | None = 100, strike_range: int | None = None) -> pd.DataFrame:
        d = _ymd(day)
        params = {"symbol": symbol, "expiration": "*", "start_date": d, "end_date": d}
        if max_dte is not None:
            params["max_dte"] = max_dte
        if strike_range is not None:
            params["strike_range"] = strike_range
        return self.get_csv("option/history/eod", params)


# --- fuente para download_event ---------------------------------------------------------------------------


def occ_ticker(symbol: str, expiration, right: str, strike: float) -> str:
    """Identificador estilo OCC: AAPL261106C00270000."""
    exp = pd.Timestamp(expiration).strftime("%y%m%d")
    return f"{symbol}{exp}{right.upper()}{int(round(float(strike) * 1000)):08d}"


class ThetaSource:
    """`ChainSource` sobre ThetaData. La cadena de cada (empresa, día) se pide una vez y se atiende desde memoria."""

    name = "theta"

    def __init__(self, client: ThetaClient | None = None, max_dte: int | None = 100, strike_range: int | None = None):
        self.client = client or ThetaClient()
        self.max_dte, self.strike_range = max_dte, strike_range
        self._chains: dict[tuple[str, str], dict[str, dict]] = {}
        self._frames: dict[tuple[str, str], pd.DataFrame] = {}
        self._lock = threading.Lock()

    def spot(self, ticker, day):
        try:
            df = self.client.stock_eod(ticker, day)
        except DefinitiveError:
            return None
        if df.empty or "close" not in df:
            return None
        close = df["close"].iloc[-1]
        return float(close) if pd.notna(close) and close > 0 else None

    def rate(self, day, symbol: str = "TREASURY_M1") -> float | None:
        """Tasa libre de riesgo anual (decimal) del último día con dato hasta `day`; None si no hay acceso."""
        end = pd.Timestamp(day)
        try:
            df = self.client.interest_rate(symbol, end - pd.Timedelta(days=7), end)
        except DefinitiveError:
            return None
        if df.empty or "rate" not in df:
            return None
        df = df[pd.to_datetime(df["created"]) <= end]
        return float(df["rate"].iloc[-1]) / 100.0 if len(df) else None

    def _load(self, ticker: str, day) -> pd.DataFrame:
        key = (ticker, str(pd.Timestamp(day).date()))
        with self._lock:
            if key not in self._frames:
                try:
                    df = self.client.option_eod(ticker, day, self.max_dte, self.strike_range)
                except DefinitiveError:
                    df = pd.DataFrame()
                if not df.empty:
                    df = df.assign(right=df["right"].astype(str).str.strip().str[0].str.upper())
                    df["option_ticker"] = [occ_ticker(ticker, e, r, k) for e, r, k in zip(df["expiration"], df["right"], df["strike"])]
                self._frames[key] = df
                self._chains[key] = {
                    row.option_ticker: {
                        "bid_price": row.bid, "ask_price": row.ask,
                        "bid_size": getattr(row, "bid_size", None), "ask_size": getattr(row, "ask_size", None),
                        "sip_timestamp": None,  # el reporte de fin de día no trae la hora de la cotización
                    }
                    for row in df.itertuples()
                    if pd.notna(row.bid) and pd.notna(row.ask)
                } if not df.empty else {}
            return self._frames[key]

    def list_contracts(self, ticker, as_of) -> list[dict]:
        df = self._load(ticker, as_of)
        if df.empty:
            return []
        return [
            {
                "ticker": r.option_ticker, "contract_type": "call" if r.right == "C" else "put",
                "expiration_date": str(r.expiration), "strike_price": float(r.strike), "shares_per_contract": 100,
            }
            for r in df.itertuples()
        ]

    def last_quote(self, option_ticker, day) -> dict | None:
        underlying = option_ticker[:-15]  # AAMMDD (6) + C/P (1) + strike (8)
        self._load(underlying, day)
        return self._chains[(underlying, str(pd.Timestamp(day).date()))].get(option_ticker)
