"""Cliente mínimo de la API REST de Massive (antes Polygon.io).

Endpoints usados (según la documentación de Massive):
  - GET /v3/reference/options/contracts                               contratos de opciones de un subyacente
  - GET /v3/quotes/{optionsTicker}                                    cotizaciones históricas (bid y ask)
  - GET /v2/aggs/ticker/{ticker}/range/1/day/{from}/{to}              barras diarias de la acción

La clave se lee de la variable de entorno MASSIVE_API_KEY (archivo .env, nunca en git).
El método de autenticación y la URL base no figuran en las páginas de endpoints, así que
se pueden cambiar con MASSIVE_BASE_URL y MASSIVE_AUTH ("header" o "query").
"""

from __future__ import annotations

import os
import time
from typing import Any, Iterator

import requests
from dotenv import load_dotenv

from crush.data.calendar import market_close_ns, market_open_ns

DEFAULT_BASE_URL = "https://api.massive.com"
RETRY_STATUS = {429, 500, 502, 503, 504}


class MassiveError(RuntimeError):
    """Error devuelto por la API, con el código HTTP y el mensaje del servidor."""


class MassiveClient:
    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        auth: str | None = None,
        session: requests.Session | None = None,
        max_retries: int = 5,
        backoff: float = 1.0,
    ) -> None:
        load_dotenv()
        self.api_key = api_key or os.getenv("MASSIVE_API_KEY", "")
        if not self.api_key:
            raise MassiveError("Falta MASSIVE_API_KEY. Crea el archivo .env a partir de .env.example.")
        self.base_url = (base_url or os.getenv("MASSIVE_BASE_URL") or DEFAULT_BASE_URL).rstrip("/")
        self.auth = (auth or os.getenv("MASSIVE_AUTH") or "header").lower()
        self.session = session or requests.Session()
        self.max_retries = max_retries
        self.backoff = backoff

    def _auth_kwargs(self) -> dict[str, Any]:
        if self.auth == "query":
            return {"params": {"apiKey": self.api_key}}
        return {"headers": {"Authorization": f"Bearer {self.api_key}"}}

    def _get(self, url: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        auth = self._auth_kwargs()
        merged = {**(params or {}), **auth.get("params", {})}
        for attempt in range(self.max_retries + 1):
            resp = self.session.get(url, params=merged, headers=auth.get("headers"), timeout=30)
            if resp.status_code == 200:
                return resp.json()
            if resp.status_code in RETRY_STATUS and attempt < self.max_retries:
                time.sleep(self.backoff * 2**attempt)  # límite de llamadas o falla del servidor: reintentar
                continue
            # El cuerpo suele explicar si el plan no incluye el endpoint. Nunca incluye la clave.
            raise MassiveError(f"HTTP {resp.status_code} en {url.split('?')[0]}: {resp.text[:300]}")
        raise MassiveError("Reintentos agotados")  # pragma: no cover

    def paginate(self, path: str, params: dict[str, Any] | None = None, max_pages: int = 50) -> Iterator[dict[str, Any]]:
        """Recorre `next_url` y entrega cada elemento de `results`."""
        url = f"{self.base_url}{path}"
        page_params = params
        for _ in range(max_pages):
            data = self._get(url, page_params)
            yield from data.get("results", [])
            url = data.get("next_url")
            page_params = None
            if not url:
                return

    # --- contratos -------------------------------------------------------------------------------
    def option_contracts(self, underlying: str, as_of: str, **filters: Any) -> list[dict[str, Any]]:
        """Contratos de opciones de `underlying` vigentes en la fecha `as_of` (YYYY-MM-DD)."""
        params = {"underlying_ticker": underlying, "as_of": as_of, "limit": 1000, **filters}
        return list(self.paginate("/v3/reference/options/contracts", params))

    def option_contracts_incl_expired(self, underlying: str, as_of: str, **filters: Any) -> list[dict[str, Any]]:
        """Contratos vigentes en `as_of`, incluidos los que ya vencieron hoy (`expired` false y true, sin repetir)."""
        seen: dict[str, dict[str, Any]] = {}
        for flag in ("false", "true"):
            for c in self.option_contracts(underlying, as_of, expired=flag, **filters):
                seen.setdefault(c["ticker"], c)
        return list(seen.values())

    # --- cotizaciones ----------------------------------------------------------------------------
    def option_quotes(self, option_ticker: str, date: str, limit: int = 50000) -> list[dict[str, Any]]:
        """Todas las cotizaciones de un contrato en un día (YYYY-MM-DD). Puede ser mucho volumen."""
        params = {"timestamp": date, "limit": limit, "order": "asc", "sort": "timestamp"}
        return list(self.paginate(f"/v3/quotes/{option_ticker}", params))

    def option_last_quote(self, option_ticker: str, date) -> dict[str, Any] | None:
        """Última cotización del contrato en la sesión de `date`, hasta el cierre real. Una llamada, un resultado."""
        params = {
            "timestamp.gte": market_open_ns(date),
            "timestamp.lte": market_close_ns(date),
            "order": "desc",
            "sort": "timestamp",
            "limit": 1,
        }
        data = self._get(f"{self.base_url}/v3/quotes/{option_ticker}", params)
        results = data.get("results") or []
        return results[0] if results else None

    # --- acción subyacente -----------------------------------------------------------------------
    def stock_close(self, ticker: str, date) -> float | None:
        """Cierre diario sin ajustar por splits posteriores, para que coincida con los strikes de ese día."""
        d = str(date)
        params = {"adjusted": "false", "sort": "asc", "limit": 10}
        data = self._get(f"{self.base_url}/v2/aggs/ticker/{ticker}/range/1/day/{d}/{d}", params)
        results = data.get("results") or []
        return float(results[0]["c"]) if results else None
