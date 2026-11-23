"""Cliente mínimo de la API REST de Massive (antes Polygon.io).

Endpoints usados (según la documentación de Massive):
  - GET /v3/reference/options/contracts   lista de contratos de opciones
  - GET /v3/quotes/{optionsTicker}        cotizaciones históricas (bid y ask)

La clave se lee de la variable de entorno MASSIVE_API_KEY (archivo .env, nunca en git).
El método de autenticación y la URL base no figuran en las páginas de endpoints, así que
se pueden cambiar con MASSIVE_BASE_URL y MASSIVE_AUTH ("header" o "query").
"""

from __future__ import annotations

import os
from typing import Any, Iterator

import requests
from dotenv import load_dotenv

DEFAULT_BASE_URL = "https://api.massive.com"


class MassiveError(RuntimeError):
    """Error devuelto por la API, con el código HTTP y el mensaje del servidor."""


class MassiveClient:
    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        auth: str | None = None,
        session: requests.Session | None = None,
    ) -> None:
        load_dotenv()
        self.api_key = api_key or os.getenv("MASSIVE_API_KEY", "")
        if not self.api_key:
            raise MassiveError("Falta MASSIVE_API_KEY. Crea el archivo .env a partir de .env.example.")
        self.base_url = (base_url or os.getenv("MASSIVE_BASE_URL") or DEFAULT_BASE_URL).rstrip("/")
        self.auth = (auth or os.getenv("MASSIVE_AUTH") or "header").lower()
        self.session = session or requests.Session()

    def _auth_kwargs(self) -> dict[str, Any]:
        if self.auth == "query":
            return {"params": {"apiKey": self.api_key}}
        return {"headers": {"Authorization": f"Bearer {self.api_key}"}}

    def _get(self, url: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        auth = self._auth_kwargs()
        merged = {**(params or {}), **auth.get("params", {})}
        resp = self.session.get(url, params=merged, headers=auth.get("headers"), timeout=30)
        if resp.status_code != 200:
            # El cuerpo suele explicar si el plan no incluye el endpoint. Nunca incluye la clave.
            raise MassiveError(f"HTTP {resp.status_code} en {url.split('?')[0]}: {resp.text[:300]}")
        return resp.json()

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

    def option_contracts(self, underlying: str, as_of: str, **filters: Any) -> list[dict[str, Any]]:
        """Contratos de opciones de `underlying` vigentes en la fecha `as_of` (YYYY-MM-DD)."""
        params = {"underlying_ticker": underlying, "as_of": as_of, "limit": 1000, **filters}
        return list(self.paginate("/v3/reference/options/contracts", params))

    def option_quotes(self, option_ticker: str, date: str, limit: int = 50000) -> list[dict[str, Any]]:
        """Cotizaciones de un contrato en un día (YYYY-MM-DD)."""
        params = {"timestamp": date, "limit": limit, "order": "asc", "sort": "timestamp"}
        return list(self.paginate(f"/v3/quotes/{option_ticker}", params))
