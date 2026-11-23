"""Guardado en disco de las respuestas crudas de una API, para no volver a pedir nunca lo mismo.

Cada pedido (endpoint + parámetros) se guarda tal como llegó, antes de procesarlo:

    <raíz>/<fuente>/<endpoint>_<símbolo>_<fecha>_<hash>.csv        respuesta correcta
    <raíz>/<fuente>/<endpoint>_<símbolo>_<fecha>_<hash>.err.json   error definitivo (p. ej. "sin datos")
    <raíz>/<fuente>/index.jsonl                                    bitácora: qué se pidió, cuándo, cuántos bytes

Todo lo demás (selección de contratos, limpieza, cadenas por evento) se calcula desde esos archivos, así que
cambiar las reglas no cuesta un solo pedido. Los errores transitorios (límite de pedidos, caídas) no se guardan,
porque conviene reintentarlos; los definitivos sí, para no pedir de nuevo lo que ya se sabe que no existe.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable


class DefinitiveError(RuntimeError):
    """Error que no mejora al reintentar (por ejemplo, 'sin datos para esa fecha')."""

    def __init__(self, status: int, message: str):
        super().__init__(f"{status}: {message}")
        self.status, self.message = status, message


class RawCache:
    def __init__(self, root: Path, source: str):
        self.dir = Path(root) / source
        self.dir.mkdir(parents=True, exist_ok=True)
        self.index = self.dir / "index.jsonl"
        self._lock = threading.Lock()

    @staticmethod
    def _key(endpoint: str, params: dict) -> str:
        blob = json.dumps({"e": endpoint, "p": params}, sort_keys=True, default=str)
        return hashlib.sha1(blob.encode()).hexdigest()[:12]

    def _stem(self, endpoint: str, params: dict) -> Path:
        tag = "_".join(str(params[k]) for k in ("symbol", "start_date") if k in params)
        name = f"{endpoint.strip('/').replace('/', '_')}_{tag}_{self._key(endpoint, params)}"
        return self.dir / name.replace("*", "ALL")

    def _log(self, **row) -> None:
        row["ts"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        with self._lock, open(self.index, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")

    def fetch(self, endpoint: str, params: dict, fn: Callable[[], str], ext: str = "csv") -> str:
        """Devuelve el texto de la respuesta: del disco si ya está, o de `fn()` (que se guarda antes de devolver)."""
        stem = self._stem(endpoint, params)
        ok, err = stem.with_suffix(f".{ext}"), stem.with_suffix(".err.json")
        if ok.exists():
            return ok.read_text(encoding="utf-8")
        if err.exists():
            e = json.loads(err.read_text(encoding="utf-8"))
            raise DefinitiveError(e["status"], e["message"])
        try:
            text = fn()
        except DefinitiveError as exc:
            err.write_text(json.dumps({"status": exc.status, "message": exc.message}, ensure_ascii=False), encoding="utf-8")
            self._log(endpoint=endpoint, params=params, file=err.name, status="error", code=exc.status)
            raise
        tmp = ok.with_suffix(ok.suffix + ".tmp")
        tmp.write_text(text, encoding="utf-8")
        os.replace(tmp, ok)  # escritura atómica: un corte no deja archivos a medias
        self._log(endpoint=endpoint, params=params, file=ok.name, status="ok", bytes=len(text.encode("utf-8")))
        return text

    def clear_errors(self) -> int:
        """Borra los errores guardados (útil si cambió el plan o se arregló algo) y devuelve cuántos eran."""
        files = list(self.dir.glob("*.err.json"))
        for f in files:
            f.unlink()
        return len(files)
