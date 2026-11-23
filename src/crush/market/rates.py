"""Tasa libre de riesgo de cada evento.

Si la fuente la entrega, `download_event` guarda `r_pre` y `r_post` (decimales anuales) en la meta del evento. Si no
(o si el día queda fuera del plan, p. ej. antes de 2024 en ThetaData gratis), se usa la tasa de respaldo.
"""

from __future__ import annotations

DEFAULT_RATE = 0.04


def event_rates(meta: dict, default: float = DEFAULT_RATE) -> tuple[float, float]:
    """(r del cierre previo, r del cierre posterior)."""
    r_pre = meta.get("r_pre")
    r_post = meta.get("r_post")
    return (default if r_pre is None else float(r_pre), default if r_post is None else float(r_post))
