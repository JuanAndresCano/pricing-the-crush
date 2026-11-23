"""Control de ritmo: garantiza un mínimo de segundos entre pedidos, también con varios hilos."""

from __future__ import annotations

import threading
import time


class RateLimiter:
    """`wait()` bloquea hasta que hayan pasado `min_interval` segundos desde el pedido anterior.

    El reloj y la pausa se pueden inyectar para probar sin esperar de verdad.
    """

    def __init__(self, min_interval: float, clock=time.monotonic, sleep=time.sleep):
        self.min_interval = float(min_interval)
        self._clock, self._sleep = clock, sleep
        self._last: float | None = None
        self._lock = threading.Lock()

    def wait(self) -> None:
        with self._lock:  # se mantiene el candado durante la pausa: los hilos hacen fila
            if self._last is not None:
                remaining = self.min_interval - (self._clock() - self._last)
                if remaining > 0:
                    self._sleep(remaining)
            self._last = self._clock()
