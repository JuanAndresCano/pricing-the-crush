"""Strike de un variance swap a partir de precios de opciones OTM.

Ecuación (4) del marco teórico, en forma numérica sobre una malla de strikes:

    K_repl = (2/T) e^{rT} [ Int_0^F P(K)/K^2 dK + Int_F^inf C(K)/K^2 dK ]

Esta versión integra con la regla del trapecio sobre la malla dada. Sirve para el control de
código (malla muy fina, colas completas). La versión con strikes de mercado, con recorte y
extrapolación, se agrega con la fase de mercado.
"""

from __future__ import annotations

import numpy as np


def k_repl_from_otm_prices(K, otm_prices, T: float, r: float) -> float:
    """Integra Q(K)/K^2 con el trapecio. `otm_prices[i]` es el put si K[i] < F0 y el call si K[i] >= F0."""
    K = np.asarray(K, dtype=float)
    q = np.asarray(otm_prices, dtype=float)
    return float((2.0 / T) * np.exp(r * T) * np.trapezoid(q / K**2, K))
