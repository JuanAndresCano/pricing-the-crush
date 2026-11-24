"""Corrección por comparaciones múltiples (Holm y Benjamini-Hochberg).

Holm controla la probabilidad de al menos un falso positivo (FWER) y vale con cualquier dependencia entre las
pruebas. Benjamini-Hochberg controla la proporción esperada de falsos positivos (FDR) y es menos estricta; con
pruebas dependientes (aquí comparten eventos) es una aproximación.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def holm(p) -> np.ndarray:
    """p-valores ajustados de Holm (paso descendente), en el orden de entrada. Los NaN se dejan en NaN."""
    p = np.asarray(p, float)
    out = np.full(p.shape, np.nan)
    ok = np.isfinite(p)
    m = int(ok.sum())
    if m == 0:
        return out
    idx = np.argsort(p[ok])
    adj = np.maximum.accumulate((m - np.arange(m)) * p[ok][idx])
    res = np.empty(m)
    res[idx] = np.minimum(adj, 1.0)
    out[ok] = res
    return out


def benjamini_hochberg(p) -> np.ndarray:
    """p-valores ajustados (q-valores) de Benjamini-Hochberg, en el orden de entrada. Los NaN se dejan en NaN."""
    p = np.asarray(p, float)
    out = np.full(p.shape, np.nan)
    ok = np.isfinite(p)
    m = int(ok.sum())
    if m == 0:
        return out
    idx = np.argsort(p[ok])
    adj = np.minimum.accumulate((m / (np.arange(m) + 1) * p[ok][idx])[::-1])[::-1]
    res = np.empty(m)
    res[idx] = np.minimum(adj, 1.0)
    out[ok] = res
    return out


def adjust_table(df: pd.DataFrame, family_col: str = "family", p_col: str = "pvalue") -> pd.DataFrame:
    """Agrega p_holm_family / p_bh_family (dentro de cada familia) y p_holm_global / p_bh_global (todas juntas)."""
    out = df.copy()
    out["p_holm_family"] = np.nan
    out["p_bh_family"] = np.nan
    for _, idx in out.groupby(family_col).groups.items():
        out.loc[idx, "p_holm_family"] = holm(out.loc[idx, p_col])
        out.loc[idx, "p_bh_family"] = benjamini_hochberg(out.loc[idx, p_col])
    out["p_holm_global"] = holm(out[p_col])
    out["p_bh_global"] = benjamini_hochberg(out[p_col])
    return out
