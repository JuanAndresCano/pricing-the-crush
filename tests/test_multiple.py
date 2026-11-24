"""Holm y Benjamini-Hochberg."""

import numpy as np
import pytest

from crush.analysis.multiple import adjust_table, benjamini_hochberg, holm


def test_holm_known_values():
    assert holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])  # 3*0.01, max(2*0.03, .03)=.06, 1*0.04 -> .06
    assert holm([0.5, 0.9]) == pytest.approx([1.0, 1.0])


def test_bh_known_values():
    assert benjamini_hochberg([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.04, 0.04])
    assert benjamini_hochberg([0.2]) == pytest.approx([0.2])


def test_adjusted_not_below_raw_and_holm_not_below_bh():
    p = np.array([0.001, 0.02, 0.04, 0.3, 0.7])
    h, b = holm(p), benjamini_hochberg(p)
    assert (h >= p).all() and (b >= p).all() and (h >= b - 1e-12).all() and (h <= 1).all()


def test_nan_is_ignored_and_kept():
    h = holm([0.01, np.nan, 0.02])
    assert np.isnan(h[1]) and h[0] == pytest.approx(0.02) and h[2] == pytest.approx(0.02)


def test_adjust_table_families():
    import pandas as pd

    df = pd.DataFrame({"family": ["A", "A", "B"], "pvalue": [0.01, 0.04, 0.02]})
    t = adjust_table(df)
    assert t.loc[0, "p_holm_family"] == pytest.approx(0.02) and t.loc[2, "p_holm_family"] == pytest.approx(0.02)
    assert t.loc[2, "p_holm_global"] == pytest.approx(0.04)
