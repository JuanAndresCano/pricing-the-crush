import pytest

from crush.models.event import split_event_variance


def test_readme_example_section_4():
    # Prueba 7: vol 60% a 5 días y 45% a 10 días -> sigma_b = 21,2%, eps = 0,00625, CR ~ 0,35
    out = split_event_variance(0.60, 5 / 252, 0.45, 10 / 252)
    assert out.sigma_b**2 == pytest.approx(0.045, abs=1e-12)
    assert out.sigma_b == pytest.approx(0.2121, abs=1e-4)
    assert out.eps == pytest.approx(0.00625, abs=1e-10)
    assert out.implied_move == pytest.approx(0.0791, abs=1e-4)
    assert out.crush_ratio == pytest.approx(0.3536, abs=1e-4)


def test_rejects_bad_maturities_and_negative_base_variance():
    with pytest.raises(ValueError):
        split_event_variance(0.6, 10 / 252, 0.45, 5 / 252)
    with pytest.raises(ValueError):
        split_event_variance(0.6, 5 / 252, 0.2, 10 / 252)  # w2 < w1
