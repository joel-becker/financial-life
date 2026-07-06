import numpy as np
import pytest

from models.agi_impact import AGIModel


def test_valid_cdf_points_accepted():
    model = AGIModel(
        {2: 0.1, 5: 0.3, 10: 0.5, 20: 0.8},
        wage_multiplier=2.0,
        returns_multiplier=1.5,
        m=10,
        years=30,
    )
    assert model.get_wage_multipliers().shape == (10, 30)


def test_non_monotonic_cdf_rejected():
    # P(AGI within 5y) > P(AGI within 10y) is contradictory input
    # (possible via independent UI sliders) and must fail loudly
    with pytest.raises(ValueError):
        AGIModel(
            {2: 0.4, 5: 0.6, 10: 0.3, 20: 0.8},
            wage_multiplier=2.0,
            returns_multiplier=1.5,
            m=10,
            years=30,
        )


def test_probability_out_of_range_rejected():
    with pytest.raises(ValueError):
        AGIModel(
            {2: 0.1, 5: 1.3},
            wage_multiplier=2.0,
            returns_multiplier=1.5,
            m=10,
            years=30,
        )


def test_multipliers_apply_from_arrival_year():
    np.random.seed(0)
    model = AGIModel(
        {1: 1.0},  # AGI arrives within a year with certainty
        wage_multiplier=3.0,
        returns_multiplier=2.0,
        m=5,
        years=10,
    )
    assert np.all(model.get_wage_multipliers()[:, 1:] == 3.0)
