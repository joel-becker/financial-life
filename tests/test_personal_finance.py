import numpy as np

from config.parameters import input_params
from models.personal_finance import PersonalFinanceModel


def test_initialization():
    model = PersonalFinanceModel(input_params)
    assert model.m == input_params["m"]
    assert model.years == input_params["years"]
    assert model.r == input_params["r"]


def test_generate_market_returns_shape_and_bounds():
    model = PersonalFinanceModel(input_params)
    returns = model.generate_market_returns()
    assert returns.shape == (input_params["m"], input_params["years"])
    assert np.all(returns > -1)


def test_generate_ar_inflation_shape_and_bounds():
    model = PersonalFinanceModel(input_params)
    inflation = model.generate_ar_inflation()
    assert inflation.shape == (input_params["m"], input_params["years"])
    assert np.all(inflation > -1)


def test_simulate_shapes_and_invariants():
    model = PersonalFinanceModel(input_params)
    model.simulate()
    results = model.get_results()
    for key, value in results.items():
        assert value.shape == (input_params["m"], input_params["years"]), (
            f"{key} has incorrect shape"
        )
    assert np.all(results["financial_wealth"] >= 0)
    assert np.all(results["consumption"] > 0)


def test_calculate_charitable_donations_cap():
    model = PersonalFinanceModel(input_params)
    model.charitable_giving_rate = 0.05
    model.charitable_giving_cap = 10000
    donations = model.calculate_charitable_donations(
        0, np.array([100000.0, 200000.0, 300000.0])
    )
    np.testing.assert_array_almost_equal(donations, np.array([5000.0, 10000.0, 10000.0]))
