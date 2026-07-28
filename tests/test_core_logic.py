import numpy as np
import pytest

from models.personal_finance import PersonalFinanceModel
from tests.test_tax_flow import make_params


def make_two_year_model(**overrides):
    params = make_params(
        m=1,
        years=2,
        years_until_retirement=30,
        years_until_death=60,
        min_income=30000,
        income_fraction_consumed_before_retirement=0.7,
        wealth_fraction_consumed_before_retirement=0.05,
        charitable_giving_rate=0.01,
    )
    params.update(overrides)
    return PersonalFinanceModel(params)


def test_initialize_simulation():
    model = make_two_year_model()
    model.initialize_simulation()
    assert model.cash[0, 0] == 20000
    assert model.market[0, 0] == 50000
    assert model.retirement_account[0, 0] == 100000
    assert model.financial_wealth[0, 0] == 170000


def test_calculate_total_income():
    model = make_two_year_model()
    model.income = np.array([[100000.0, 102000.0]])
    model.pension_income = np.array([[0.0, 0.0]])
    total_income = model.calculate_total_income(0, 30)
    assert total_income[0] == 100000


def test_calculate_retirement_contribution():
    model = make_two_year_model()
    contribution = model.calculate_retirement_contribution(0, 100000, 30)
    assert contribution == 5000


def test_calculate_after_tax_income_less_than_pre_tax():
    model = make_two_year_model()
    model.capital_gains = np.array([[1000.0, 1000.0]])
    model.charitable_donations = np.array([[1000.0, 1000.0]])
    model.retirement_contributions = np.array([[5000.0, 5000.0]])
    after_tax_income = model.calculate_after_tax_income(0, np.array([100000.0]))
    assert after_tax_income[0] < 100000


def test_update_wealth_exact_increase():
    model = make_two_year_model()
    model.initialize_simulation()
    model.savings = np.array([[10000.0, 10000.0]])
    model.retirement_contributions = np.array([[5000.0, 5000.0]])
    model.retirement_withdrawals = np.array([[0.0, 0.0]])
    model.capital_gains = np.array([[3500.0, 3500.0]])
    real_market_returns = np.array([[0.05, 0.05]])

    initial_market = model.market[0, 0]
    initial_retirement = model.retirement_account[0, 0]
    initial_wealth = model.financial_wealth[0, 0]

    model.update_wealth(0, real_market_returns, False)

    expected_increase = (
        model.savings[0, 0]
        + initial_market * real_market_returns[0, 0]
        + initial_retirement * real_market_returns[0, 0]
        + model.retirement_contributions[0, 0]
    )
    assert model.financial_wealth[0, 0] - initial_wealth == pytest.approx(
        expected_increase, abs=0.01
    )


def test_update_wealth_cash_capped_at_max():
    model = make_two_year_model(cash_start=60000)  # exceeds max threshold
    model.initialize_simulation()
    model.savings = np.array([[10000.0, 10000.0]])
    initial_market = model.market[0, 0]

    model.update_wealth(0, np.array([[0.05, 0.05]]), False)

    assert model.cash[0, 0] == model.max_cash_threshold
    assert model.market[0, 0] > initial_market  # excess cash moved to market


def test_update_wealth_cash_held_at_min_under_dissaving():
    model = make_two_year_model(cash_start=15000, market_start=10000)
    model.initialize_simulation()
    model.consumption = np.array([[14000.0, 14000.0]])
    model.savings = np.array([[-14000.0, -14000.0]])
    real_market_returns = np.array([[0.05, 0.05]])
    initial_total = model.cash[0, 0] + model.market[0, 0]

    model.update_wealth(0, real_market_returns, False)

    final_total = model.cash[0, 0] + model.market[0, 0]
    assert model.cash[0, 0] == model.min_cash_threshold
    assert final_total < initial_total
    assert model.market[0, 0] >= 0
    assert final_total == pytest.approx(
        initial_total + model.savings[0, 0] + 10000.0 * 0.05, abs=0.01
    )


def test_simulate_year_updates_key_variables():
    model = make_two_year_model()
    model.initialize_simulation()
    model.income = np.array([[100000.0, 102000.0]])
    model.simulate_year(0, np.array([[0.05, 0.05]]))
    assert model.consumption[0, 0] > 0
    assert model.savings[0, 0] > 0
    assert model.tax_paid[0, 0] > 0


def test_simulate_produces_correct_shapes():
    model = make_two_year_model()
    model.generate_market_returns = lambda: np.array([[0.05, 0.05]])
    model.generate_ar_inflation = lambda: np.array([[0.02, 0.02]])
    model.generate_income = lambda: np.array([[100000.0, 102000.0]])
    model.simulate()
    assert model.total_wealth.shape == (1, 2)
