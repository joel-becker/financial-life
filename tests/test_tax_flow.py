import numpy as np
import pytest

from models.income_paths import ConstantRealIncomePath
from models.personal_finance import PersonalFinanceModel


def make_params(**overrides):
    params = {
        "m": 1,
        "years": 3,
        "r": 0.02,
        "years_until_retirement": 30,
        "years_until_death": 60,
        "claim_age": 67,
        "current_age": 30,
        "retirement_income": 40000,
        "income_path": ConstantRealIncomePath(100000, 0),
        "min_income": 0,
        "inflation_rate": 0.0,
        "ar_inflation_coefficients": [0],
        "ar_inflation_sd": 0,
        "income_fraction_consumed_before_retirement": 0.7,
        "income_fraction_consumed_after_retirement": 0.9,
        "wealth_fraction_consumed_before_retirement": 0.05,
        "wealth_fraction_consumed_after_retirement": 0.06,
        "min_cash_threshold": 10000,
        "max_cash_threshold": 30000,
        "cash_start": 20000,
        "market_start": 50000,
        "retirement_account_start": 100000,
        "retirement_contribution_rate": 0.05,
        "charitable_giving_rate": 0.0,
        "charitable_giving_cap": 10000,
        "tax_region": "California",
        "portfolio_weights": [1.0],
        "asset_returns": [0.07],
        "asset_volatilities": [0.15],
        "asset_correlations": [[1.0]],
    }
    params.update(overrides)
    return params


# calculate_after_tax_income

def test_withdrawals_are_taxed_as_income():
    # Pre-tax retirement withdrawals must increase taxable income
    model = PersonalFinanceModel(make_params())
    model.retirement_withdrawals[:, 0] = 0
    model.calculate_after_tax_income(0, np.array([50000.0]))
    tax_without_withdrawal = model.tax_paid[0, 0]

    model = PersonalFinanceModel(make_params())
    model.retirement_withdrawals[:, 0] = 20000
    model.calculate_after_tax_income(0, np.array([50000.0]))
    tax_with_withdrawal = model.tax_paid[0, 0]

    assert tax_with_withdrawal > tax_without_withdrawal
    assert model.real_taxable_income[0, 0] == 70000.0


def test_taxable_income_formula():
    # taxable = income - contributions + withdrawals; the donation
    # deduction is applied inside TaxSystem, not subtracted here too
    model = PersonalFinanceModel(make_params())
    model.retirement_contributions[:, 0] = 5000
    model.retirement_withdrawals[:, 0] = 2000
    model.charitable_donations[:, 0] = 3000
    model.calculate_after_tax_income(0, np.array([100000.0]))
    assert model.real_taxable_income[0, 0] == 97000.0


def test_donations_deducted_exactly_once():
    model = PersonalFinanceModel(make_params())
    model.retirement_contributions[:, 0] = 5000
    model.charitable_donations[:, 0] = 3000
    model.calculate_after_tax_income(0, np.array([100000.0]))
    expected_tax = model.tax_system.calculate_tax(
        np.array([95000.0]), np.array([0.0]), np.array([3000.0])
    )
    assert model.tax_paid[0, 0] == pytest.approx(expected_tax[0])


def test_does_not_modify_contributions():
    # must not recompute or overwrite contributions set by simulate_year
    model = PersonalFinanceModel(make_params())
    model.retirement_contributions[:, 0] = 1234.0
    model.calculate_after_tax_income(0, np.array([100000.0]))
    assert model.retirement_contributions[0, 0] == 1234.0


def test_spendable_income_accounting():
    # after-tax = income + withdrawals - tax - contributions - donations
    model = PersonalFinanceModel(make_params())
    model.retirement_contributions[:, 0] = 5000
    model.retirement_withdrawals[:, 0] = 2000
    model.charitable_donations[:, 0] = 3000
    after_tax = model.calculate_after_tax_income(0, np.array([100000.0]))
    expected = 100000.0 + 2000.0 - model.tax_paid[0, 0] - 5000.0 - 3000.0
    assert after_tax[0] == pytest.approx(expected)


# simulate_year tax flow

def test_retirees_make_no_contributions():
    params = make_params(
        years=3,
        years_until_retirement=0,
        years_until_death=30,
        retirement_income=40000,
    )
    model = PersonalFinanceModel(params)
    model.income = np.zeros((1, 3))
    model.inflation = np.zeros((1, 3))
    model.initialize_simulation()
    model.simulate_year(1, np.zeros((1, 3)))
    assert model.retirement_contributions[0, 1] == 0.0


def test_capital_gains_taxed_in_year_earned():
    # gains in year t must be visible to that year's tax calculation
    params = make_params(charitable_giving_rate=0.0)
    model = PersonalFinanceModel(params)
    model.income = np.full((1, 3), 100000.0)
    model.inflation = np.zeros((1, 3))
    model.initialize_simulation()
    model.simulate_year(0, np.full((1, 3), 0.10))
    # market_start=50000, 10% return -> 5000 of gains
    assert model.capital_gains[0, 0] == pytest.approx(5000.0)

    model_no_gains = PersonalFinanceModel(params)
    model_no_gains.income = np.full((1, 3), 100000.0)
    model_no_gains.inflation = np.zeros((1, 3))
    model_no_gains.initialize_simulation()
    model_no_gains.simulate_year(0, np.zeros((1, 3)))
    assert model.tax_paid[0, 0] > model_no_gains.tax_paid[0, 0]
