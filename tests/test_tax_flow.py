"""Tests for the annual tax flow in PersonalFinanceModel.

Covers the tax treatment of retirement contributions, withdrawals,
charitable donations, and capital gains in calculate_after_tax_income
and simulate_year.
"""

import unittest

import numpy as np

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


class TestAfterTaxIncome(unittest.TestCase):
    def test_withdrawals_are_taxed_as_income(self):
        """Pre-tax retirement withdrawals must increase taxable income."""
        model = make_model = PersonalFinanceModel(make_params())
        model.retirement_contributions[:, 0] = 0
        model.retirement_withdrawals[:, 0] = 0
        model.calculate_after_tax_income(0, np.array([50000.0]))
        tax_without_withdrawal = model.tax_paid[0, 0]

        model = PersonalFinanceModel(make_params())
        model.retirement_contributions[:, 0] = 0
        model.retirement_withdrawals[:, 0] = 20000
        model.calculate_after_tax_income(0, np.array([50000.0]))
        tax_with_withdrawal = model.tax_paid[0, 0]

        self.assertGreater(tax_with_withdrawal, tax_without_withdrawal)
        self.assertEqual(model.real_taxable_income[0, 0], 70000.0)

    def test_taxable_income_formula(self):
        """taxable = income - contributions + withdrawals; the donation
        deduction is applied inside TaxSystem, not subtracted here too."""
        model = PersonalFinanceModel(make_params())
        model.retirement_contributions[:, 0] = 5000
        model.retirement_withdrawals[:, 0] = 2000
        model.charitable_donations[:, 0] = 3000
        model.calculate_after_tax_income(0, np.array([100000.0]))
        self.assertEqual(model.real_taxable_income[0, 0], 97000.0)

    def test_donations_deducted_exactly_once(self):
        """Tax must equal TaxSystem applied to (income - contributions +
        withdrawals) with donations passed through for deduction once."""
        model = PersonalFinanceModel(make_params())
        model.retirement_contributions[:, 0] = 5000
        model.charitable_donations[:, 0] = 3000
        model.calculate_after_tax_income(0, np.array([100000.0]))
        expected_tax = model.tax_system.calculate_tax(
            np.array([95000.0]), np.array([0.0]), np.array([3000.0])
        )
        self.assertAlmostEqual(model.tax_paid[0, 0], expected_tax[0], places=6)

    def test_does_not_modify_contributions(self):
        """calculate_after_tax_income must not recompute or overwrite the
        contributions already set by simulate_year."""
        model = PersonalFinanceModel(make_params())
        model.retirement_contributions[:, 0] = 1234.0
        model.calculate_after_tax_income(0, np.array([100000.0]))
        self.assertEqual(model.retirement_contributions[0, 0], 1234.0)

    def test_spendable_income_accounting(self):
        """after-tax income = income + withdrawals - tax - contributions
        - donations (withdrawn dollars are spendable; contributed and
        donated dollars are not)."""
        model = PersonalFinanceModel(make_params())
        model.retirement_contributions[:, 0] = 5000
        model.retirement_withdrawals[:, 0] = 2000
        model.charitable_donations[:, 0] = 3000
        after_tax = model.calculate_after_tax_income(0, np.array([100000.0]))
        expected = 100000.0 + 2000.0 - model.tax_paid[0, 0] - 5000.0 - 3000.0
        self.assertAlmostEqual(after_tax[0], expected, places=6)


class TestSimulateYearTaxFlow(unittest.TestCase):
    def test_retirees_make_no_contributions(self):
        """After retirement, no phantom contributions may be recorded or
        deducted from taxable income."""
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
        self.assertEqual(model.retirement_contributions[0, 1], 0.0)

    def test_capital_gains_taxed_in_year_earned(self):
        """Market gains in year t must be visible to the tax calculation
        for year t (not always zero)."""
        params = make_params(charitable_giving_rate=0.0)
        model = PersonalFinanceModel(params)
        model.income = np.full((1, 3), 100000.0)
        model.inflation = np.zeros((1, 3))
        model.initialize_simulation()
        returns = np.full((1, 3), 0.10)
        model.simulate_year(0, returns)
        # market_start=50000, 10% return -> 5000 of gains
        self.assertAlmostEqual(model.capital_gains[0, 0], 5000.0, places=6)
        # and those gains must have fed the tax computation
        model_no_gains = PersonalFinanceModel(params)
        model_no_gains.income = np.full((1, 3), 100000.0)
        model_no_gains.inflation = np.zeros((1, 3))
        model_no_gains.initialize_simulation()
        model_no_gains.simulate_year(0, np.zeros((1, 3)))
        self.assertGreater(model.tax_paid[0, 0], model_no_gains.tax_paid[0, 0])


if __name__ == "__main__":
    unittest.main()
