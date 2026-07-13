import unittest
from unittest.mock import patch
import numpy as np
from models.personal_finance import PersonalFinanceModel
from models.income_paths import ConstantRealIncomePath

class TestPersonalFinanceModel(unittest.TestCase):

    def setUp(self):
        self.base_params = {
            "m": 1,
            "years": 2,
            "r": 0.02,
            "years_until_retirement": 30,
            "years_until_death": 60,
            "claim_age": 67,
            "current_age": 30,
            "retirement_income": 40000,
            "income_path": ConstantRealIncomePath(100000, 0),
            "min_income": 30000,
            "inflation_rate": 0.02,
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
            "charitable_giving_rate": 0.01,
            "charitable_giving_cap": 10000,
            "tax_region": "California",
            "portfolio_weights": [1.0],
            "asset_returns": [0.07],
            "asset_volatilities": [0.15],
            "asset_correlations": [[1.0]]
        }
        self.model = PersonalFinanceModel(self.base_params)

    def test_initialize_simulation(self):
        self.model.initialize_simulation()
        self.assertEqual(self.model.cash[0, 0], 20000)
        self.assertEqual(self.model.market[0, 0], 50000)
        self.assertEqual(self.model.retirement_account[0, 0], 100000)
        self.assertEqual(self.model.financial_wealth[0, 0], 170000)

    def test_calculate_total_income(self):
        self.model.income = np.array([[100000, 102000]])
        self.model.pension_income = np.array([[0, 0]])
        total_income = self.model.calculate_total_income(0, 30)
        self.assertEqual(total_income[0], 100000)

    def test_calculate_retirement_contribution(self):
        contribution = self.model.calculate_retirement_contribution(0, 100000, 30)
        self.assertEqual(contribution, 5000)

    def test_calculate_after_tax_income(self):
        self.model.capital_gains = np.array([[1000, 1000]])
        self.model.charitable_donations = np.array([[1000, 1000]])
        self.model.retirement_contributions = np.array([[5000, 5000]])
        after_tax_income = self.model.calculate_after_tax_income(0, 100000)
        # The exact value will depend on the tax calculation, but we can check it's less than the pre-tax income
        self.assertLess(after_tax_income[0], 100000)

    def test_update_wealth(self):
        self.model.initialize_simulation()
        self.model.savings = np.array([[10000, 10000]])
        self.model.retirement_contributions = np.array([[5000, 5000]])
        self.model.retirement_withdrawals = np.array([[0, 0]])
        self.model.capital_gains = np.array([[3500, 3500]])
        real_market_returns = np.array([[0.05, 0.05]])

        print("Initial state:")
        initial_cash = self.model.cash[0, 0]
        initial_market = self.model.market[0, 0]
        initial_retirement = self.model.retirement_account[0, 0]
        initial_wealth = self.model.financial_wealth[0, 0]
        print(f"Cash: {initial_cash}")
        print(f"Market: {initial_market}")
        print(f"Retirement account: {initial_retirement}")
        print(f"Financial wealth: {initial_wealth}")

        self.model.update_wealth(0, real_market_returns, False)

        print("\nAfter update_wealth:")
        print(f"Cash: {self.model.cash[0, 0]}")
        print(f"Market: {self.model.market[0, 0]}")
        print(f"Retirement account: {self.model.retirement_account[0, 0]}")
        print(f"Financial wealth: {self.model.financial_wealth[0, 0]}")

        # Check individual components
        self.assertGreater(self.model.cash[0, 0], initial_cash, "Cash should increase")
        self.assertGreater(self.model.market[0, 0], initial_market, "Market investments should increase")
        self.assertGreater(self.model.retirement_account[0, 0], initial_retirement, "Retirement account should increase")

        # Check overall financial wealth
        final_wealth = self.model.financial_wealth[0, 0]
        print(f"\nInitial wealth: {initial_wealth}")
        print(f"Final wealth: {final_wealth}")
        self.assertGreater(final_wealth, initial_wealth, "Overall financial wealth should increase")

        # Calculate expected wealth increase
        expected_increase = (
            self.model.savings[0, 0] +  # Savings
            initial_market * real_market_returns[0, 0] +  # Market returns
            initial_retirement * real_market_returns[0, 0] +  # Retirement account returns
            self.model.retirement_contributions[0, 0]  # Retirement contributions
        )
        print(f"Expected increase: {expected_increase}")
        print(f"Actual increase: {final_wealth - initial_wealth}")

        self.assertAlmostEqual(final_wealth - initial_wealth, expected_increase, places=2, 
                            msg="Wealth increase should match expected increase")

    def test_update_wealth_cash_exceeds_max(self):
        params = self.base_params.copy()
        params["cash_start"] = 60000  # Exceeds max_cash_threshold
        model = PersonalFinanceModel(params)
        model.initialize_simulation()
        model.savings = np.array([[10000, 10000]])
        real_market_returns = np.array([[0.05, 0.05]])

        initial_cash = model.cash[0, 0]
        initial_market = model.market[0, 0]
        model.update_wealth(0, real_market_returns, False, True)

        self.assertEqual(model.cash[0, 0], params["max_cash_threshold"], 
                         "Cash should be capped at max_cash_threshold")
        self.assertGreater(model.market[0, 0], initial_market, 
                           "Excess cash should be moved to market")

    def test_update_wealth_cash_below_min(self):
        params = self.base_params.copy()
        params["cash_start"] = 15000  # Just above min_cash_threshold
        params["market_start"] = 10000
        model = PersonalFinanceModel(params)
        model.initialize_simulation()
        
        # Set high consumption to potentially push cash below minimum
        model.consumption = np.array([[14000, 14000]])
        model.savings = np.array([[-14000, -14000]])  # Negative savings due to high consumption
        real_market_returns = np.array([[0.05, 0.05]])

        initial_cash = model.cash[0, 0]
        initial_market = model.market[0, 0]
        initial_total = initial_cash + initial_market

        print(f"Initial cash: {initial_cash}")
        print(f"Initial market: {initial_market}")
        print(f"Initial total: {initial_total}")
        print(f"Consumption: {model.consumption[0, 0]}")
        print(f"Savings: {model.savings[0, 0]}")

        model.update_wealth(0, real_market_returns, False, True)

        final_cash = model.cash[0, 0]
        final_market = model.market[0, 0]
        final_total = final_cash + final_market

        print(f"Final cash: {final_cash}")
        print(f"Final market: {final_market}")
        print(f"Final total: {final_total}")

        self.assertEqual(final_cash, params["min_cash_threshold"], 
                         "Cash should be maintained at min_cash_threshold")
        self.assertLess(final_total, initial_total, 
                        "Total liquid assets should decrease due to high consumption")
        self.assertGreaterEqual(final_market, 0, 
                                "Market value should not become negative")
        self.assertAlmostEqual(final_total, initial_total + model.savings[0, 0] + initial_market * real_market_returns[0, 0], 
                               places=2, msg="Total change should match savings and returns")
    
    def test_simulate_year(self):
        self.model.initialize_simulation()
        real_market_returns = np.array([[0.05, 0.05]])
        self.model.simulate_year(0, real_market_returns)
        
        # Check that key variables have been updated
        self.assertGreater(self.model.consumption[0, 0], 0)
        self.assertGreater(self.model.savings[0, 0], 0)
        self.assertGreater(self.model.tax_paid[0, 0], 0)

    @patch('models.personal_finance.PersonalFinanceModel.generate_market_returns')
    @patch('models.personal_finance.PersonalFinanceModel.generate_ar_inflation')
    @patch('models.personal_finance.PersonalFinanceModel.generate_income')
    def test_simulate(self, mock_generate_income, mock_generate_ar_inflation, mock_generate_market_returns):
        mock_generate_market_returns.return_value = np.array([[0.05, 0.05]])
        mock_generate_ar_inflation.return_value = np.array([[0.02, 0.02]])
        mock_generate_income.return_value = np.array([[100000, 102000]])
        
        self.model.simulate()
        
        # Check that simulation has run and produced results
        self.assertIsNotNone(self.model.total_wealth)
        self.assertEqual(self.model.total_wealth.shape, (1, 2))

if __name__ == '__main__':
    unittest.main()