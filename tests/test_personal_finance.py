import unittest
import numpy as np
import sys
import os

print(f"Python version: {sys.version}")
print(f"Current working directory: {os.getcwd()}")
print(f"PYTHONPATH: {sys.path}")
# Add the project root directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.personal_finance import PersonalFinanceModel
from config.parameters import input_params

class TestPersonalFinanceModel(unittest.TestCase):
    def setUp(self):
        # Set up a basic model for testing
        self.input_params = input_params
        self.model = PersonalFinanceModel(self.input_params)

    def test_initialization(self):
        self.assertEqual(self.model.m, input_params["m"])
        self.assertEqual(self.model.years, input_params["years"])
        self.assertEqual(self.model.r, input_params["r"])

    def test_generate_market_returns(self):
        returns = self.model.generate_market_returns()
        self.assertEqual(returns.shape, (input_params["m"], input_params["years"]))
        self.assertTrue(np.all(returns > -1))  # Returns should be greater than -100%

    def test_generate_ar_inflation(self):
        inflation = self.model.generate_ar_inflation()
        self.assertEqual(inflation.shape, (input_params["m"], input_params["years"]))
        self.assertTrue(np.all(inflation > -1))  # Inflation should be positive

    def test_simulate(self):
        self.model.simulate()
        results = self.model.get_results()
        
        # Check that all result arrays have the correct shape
        for key, value in results.items():
            self.assertEqual(value.shape, (input_params["m"], input_params["years"]), f"{key} has incorrect shape")

        # Check that financial wealth is non-negative
        self.assertTrue(np.all(results['financial_wealth'] >= 0))

        # Check that consumption is always positive
        self.assertTrue(np.all(results['consumption'] > 0))

    def test_calculate_charitable_donations(self):
        self.model.charitable_giving_rate = 0.05
        self.model.charitable_giving_cap = 10000
        total_real_income = np.array([100000, 200000, 300000])
        donations = self.model.calculate_charitable_donations(0, total_real_income)
        expected_donations = np.array([5000, 10000, 10000])  # Cap should be applied to last value
        np.testing.assert_array_almost_equal(donations, expected_donations)

if __name__ == '__main__':
    unittest.main()