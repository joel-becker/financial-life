"""Tests for consumption policy in PersonalFinanceModel."""

import unittest

import numpy as np

from models.personal_finance import PersonalFinanceModel
from tests.test_tax_flow import make_params


class TestConsumptionFloor(unittest.TestCase):
    def test_minimum_consumption_is_constant_in_real_terms(self):
        """The model runs entirely in real terms, so the consumption floor
        must not be deflated again — it applies at full value in every
        year regardless of inflation or t."""
        model = PersonalFinanceModel(make_params(minimum_consumption=25000))
        # Nonzero inflation on record; late year; resources below floor
        model.inflation = np.full((1, 3), 0.03)
        model.total_wealth[:, 2] = 0
        model.cash[:, 2] = 100000
        model.market[:, 2] = 0
        model.retirement_account[:, 2] = 0
        low_income = np.array([1000.0])
        consumption = model.calculate_consumption_amount(
            2, low_income, is_retired=np.array([False]), years_left=30
        )
        self.assertAlmostEqual(consumption[0], 25000.0, places=6)


if __name__ == "__main__":
    unittest.main()
