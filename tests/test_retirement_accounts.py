"""Tests for RetirementAccounts."""

import unittest

import numpy as np

from models.retirement_accounts import RetirementAccounts


class TestRMD(unittest.TestCase):
    def setUp(self):
        self.accounts = RetirementAccounts("California")

    def test_rmd_before_rmd_age_is_zero(self):
        rmd = self.accounts.calculate_rmd(np.array([100000.0]), np.array([70]))
        self.assertEqual(rmd[0], 0.0)

    def test_rmd_at_72_uses_remaining_horizon(self):
        rmd = self.accounts.calculate_rmd(np.array([180000.0]), np.array([72]))
        self.assertAlmostEqual(rmd[0], 180000.0 / 18, places=6)

    def test_rmd_at_and_beyond_90_is_finite_and_positive(self):
        """Ages >= 90 must not divide by zero or produce negative RMDs."""
        balances = np.array([100000.0, 100000.0])
        ages = np.array([90, 95])
        rmd = self.accounts.calculate_rmd(balances, ages)
        self.assertTrue(np.all(np.isfinite(rmd)))
        self.assertTrue(np.all(rmd > 0))
        self.assertTrue(np.all(rmd <= balances))

    def test_uk_has_no_rmd(self):
        uk = RetirementAccounts("UK")
        rmd = uk.calculate_rmd(np.array([100000.0]), np.array([75]))
        self.assertEqual(rmd[0], 0.0)


class TestContributionLimits(unittest.TestCase):
    def test_us_401k_limit_applies(self):
        accounts = RetirementAccounts("California")
        contribution = accounts.calculate_contribution(
            np.array([500000.0]), np.array([40]), 0.1
        )
        self.assertEqual(contribution[0], 22500.0)

    def test_us_catchup_after_50(self):
        accounts = RetirementAccounts("California")
        contribution = accounts.calculate_contribution(
            np.array([500000.0]), np.array([55]), 0.1
        )
        self.assertEqual(contribution[0], 30000.0)


if __name__ == "__main__":
    unittest.main()
