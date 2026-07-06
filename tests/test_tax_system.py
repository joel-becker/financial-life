"""Exact-value tests for TaxSystem.

These pin the bracket math to hand-computed figures so silent changes to
the tax logic (or its constants) fail loudly.
"""

import unittest

import numpy as np

from models.tax_system import TaxSystem


class TestBracketedTax(unittest.TestCase):
    def test_federal_tax_at_50k(self):
        tax = TaxSystem("Texas")
        # 9950*0.10 + (40525-9950)*0.12 + (50000-40525)*0.22
        expected = 995.0 + 30575.0 * 0.12 + 9475.0 * 0.22
        result = tax._calculate_bracketed_tax(
            np.array([50000.0]), tax.federal_brackets
        )
        self.assertAlmostEqual(result[0], expected, places=2)

    def test_federal_tax_top_bracket(self):
        tax = TaxSystem("Texas")
        result_600k = tax._calculate_bracketed_tax(
            np.array([600000.0]), tax.federal_brackets
        )
        result_523600 = tax._calculate_bracketed_tax(
            np.array([523600.0]), tax.federal_brackets
        )
        # marginal rate above the top threshold must be exactly 37%
        self.assertAlmostEqual(
            result_600k[0] - result_523600[0], (600000 - 523600) * 0.37, places=2
        )

    def test_zero_income_zero_tax(self):
        tax = TaxSystem("California")
        result = tax.calculate_tax(np.array([0.0]))
        self.assertEqual(result[0], 0.0)


class TestUSTotalTax(unittest.TestCase):
    def test_texas_50k_no_state_tax(self):
        tax = TaxSystem("Texas")
        federal = 995.0 + 30575.0 * 0.12 + 9475.0 * 0.22
        ss = 50000.0 * 0.062
        medicare = 50000.0 * 0.0145
        result = tax.calculate_tax(np.array([50000.0]))
        self.assertAlmostEqual(result[0], federal + ss + medicare, places=2)

    def test_ss_tax_capped_at_wage_base(self):
        tax = TaxSystem("Texas")
        r1 = tax.calculate_tax(np.array([142800.0]))
        r2 = tax.calculate_tax(np.array([200000.0]))
        ss_at_cap = 142800.0 * 0.062
        # SS component of the 200k bill must equal the capped amount:
        # subtract federal+medicare differences to isolate it
        fed_1 = tax._calculate_bracketed_tax(np.array([142800.0]), tax.federal_brackets)
        fed_2 = tax._calculate_bracketed_tax(np.array([200000.0]), tax.federal_brackets)
        medicare_1 = 142800.0 * 0.0145
        medicare_2 = 200000.0 * 0.0145
        ss_2 = r2[0] - fed_2[0] - medicare_2
        self.assertAlmostEqual(ss_2, ss_at_cap, places=2)
        self.assertAlmostEqual(r1[0] - fed_1[0] - medicare_1, ss_at_cap, places=2)

    def test_charitable_deduction_capped_at_60pct_agi(self):
        tax = TaxSystem("Texas")
        income = np.array([100000.0])
        capped = tax.calculate_tax(income, 0, np.array([90000.0]))
        at_cap = tax.calculate_tax(income, 0, np.array([60000.0]))
        self.assertAlmostEqual(capped[0], at_cap[0], places=2)


class TestUKTax(unittest.TestCase):
    def test_uk_50k(self):
        tax = TaxSystem("UK")
        income_tax = (50000.0 - 12570.0) * 0.20
        ni = (50000.0 - 9568.0) * 0.12
        result = tax.calculate_tax(np.array([50000.0]))
        self.assertAlmostEqual(result[0], income_tax + ni, places=2)

    def test_uk_below_personal_allowance(self):
        tax = TaxSystem("UK")
        result = tax.calculate_tax(np.array([10000.0]))
        # No income tax below the allowance; NI applies above its threshold
        expected_ni = (10000.0 - 9568.0) * 0.12
        self.assertAlmostEqual(result[0], expected_ni, places=2)


if __name__ == "__main__":
    unittest.main()
