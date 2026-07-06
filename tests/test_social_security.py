"""Tests for the US Social Security benefit calculation."""

import unittest

import numpy as np

from models.personal_finance import PersonalFinanceModel
from tests.test_tax_flow import make_params


def make_retired_model(career_years, annual_income, total_years=45):
    """Model at claim age with a career of the given length."""
    params = make_params(
        m=1,
        years=total_years,
        years_until_retirement=career_years,
        years_until_death=total_years,
        claim_age=67,
        current_age=30,
    )
    model = PersonalFinanceModel(params)
    model.income = np.zeros((1, total_years))
    model.income[0, :career_years] = annual_income
    return model


class TestSocialSecurityAIME(unittest.TestCase):
    def test_short_career_earns_less_than_35_year_career(self):
        """AIME averages the top 35 years including zeros, so a 10-year
        career at the same salary must produce a smaller benefit than a
        35-year career."""
        short = make_retired_model(career_years=10, annual_income=80000.0)
        long = make_retired_model(career_years=35, annual_income=80000.0)
        age_at_claim = np.array([67])
        benefit_short = short.calculate_us_social_security(37, age_at_claim)
        benefit_long = long.calculate_us_social_security(37, age_at_claim)
        self.assertLess(benefit_short[0], benefit_long[0])
        self.assertGreater(benefit_short[0], 0)

    def test_career_beyond_35_years_uses_top_35(self):
        """Years beyond the top 35 must not dilute or inflate the average
        when all years pay the same."""
        career_35 = make_retired_model(35, 80000.0, total_years=45)
        career_37 = make_retired_model(37, 80000.0, total_years=45)
        age_at_claim = np.array([67])
        b35 = career_35.calculate_us_social_security(40, age_at_claim)
        b37 = career_37.calculate_us_social_security(40, age_at_claim)
        self.assertAlmostEqual(b35[0], b37[0], places=2)

    def test_before_claim_age_is_zero(self):
        model = make_retired_model(35, 80000.0)
        benefit = model.calculate_us_social_security(20, np.array([50]))
        self.assertEqual(benefit[0], 0.0)


if __name__ == "__main__":
    unittest.main()
