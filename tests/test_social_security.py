import numpy as np
import pytest

from models.personal_finance import PersonalFinanceModel
from tests.test_tax_flow import make_params


def make_retired_model(career_years, annual_income, total_years=45):
    # model at claim age with a career of the given length
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


def test_short_career_earns_less_than_35_year_career():
    # AIME averages the top 35 years including zeros, so a 10-year career
    # at the same salary must produce a smaller benefit
    short = make_retired_model(career_years=10, annual_income=80000.0)
    long = make_retired_model(career_years=35, annual_income=80000.0)
    benefit_short = short.calculate_us_social_security(37, np.array([67]))
    benefit_long = long.calculate_us_social_security(37, np.array([67]))
    assert benefit_short[0] < benefit_long[0]
    assert benefit_short[0] > 0


def test_career_beyond_35_years_uses_top_35():
    # years beyond the top 35 must not change the average when all years
    # pay the same
    career_35 = make_retired_model(35, 80000.0, total_years=45)
    career_37 = make_retired_model(37, 80000.0, total_years=45)
    b35 = career_35.calculate_us_social_security(40, np.array([67]))
    b37 = career_37.calculate_us_social_security(40, np.array([67]))
    assert b35[0] == pytest.approx(b37[0], abs=0.01)


def test_before_claim_age_is_zero():
    model = make_retired_model(35, 80000.0)
    benefit = model.calculate_us_social_security(20, np.array([50]))
    assert benefit[0] == 0.0


def test_exact_pia_all_three_bands():
    # AIME 6666.67 (35y at 80k): 0.9*1115 + 0.32*(6666.67-1115) = 2780.03/mo
    model = make_retired_model(35, 80000.0)
    benefit = model.calculate_us_social_security(37, np.array([67]))
    aime = 80000.0 * 35 / (35 * 12)
    expected_monthly = 0.9 * 1115.0 + 0.32 * (aime - 1115.0)
    assert benefit[0] == pytest.approx(expected_monthly * 12, abs=1.0)


def test_early_claiming_reduces_benefit():
    # claiming at 62 must pay ~30% less than at FRA (67), never more
    def benefit_at(claim_age):
        params = make_params(
            m=1,
            years=45,
            years_until_retirement=32,
            years_until_death=45,
            claim_age=claim_age,
            current_age=30,
        )
        model = PersonalFinanceModel(params)
        model.income = np.zeros((1, 45))
        model.income[0, :32] = 60000.0
        return model.calculate_us_social_security(37, np.array([claim_age]))[0]

    at_62, at_67 = benefit_at(62), benefit_at(67)
    assert at_62 < at_67
    assert at_62 / at_67 == pytest.approx(0.70, abs=0.01)
