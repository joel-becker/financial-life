import numpy as np
import pytest

from models.tax_system import TaxSystem


# Bracketed federal tax — hand-computed values so silent changes to the
# bracket math or constants fail loudly

def test_federal_tax_at_50k():
    tax = TaxSystem("Texas")
    # 9950*0.10 + (40525-9950)*0.12 + (50000-40525)*0.22
    expected = 995.0 + 30575.0 * 0.12 + 9475.0 * 0.22
    result = tax._calculate_bracketed_tax(np.array([50000.0]), tax.federal_brackets)
    assert result[0] == pytest.approx(expected, abs=0.01)


def test_federal_tax_top_bracket_marginal_rate():
    tax = TaxSystem("Texas")
    at_600k = tax._calculate_bracketed_tax(np.array([600000.0]), tax.federal_brackets)
    at_threshold = tax._calculate_bracketed_tax(
        np.array([523600.0]), tax.federal_brackets
    )
    assert at_600k[0] - at_threshold[0] == pytest.approx(
        (600000 - 523600) * 0.37, abs=0.01
    )


def test_zero_income_zero_tax():
    tax = TaxSystem("California")
    assert tax.calculate_tax(np.array([0.0]))[0] == 0.0


# US total tax

def test_texas_50k_no_state_tax():
    tax = TaxSystem("Texas")
    federal = 995.0 + 30575.0 * 0.12 + 9475.0 * 0.22
    ss = 50000.0 * 0.062
    medicare = 50000.0 * 0.0145
    result = tax.calculate_tax(np.array([50000.0]))
    assert result[0] == pytest.approx(federal + ss + medicare, abs=0.01)


def test_ss_tax_capped_at_wage_base():
    tax = TaxSystem("Texas")
    ss_at_cap = 142800.0 * 0.062
    for income in [142800.0, 200000.0]:
        total = tax.calculate_tax(np.array([income]))
        federal = tax._calculate_bracketed_tax(np.array([income]), tax.federal_brackets)
        medicare = income * 0.0145
        assert total[0] - federal[0] - medicare == pytest.approx(ss_at_cap, abs=0.01)


def test_charitable_deduction_capped_at_60pct_agi():
    tax = TaxSystem("Texas")
    income = np.array([100000.0])
    above_cap = tax.calculate_tax(income, 0, np.array([90000.0]))
    at_cap = tax.calculate_tax(income, 0, np.array([60000.0]))
    assert above_cap[0] == pytest.approx(at_cap[0], abs=0.01)


# UK tax

def test_uk_50k():
    tax = TaxSystem("UK")
    income_tax = (50000.0 - 12570.0) * 0.20
    ni = (50000.0 - 9568.0) * 0.12
    result = tax.calculate_tax(np.array([50000.0]))
    assert result[0] == pytest.approx(income_tax + ni, abs=0.01)


def test_uk_below_personal_allowance():
    tax = TaxSystem("UK")
    result = tax.calculate_tax(np.array([10000.0]))
    expected_ni = (10000.0 - 9568.0) * 0.12
    assert result[0] == pytest.approx(expected_ni, abs=0.01)
