import numpy as np
import pytest

from models.tax_system import TaxSystem


# Bracketed federal tax — hand-computed values so silent changes to the
# bracket math or constants fail loudly

def test_federal_tax_at_50k():
    tax = TaxSystem("Texas")
    # 2026 single brackets: 12400*0.10 + (50000-12400)*0.12
    expected = 1240.0 + 37600.0 * 0.12
    result = tax._calculate_bracketed_tax(np.array([50000.0]), tax.federal_brackets)
    assert result[0] == pytest.approx(expected, abs=0.01)


def test_federal_tax_top_bracket_marginal_rate():
    tax = TaxSystem("Texas")
    at_700k = tax._calculate_bracketed_tax(np.array([700000.0]), tax.federal_brackets)
    at_threshold = tax._calculate_bracketed_tax(
        np.array([640600.0]), tax.federal_brackets
    )
    assert at_700k[0] - at_threshold[0] == pytest.approx(
        (700000 - 640600) * 0.37, abs=0.01
    )


def test_zero_income_zero_tax():
    tax = TaxSystem("California")
    assert tax.calculate_tax(np.array([0.0]))[0] == 0.0


# US total tax

def test_texas_50k_no_state_tax():
    tax = TaxSystem("Texas")
    # ordinary taxable = 50000 - 16100 standard deduction = 33900
    federal = 1240.0 + (33900.0 - 12400.0) * 0.12
    ss = 50000.0 * 0.062
    medicare = 50000.0 * 0.0145
    result = tax.calculate_tax(np.array([50000.0]))
    assert result[0] == pytest.approx(federal + ss + medicare, abs=0.01)


def test_ss_tax_capped_at_wage_base():
    tax = TaxSystem("Texas")
    ss_at_cap = 184500.0 * 0.062
    for income in [184500.0, 250000.0]:
        total = tax.calculate_tax(np.array([income]))
        federal = tax._calculate_bracketed_tax(
            np.array([income - 16100.0]), tax.federal_brackets
        )
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
    ni = (50000.0 - 12570.0) * 0.08
    result = tax.calculate_tax(np.array([50000.0]))
    assert result[0] == pytest.approx(income_tax + ni, abs=0.01)


def test_uk_below_personal_allowance():
    tax = TaxSystem("UK")
    # below both the personal allowance and the NI primary threshold
    result = tax.calculate_tax(np.array([10000.0]))
    assert result[0] == pytest.approx(0.0, abs=0.01)


# Capital gains: excluded from ordinary brackets, LTCG stacked on top

def test_gains_not_taxed_at_federal_ordinary_rates():
    tax = TaxSystem("Texas")
    base = tax.calculate_tax(np.array([100000.0]), np.array([0.0]))
    with_gains = tax.calculate_tax(np.array([100000.0]), np.array([10000.0]))
    # At 100k ordinary income, gains land entirely in the 15% LTCG band;
    # Texas has no state tax, so the marginal cost must be exactly 15%
    assert with_gains[0] - base[0] == pytest.approx(10000.0 * 0.15, abs=0.01)


def test_ltcg_brackets_stack_on_ordinary_income():
    tax = TaxSystem("Texas")
    # With zero ordinary income, gains below the 40400 threshold pay 0%
    low = tax.calculate_tax(np.array([0.0]), np.array([30000.0]))
    assert low[0] == pytest.approx(0.0, abs=0.01)
    # With 100k ordinary income the same gains start above the threshold
    base = tax.calculate_tax(np.array([100000.0]), np.array([0.0]))
    stacked = tax.calculate_tax(np.array([100000.0]), np.array([30000.0]))
    assert stacked[0] - base[0] == pytest.approx(30000.0 * 0.15, abs=0.01)


# Payroll tax applies to wages only

def test_no_payroll_tax_on_non_wage_income():
    tax = TaxSystem("Texas")
    all_wages = tax.calculate_tax(np.array([70000.0]))
    no_wages = tax.calculate_tax(np.array([70000.0]), wage_income=np.array([0.0]))
    fica = 70000.0 * (0.062 + 0.0145)
    assert all_wages[0] - no_wages[0] == pytest.approx(fica, abs=0.01)


def test_uk_no_ni_on_non_wage_income():
    tax = TaxSystem("UK")
    all_wages = tax.calculate_tax(np.array([50000.0]))
    no_wages = tax.calculate_tax(np.array([50000.0]), wage_income=np.array([0.0]))
    ni = (50000.0 - 12570.0) * 0.08
    assert all_wages[0] - no_wages[0] == pytest.approx(ni, abs=0.01)


# Standard deduction and capital-loss cap

def test_standard_deduction_zeroes_low_income_federal_tax():
    tax = TaxSystem("Texas")
    # income at the deduction: only payroll taxes remain
    result = tax.calculate_tax(np.array([16100.0]))
    payroll = 16100.0 * (0.062 + 0.0145)
    assert result[0] == pytest.approx(payroll, abs=0.01)


def test_capital_losses_offset_at_most_3000():
    tax = TaxSystem("Texas")
    base = tax.calculate_tax(np.array([100000.0]), np.array([0.0]))
    small_loss = tax.calculate_tax(np.array([100000.0]), np.array([-3000.0]))
    huge_loss = tax.calculate_tax(np.array([100000.0]), np.array([-50000.0]))
    # -50k must be treated exactly like -3k (no carryforward modeled)
    assert huge_loss[0] == pytest.approx(small_loss[0], abs=0.01)
    # and the offset saves tax at the 22% marginal rate
    assert base[0] - small_loss[0] == pytest.approx(3000.0 * 0.22, abs=0.01)
