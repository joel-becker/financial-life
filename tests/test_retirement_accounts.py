import numpy as np
import pytest

from models.retirement_accounts import RetirementAccounts


# RMDs

def test_rmd_before_rmd_age_is_zero():
    accounts = RetirementAccounts("California")
    rmd = accounts.calculate_rmd(np.array([100000.0]), np.array([70]))
    assert rmd[0] == 0.0


def test_rmd_starts_at_73_secure_2_0():
    accounts = RetirementAccounts("California")
    # SECURE 2.0 moved the RMD age to 73
    assert accounts.calculate_rmd(np.array([180000.0]), np.array([72]))[0] == 0.0
    rmd = accounts.calculate_rmd(np.array([180000.0]), np.array([73]))
    assert rmd[0] == pytest.approx(180000.0 / 17)


def test_rmd_at_and_beyond_90_is_finite_and_positive():
    # ages >= 90 must not divide by zero or produce negative RMDs
    accounts = RetirementAccounts("California")
    balances = np.array([100000.0, 100000.0])
    rmd = accounts.calculate_rmd(balances, np.array([90, 95]))
    assert np.all(np.isfinite(rmd))
    assert np.all(rmd > 0)
    assert np.all(rmd <= balances)


def test_uk_has_no_rmd():
    uk = RetirementAccounts("UK")
    rmd = uk.calculate_rmd(np.array([100000.0]), np.array([75]))
    assert rmd[0] == 0.0


# Contribution limits

def test_us_401k_limit_applies():
    accounts = RetirementAccounts("California")
    contribution = accounts.calculate_contribution(
        np.array([500000.0]), np.array([40]), 0.1
    )
    assert contribution[0] == 24500.0


def test_us_catchup_after_50():
    accounts = RetirementAccounts("California")
    contribution = accounts.calculate_contribution(
        np.array([500000.0]), np.array([55]), 0.1
    )
    assert contribution[0] == 32500.0
