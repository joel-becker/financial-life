import numpy as np

class RetirementAccounts:
    """Contribution limits and RMDs for one region (2023 values, treated
    as constant in real terms — see TaxSystem)."""

    def __init__(self, region):
        self.region = region
        self._initialize_account_parameters()

    def _initialize_account_parameters(self):
        if self.region == "UK":
            self._initialize_uk_parameters()
        elif self.region in ["California", "Massachusetts", "New York", "DC", "Texas"]:
            self._initialize_us_parameters()
        else:
            raise ValueError(f"Unsupported region: {self.region}")

    def _initialize_uk_parameters(self):
        # UK 2025/26; the lifetime allowance was abolished in 2024
        self.pension_annual_allowance = 60000

    def _initialize_us_parameters(self):
        # US 2026 limits; RMD age 73 per SECURE 2.0
        self.traditional_401k_limit = 24500
        self.roth_401k_limit = 24500
        self.traditional_ira_limit = 7500
        self.roth_ira_limit = 7500
        self.catchup_contribution_age = 50
        self.catchup_401k = 8000
        self.catchup_ira = 1100
        self.rmd_age = 73

    def calculate_contribution(self, income, age, contribution_rate):
        if self.region == "UK":
            return self._calculate_uk_pension_contribution(income, contribution_rate)
        elif self.region in ["California", "Massachusetts", "New York", "DC", "Texas"]:
            return self._calculate_us_401k_contribution(income, age, contribution_rate)

    def _calculate_uk_pension_contribution(self, income, contribution_rate):
        contribution = income * contribution_rate
        return np.minimum(contribution, self.pension_annual_allowance)

    def _calculate_us_401k_contribution(self, income, age, contribution_rate):
        base_contribution = income * contribution_rate
        limit = self.traditional_401k_limit
        if age >= self.catchup_contribution_age:
            limit += self.catchup_401k
        return np.minimum(base_contribution, limit)

    def calculate_rmd(self, account_balance, age):
        if self.region == "UK":
            return np.zeros_like(account_balance)  # UK pensions don't have RMDs
        elif self.region in ["California", "Massachusetts", "New York", "DC", "Texas"]:
            return self._calculate_us_rmd(account_balance, age)

    def _calculate_us_rmd(self, account_balance, age):
        # Simplified stand-in for the IRS Uniform Lifetime Table: spread the
        # balance over the years to age 90, with a floor of 1 year so ages
        # >= 89 don't divide by zero or go negative
        divisor = np.maximum(90 - age, 1)
        return np.where(age >= self.rmd_age, account_balance / divisor, 0)