import numpy as np


class TaxSystem:
    """Tax math for one region.

    All monetary thresholds are 2023 values and are treated as constant in
    REAL terms: the simulation runs on inflation-adjusted dollars, and
    since US federal brackets (and, roughly, the other thresholds) are
    indexed to inflation, holding them fixed in real terms is the intended
    approximation. When updating for a new tax year, replace the values
    below wholesale.
    """

    def __init__(self, region):
        self.region = region
        self._initialize_tax_parameters()

    def _initialize_tax_parameters(self):
        if self.region == "UK":
            self._initialize_uk_parameters()
        elif self.region in ["California", "Massachusetts", "New York", "DC", "Texas"]:
            self._initialize_us_parameters()
            if self.region == "California":
                self._initialize_california_parameters()
            elif self.region == "Massachusetts":
                self._initialize_massachusetts_parameters()
            elif self.region == "New York":
                self._initialize_new_york_parameters()
            elif self.region == "DC":
                self._initialize_dc_parameters()
            elif self.region == "Texas":
                self._initialize_texas_parameters()
        else:
            raise ValueError(f"Unsupported tax region: {self.region}")

    def _initialize_uk_parameters(self):
        self.personal_allowance = 12570.0
        self.basic_rate_threshold = 50270.0
        self.higher_rate_threshold = 150000.0
        self.additional_rate_threshold = 125140.0
        self.basic_rate = 0.20
        self.higher_rate = 0.40
        self.additional_rate = 0.45
        self.ni_primary_threshold = 9568.0
        self.ni_upper_earnings_limit = 50270.0
        self.ni_basic_rate = 0.12
        self.ni_higher_rate = 0.02
        self.capital_gains_allowance = 6000.0
        self.capital_gains_basic_rate = 0.10
        self.capital_gains_higher_rate = 0.20
        self.uk_full_pension = 10600.0
        self.uk_qualifying_years = 35.0

    def _initialize_us_parameters(self):
        # Federal tax parameters (common for all US states)
        self.federal_brackets = [
            (0.0, 0.10), (9950.0, 0.12), (40525.0, 0.22),
            (86375.0, 0.24), (164925.0, 0.32), (209425.0, 0.35),
            (523600.0, 0.37)
        ]
        self.ss_wage_base = 142800.0
        self.ss_rate = 0.062
        self.medicare_rate = 0.0145
        self.capital_gains_brackets = [
            (0.0, 0.0), (40400.0, 0.15), (445850.0, 0.20)
        ]
        self.max_taxable_earnings = 160200.0
        self.bend_points = [1115.0, 6721.0]
        self.pia_factors = [0.9, 0.32, 0.15]
        self.fra = 67.0
        # Maximum monthly SS benefit by claiming age (2023)
        self.ss_max_monthly_benefit = {62: 2572.0, 67: 3627.0, 70: 4555.0}
        self.charitable_donation_limit = 0.60

    def _initialize_california_parameters(self):
        self.ca_brackets = [
            (0.0, 0.01), (8932.0, 0.02), (21175.0, 0.04),
            (33421.0, 0.06), (46394.0, 0.08), (58634.0, 0.093),
            (299508.0, 0.103), (359407.0, 0.113), (599012.0, 0.123)
        ]

    def _initialize_massachusetts_parameters(self):
        self.ma_rate = 0.05  # Massachusetts has a flat income tax rate

    def _initialize_new_york_parameters(self):
        self.ny_brackets = [
            (0.0, 0.04), (8500.0, 0.045), (11700.0, 0.0525),
            (13900.0, 0.0590), (21400.0, 0.0609), (80650.0, 0.0641),
            (215400.0, 0.0685), (1077550.0, 0.0965), (5000000.0, 0.103),
            (25000000.0, 0.109)
        ]

    def _initialize_dc_parameters(self):
        self.dc_brackets = [
            (0.0, 0.04), (10000.0, 0.06), (40000.0, 0.065),
            (60000.0, 0.085), (350000.0, 0.0875), (1000000.0, 0.0895)
        ]

    def _initialize_texas_parameters(self):
        # Texas has no state income tax
        pass

    def calculate_tax(self, income, capital_gains=0, charitable_donations=0, wage_income=None):
        # wage_income is the payroll-tax base (FICA / National Insurance
        # apply to wages only, not pensions or retirement withdrawals);
        # defaults to income for backward-looking callers taxing pure wages
        if wage_income is None:
            wage_income = income
        if self.region == "UK":
            return self._calculate_uk_tax(income, capital_gains, wage_income)
        elif self.region in ["California", "Massachusetts", "New York", "DC", "Texas"]:
            return self._calculate_us_tax(income, capital_gains, charitable_donations, wage_income)

    def _calculate_us_tax(self, income, capital_gains, charitable_donations, wage_income):
        # Calculate Adjusted Gross Income (AGI)
        agi = income + capital_gains

        # Limit charitable donations to 60% of AGI
        deductible_donations = np.minimum(charitable_donations, agi * self.charitable_donation_limit)

        # Ordinary income is taxed at federal ordinary brackets; capital
        # gains are excluded from that base and taxed at the LTCG brackets
        # STACKED on top of ordinary income (gains fill brackets starting
        # where ordinary income ends). States tax gains as ordinary income.
        ordinary_taxable = np.maximum(income - deductible_donations, 0)

        federal_income_tax = self._calculate_bracketed_tax(ordinary_taxable, self.federal_brackets)
        capital_gains_tax = self._calculate_bracketed_tax(
            ordinary_taxable + capital_gains, self.capital_gains_brackets
        ) - self._calculate_bracketed_tax(ordinary_taxable, self.capital_gains_brackets)
        state_income_tax = self._calculate_state_tax(
            np.maximum(ordinary_taxable + capital_gains, 0)
        )
        ss_tax = np.minimum(wage_income, self.ss_wage_base) * self.ss_rate
        medicare_tax = wage_income * self.medicare_rate

        return federal_income_tax + state_income_tax + ss_tax + medicare_tax + capital_gains_tax

    def _calculate_uk_tax(self, income, capital_gains, wage_income):
        taxable_income = np.maximum(income - self.personal_allowance, 0.0)
        basic_rate_tax = np.minimum(taxable_income, self.basic_rate_threshold - self.personal_allowance) * self.basic_rate
        higher_rate_tax = np.maximum(np.minimum(taxable_income, self.higher_rate_threshold) - (self.basic_rate_threshold - self.personal_allowance), 0.0) * self.higher_rate
        additional_rate_tax = np.maximum(taxable_income - self.higher_rate_threshold, 0.0) * self.additional_rate

        total_income_tax = basic_rate_tax + higher_rate_tax + additional_rate_tax
        
        ni_contributions = np.minimum(np.maximum(wage_income - self.ni_primary_threshold, 0.0), self.ni_upper_earnings_limit - self.ni_primary_threshold) * self.ni_basic_rate + np.maximum(wage_income - self.ni_upper_earnings_limit, 0.0) * self.ni_higher_rate

        taxable_capital_gains = np.maximum(capital_gains - self.capital_gains_allowance, 0.0)
        capital_gains_tax = np.minimum(taxable_capital_gains, np.maximum(self.basic_rate_threshold - income, 0)) * self.capital_gains_basic_rate + np.maximum(taxable_capital_gains - np.maximum(self.basic_rate_threshold - income, 0), 0.0) * self.capital_gains_higher_rate

        return total_income_tax + ni_contributions + capital_gains_tax

    def _calculate_state_tax(self, taxable_income):
        if self.region == "California":
            return self._calculate_bracketed_tax(taxable_income, self.ca_brackets)
        elif self.region == "Massachusetts":
            return taxable_income * self.ma_rate
        elif self.region == "New York":
            return self._calculate_bracketed_tax(taxable_income, self.ny_brackets)
        elif self.region == "DC":
            return self._calculate_bracketed_tax(taxable_income, self.dc_brackets)
        elif self.region == "Texas":
            return np.zeros_like(taxable_income)  # No state income tax in Texas

    def _calculate_bracketed_tax(self, amount, brackets):
        tax = np.zeros_like(amount, dtype=np.float64)
        for i, (threshold, rate) in enumerate(brackets):
            if i == len(brackets) - 1:
                tax += np.maximum(amount - threshold, 0.0) * rate
            else:
                next_threshold = brackets[i+1][0]
                tax += np.minimum(np.maximum(amount - threshold, 0.0), next_threshold - threshold) * rate
        return tax

