import numpy as np
import squigglepy as sq

from models.agi_impact import AGIModel
from models.income_paths import (
    ARIncomePath,
    ConstantRealIncomePath,
    ExponentialGrowthIncomePath,
    LinearGrowthIncomePath,
)
from models.retirement_accounts import RetirementAccounts
from models.tax_system import TaxSystem
from utils.helpers import sample_or_broadcast


class PersonalFinanceModel:
    def __init__(self, input_params):
        # Initialize parameters
        self.m = input_params["m"]
        self.years = input_params["years"]
        self.r = input_params["r"]
        self.years_until_retirement = input_params["years_until_retirement"]
        self.years_until_death = input_params["years_until_death"]
        self.claim_age = input_params.get("claim_age", 67)
        self.current_age = input_params.get("current_age", 30)
        self.retirement_income = sample_or_broadcast(
            input_params["retirement_income"], self.m
        )

        self.income_path = input_params["income_path"]
        self.min_income = input_params["min_income"]
        self.inflation_rate = sample_or_broadcast(
            input_params["inflation_rate"], self.m
        )
        self.ar_inflation_coefficients = input_params["ar_inflation_coefficients"]
        self.ar_inflation_sd = input_params["ar_inflation_sd"]

        self.income_fraction_consumed_before_retirement = input_params.get(
            "income_fraction_consumed_before_retirement", 0.7
        )
        self.income_fraction_consumed_after_retirement = input_params.get(
            "income_fraction_consumed_after_retirement", 0.9
        )
        self.wealth_fraction_consumed_before_retirement = input_params.get(
            "wealth_fraction_consumed_before_retirement", 0.05
        )
        self.wealth_fraction_consumed_after_retirement = sample_or_broadcast(
            input_params.get(
                "wealth_fraction_consumed_after_retirement", sq.to(0.04, 0.06)
            ),
            self.m,
        )

        self.min_cash_threshold = input_params["min_cash_threshold"]
        self.max_cash_threshold = input_params["max_cash_threshold"]

        self.cash_start = input_params["cash_start"]
        self.market_start = input_params["market_start"]
        self.retirement_account_start = input_params.get("retirement_account_start", 0)

        self.tax_region = input_params["tax_region"]
        self.portfolio_weights = input_params["portfolio_weights"]
        self.asset_returns = input_params["asset_returns"]
        self.asset_volatilities = input_params["asset_volatilities"]
        self.asset_correlations = input_params["asset_correlations"]

        self.tax_system = TaxSystem(input_params["tax_region"])
        self.retirement_accounts = RetirementAccounts(input_params["tax_region"])

        self.retirement_contribution_rate = input_params.get(
            "retirement_contribution_rate", 0.1
        )

        self.minimum_consumption = input_params.get("minimum_consumption", 1000)
        self.maximum_consumption_fraction = input_params.get(
            "maximum_consumption_fraction", 2
        )

        # Initialize AGI model if parameters are provided
        if "agi_params" in input_params:
            agi_params = input_params["agi_params"]
            self.agi_model = AGIModel(
                agi_params["cdf_points"],
                agi_params["wage_multiplier"],
                agi_params["returns_multiplier"],
                self.m,
                self.years,
            )
        else:
            self.agi_model = None

        self.initialize_arrays()

        self.charitable_giving_rate = input_params.get("charitable_giving_rate", 0.0)
        self.charitable_giving_cap = input_params.get(
            "charitable_giving_cap", float("inf")
        )

        self.real_after_tax_income = np.zeros((self.m, self.years))
        self.real_pre_tax_income = np.zeros((self.m, self.years))
        self.real_taxable_income = np.zeros((self.m, self.years))

    def initialize_arrays(self):
        self.income = np.zeros((self.m, self.years))
        self.inflation = np.zeros((self.m, self.years))
        self.cash = np.zeros((self.m, self.years))
        self.market = np.zeros((self.m, self.years))
        self.retirement_account = np.zeros((self.m, self.years))
        self.financial_wealth = np.zeros((self.m, self.years))
        self.consumption = np.zeros((self.m, self.years))
        self.savings = np.zeros((self.m, self.years))
        self.non_financial_wealth = np.zeros((self.m, self.years))
        self.total_wealth = np.zeros((self.m, self.years))
        self.tax_paid = np.zeros((self.m, self.years))
        self.capital_gains = np.zeros((self.m, self.years))
        self.retirement_contributions = np.zeros((self.m, self.years))
        self.retirement_withdrawals = np.zeros((self.m, self.years))
        self.pension_income = np.zeros((self.m, self.years))
        self.charitable_donations = np.zeros((self.m, self.years))
        self.real_after_tax_income = np.zeros((self.m, self.years))
        self.real_pre_tax_income = np.zeros((self.m, self.years))
        self.real_taxable_income = np.zeros((self.m, self.years))
        self.cash_savings = np.zeros((self.m, self.years))
        self.market_savings = np.zeros((self.m, self.years))

    def generate_market_returns(self):
        num_assets = len(self.portfolio_weights)
        cov_matrix = (
            np.outer(self.asset_volatilities, self.asset_volatilities)
            * self.asset_correlations
        )
        nominal_returns = np.random.multivariate_normal(
            self.asset_returns, cov_matrix, (self.m, self.years)
        )

        # Apply AGI effects to returns if model exists
        if self.agi_model:
            returns_multipliers = self.agi_model.get_returns_multipliers()
            nominal_returns = nominal_returns * returns_multipliers[:, :, np.newaxis]

        return np.sum(nominal_returns * self.portfolio_weights, axis=2)

    def generate_ar_inflation(self):
        return ARIncomePath(
            self.inflation_rate, self.ar_inflation_coefficients, self.ar_inflation_sd
        ).generate(self.years, self.m)

    def generate_income(self):
        income = self.income_path.generate(self.years, self.m)

        # Apply AGI effects to income if model exists
        if self.agi_model:
            wage_multipliers = self.agi_model.get_wage_multipliers()
            income = income * wage_multipliers

        # Set income to zero after retirement
        retirement_mask = np.arange(self.years) >= self.years_until_retirement
        income[:, retirement_mask] = 0
        return income

    def simulate(self):
        self.market_returns = self.generate_market_returns()
        self.inflation = self.generate_ar_inflation()
        self.income = self.generate_income()

        cumulative_inflation = np.cumprod(1 + self.inflation, axis=1)
        real_market_returns = (1 + self.market_returns) / (1 + self.inflation) - 1

        self.income = self.income / cumulative_inflation
        self.apply_income_floor()

        self.initialize_simulation()

        for t in range(self.years):
            self.simulate_year(t, real_market_returns)

        self.calculate_non_financial_wealth()

    def initialize_simulation(self):
        self.cash[:, 0] = self.cash_start
        self.market[:, 0] = self.market_start
        self.retirement_account[:, 0] = self.retirement_account_start
        self.financial_wealth[:, 0] = (
            self.cash_start + self.market_start + self.retirement_account_start
        )
        # total_wealth[:, 0] and consumption[:, 0] are computed by
        # simulate_year(0); nothing reads them before that

    def simulate_year(self, t, real_market_returns):
        current_age = self.current_age + t
        is_retired = t >= self.years_until_retirement
        years_left = self.years_until_death - t

        # Calculate total wealth including future pension benefits
        self.calculate_total_wealth(t, current_age)

        # Calculate income and pension
        total_real_income = self.calculate_total_income(t, current_age)
        self.real_pre_tax_income[:, t] = total_real_income

        # Handle retirement contributions or withdrawals
        if not is_retired:
            self.calculate_retirement_contribution(t, total_real_income, current_age)
        else:
            # Withdrawals target last year's consumption; at t=0 there is
            # none yet (and consumption[:, -1] would read the last column)
            prev_consumption = (
                self.consumption[:, t - 1] if t > 0 else np.zeros(self.m)
            )
            self.calculate_retirement_withdrawal(
                t, current_age, total_real_income, prev_consumption
            )

        # Calculate charitable donations
        self.charitable_donations[:, t] = self.calculate_charitable_donations(
            t, total_real_income
        )

        # Mark-to-market capital gains for this year, computed before the
        # tax calculation so they are taxed in the year they occur
        self.capital_gains[:, t] = self.market[:, t] * real_market_returns[:, t]

        # Calculate real after-tax income
        real_after_tax_income = self.calculate_after_tax_income(t, total_real_income)
        self.real_after_tax_income[:, t] = real_after_tax_income

        # Calculate consumption based on real after-tax income and total wealth
        self.consumption[:, t] = self.calculate_consumption_amount(
            t, real_after_tax_income, is_retired, years_left
        )

        # Calculate savings
        self.savings[:, t] = real_after_tax_income - self.consumption[:, t]

        # Update wealth
        self.update_wealth(t, real_market_returns, is_retired)

    def calculate_total_wealth(self, t, current_age):
        financial_wealth = self.cash[:, t] + self.market[:, t]
        retirement_wealth = self.retirement_account[:, t]
        future_income = self.calculate_future_income(t)
        future_pension = self.calculate_future_pension(t, current_age)

        # Estimate tax on retirement account
        retirement_tax = self.estimate_retirement_account_tax(
            retirement_wealth, current_age
        )

        # Estimate tax on future income and pension
        future_income_tax = self.estimate_future_income_tax(
            future_income, future_pension, current_age
        )

        # No haircut for embedded capital gains: the model taxes gains
        # mark-to-market every year, so there are no deferred gains left
        self.total_wealth[:, t] = (
            financial_wealth
            + retirement_wealth
            + future_income
            + future_pension
            - retirement_tax
            - future_income_tax
        )

    def estimate_retirement_account_tax(self, retirement_wealth, current_age):
        if self.tax_region == "UK":
            # In UK, 25% of pension is tax-free
            taxable_amount = retirement_wealth * 0.75
        else:  # US
            taxable_amount = retirement_wealth

        # Estimate annual withdrawal over remaining lifespan
        years_left = max(1, self.years_until_death - (current_age - self.current_age))
        annual_withdrawal = taxable_amount / years_left

        # Withdrawals are ordinary income but not wages (no payroll tax)
        annual_tax = self.tax_system.calculate_tax(annual_withdrawal, 0, wage_income=0)

        return annual_tax * years_left

    def estimate_future_contributions(self, t, current_age):
        years_until_retirement = max(
            0, self.years_until_retirement - (current_age - self.current_age)
        )
        future_income = np.sum(self.income[:, t : t + years_until_retirement], axis=1)
        return future_income * self.retirement_contribution_rate

    def estimate_future_income_tax(self, future_income, future_pension, current_age):
        years_left = max(1, self.years_until_death - (current_age - self.current_age))
        annual_income = (future_income + future_pension) / years_left

        # Estimate future contributions
        years_until_retirement = max(
            0, self.years_until_retirement - (current_age - self.current_age)
        )
        annual_contribution = (
            (future_income / years_left)
            * self.retirement_contribution_rate
            * (years_until_retirement / years_left)
        )

        # Payroll tax applies to the wage portion only, not the pension
        annual_wages = future_income / years_left
        annual_tax = self.tax_system.calculate_tax(
            (annual_income - annual_contribution), 0, wage_income=annual_wages
        )

        return annual_tax * years_left

    def apply_income_floor(self):
        # min_income is a reservation wage: it floors labor income during
        # working years only, and must not resurrect wages for retirees
        working_years = min(self.years_until_retirement, self.years)
        self.income[:, :working_years] = np.maximum(
            self.income[:, :working_years], self.min_income
        )

    def calculate_total_income(self, t, current_age):
        base_income = self.income[:, t]
        self.pension_income[:, t] = self.calculate_pension_income(t, current_age)

        total_income = base_income + self.pension_income[:, t]

        # Non-wage retirement income (annuities, part-time work modeled as
        # a lump sum) — the app's "Retirement income" input, previously a
        # silent no-op
        if t >= self.years_until_retirement:
            total_income = total_income + self.retirement_income

        return total_income

    def calculate_retirement_withdrawal(
        self, t, current_age, total_real_income, consumption
    ):
        required_withdrawal = np.maximum(consumption - total_real_income, 0)

        # Calculate proportions of pension and non-pension wealth
        pension_wealth = self.calculate_future_pension(t, current_age)
        non_pension_wealth = np.maximum(
            self.total_wealth[:, t] - pension_wealth, 0
        )  # Ensure non-negative
        total_wealth_safe = np.maximum(
            self.total_wealth[:, t], 1e-10
        )  # Avoid division by zero
        pension_proportion = np.clip(pension_wealth / total_wealth_safe, 0, 1)
        non_pension_proportion = np.clip(non_pension_wealth / total_wealth_safe, 0, 1)

        # Calculate withdrawals from each source
        pension_withdrawal = required_withdrawal * pension_proportion
        non_pension_withdrawal = required_withdrawal * non_pension_proportion

        # Ensure we don't withdraw more than available from retirement account
        max_retirement_withdrawal = self.retirement_account[:, t]
        actual_retirement_withdrawal = np.minimum(
            non_pension_withdrawal, max_retirement_withdrawal
        )

        self.retirement_withdrawals[:, t] = actual_retirement_withdrawal

    def calculate_pension_income(self, t, current_age):
        if self.tax_region == "UK":
            return self.calculate_uk_pension(t, current_age)
        elif self.tax_region in [
            "California",
            "Massachusetts",
            "New York",
            "DC",
            "Texas",
        ]:
            return self.calculate_us_social_security(t, current_age)
        else:
            return np.zeros(self.m)

    def calculate_uk_pension(self, t, current_age):
        pension_amount = np.where(
            current_age >= self.claim_age,
            (
                np.minimum(
                    self.claim_age - self.current_age,
                    self.tax_system.uk_qualifying_years,
                )
                / self.tax_system.uk_qualifying_years
            )
            * self.tax_system.uk_full_pension,
            0,
        )
        return pension_amount

    def calculate_us_social_security(self, t, current_age):
        if np.all(current_age < self.claim_age):
            return np.zeros(self.m)

        # Calculate AIME (Average Indexed Monthly Earnings): the top 35
        # earning years divided by 35 years of months — shorter careers
        # average in zeros, per SSA rules
        max_taxable_earnings = self.tax_system.max_taxable_earnings
        indexed_earnings = np.minimum(
            self.income[:, : self.years_until_retirement], max_taxable_earnings
        )
        top_35 = np.sort(indexed_earnings, axis=1)[:, -35:]
        aime = np.sum(top_35, axis=1) / (35 * 12)

        # Calculate PIA (Primary Insurance Amount): 90% of AIME up to the
        # first bend point, 32% between the bend points, 15% above
        bend_1, bend_2 = self.tax_system.bend_points
        factor_1, factor_2, factor_3 = self.tax_system.pia_factors
        pia = (
            factor_1 * np.minimum(aime, bend_1)
            + factor_2 * np.clip(aime - bend_1, 0, bend_2 - bend_1)
            + factor_3 * np.maximum(aime - bend_2, 0)
        )

        # Adjust for claiming age: ~5/9% per month reduction for the first
        # 36 months before FRA, ~5/12% per month beyond that; ~2/3% per
        # month delayed-retirement credit after FRA
        if self.claim_age < self.tax_system.fra:
            months_early = (self.tax_system.fra - self.claim_age) * 12
            age_adjustment = (
                1
                - 0.00555556 * np.minimum(36, months_early)
                - 0.00416667 * np.maximum(0, months_early - 36)
            )
        else:
            months_delayed = (self.claim_age - self.tax_system.fra) * 12
            age_adjustment = 1 + 0.00666667 * months_delayed

        pia *= age_adjustment

        # Apply the maximum monthly benefit for the claiming age (falling
        # back to the FRA cap for ages without a published maximum)
        benefit_caps = self.tax_system.ss_max_monthly_benefit
        max_benefit = benefit_caps.get(int(self.claim_age), benefit_caps[67])

        pia = np.minimum(pia, max_benefit)

        # Return the annual benefit in real terms
        return np.where(current_age >= self.claim_age, pia * 12, 0)

    def calculate_consumption_amount(
        self, t, total_real_income, is_retired, years_left
    ):
        wealth_consumption_rate = np.where(
            is_retired,
            self.wealth_fraction_consumed_after_retirement,
            self.wealth_fraction_consumed_before_retirement,
        )
        income_consumption_rate = np.where(
            is_retired,
            self.income_fraction_consumed_after_retirement,
            self.income_fraction_consumed_before_retirement,
        )

        # Calculate annualized wealth
        annualized_wealth = np.where(
            years_left > 0,
            self.total_wealth[:, t] / years_left,
            self.total_wealth[:, t],
        )

        # Calculate consumption from annualized wealth and income
        consumption_from_wealth = wealth_consumption_rate * annualized_wealth
        consumption_from_income = income_consumption_rate * total_real_income

        base_consumption = consumption_from_income + consumption_from_wealth

        # The model runs in real terms, so the floor applies at full value
        # every year — no further deflation
        min_consumption = self.minimum_consumption

        # Calculate currently available resources
        current_financial_wealth = (
            self.cash[:, t] + self.market[:, t] + self.retirement_account[:, t]
        )
        currently_available_resources = total_real_income + current_financial_wealth

        # Ensure max_consumption doesn't exceed currently available resources
        max_consumption = np.minimum(
            self.maximum_consumption_fraction * currently_available_resources,
            currently_available_resources,
        )

        return np.clip(base_consumption, min_consumption, max_consumption)

    def update_wealth(self, t, real_market_returns, is_retired, verbose=False):
        # Calculate total savings
        total_savings = self.savings[:, t]

        # Update retirement account
        self.retirement_account[:, t] *= 1 + real_market_returns[:, t]
        if not is_retired:
            self.retirement_account[:, t] += self.retirement_contributions[:, t]
        else:
            self.retirement_account[:, t] -= self.retirement_withdrawals[:, t]

        # Update market value (capital gains were already recorded in
        # simulate_year, before the tax calculation)
        self.market[:, t] *= 1 + real_market_returns[:, t]

        # Determine cash savings (up to max_cash_threshold)
        cash_savings = np.minimum(
            total_savings, self.max_cash_threshold - self.cash[:, t]
        )
        self.cash_savings[:, t] = cash_savings

        # Remaining savings go to market
        market_savings = total_savings - cash_savings
        self.market_savings[:, t] = market_savings

        # Update cash and market
        self.cash[:, t] += cash_savings
        self.market[:, t] += market_savings

        # Adjust cash and market
        self.adjust_cash_and_market(t)

        # Ensure no negative values
        self.cash[:, t] = np.maximum(self.cash[:, t], 0)
        self.market[:, t] = np.maximum(self.market[:, t], 0)
        self.retirement_account[:, t] = np.maximum(self.retirement_account[:, t], 0)

        # Update financial wealth
        self.financial_wealth[:, t] = (
            self.cash[:, t] + self.market[:, t] + self.retirement_account[:, t]
        )

        if t < self.years - 1:
            self.cash[:, t + 1] = self.cash[:, t]
            self.market[:, t + 1] = self.market[:, t]
            self.retirement_account[:, t + 1] = self.retirement_account[:, t]
            self.financial_wealth[:, t + 1] = self.financial_wealth[:, t]

        if verbose:
            print(f"\nEntering update_wealth for t={t}")
            print(
                f"Initial state: cash={self.cash[0, t]}, market={self.market[0, t]}, retirement={self.retirement_account[0, t]}"
            )
            print(f"Total savings: {total_savings}")
            print(
                f"After retirement update: retirement={self.retirement_account[0, t]}"
            )
            print(f"Capital gains: {self.capital_gains[0, t]}")
            print(f"After market update: market={self.market[0, t]}")
            print(f"Cash savings: {cash_savings}")
            print(f"Market savings: {market_savings}")
            print(
                f"After savings update: cash={self.cash[0, t]}, market={self.market[0, t]}"
            )
            print(
                f"After adjustment: cash={self.cash[0, t]}, market={self.market[0, t]}"
            )
            print(
                f"Final state: cash={self.cash[0, t]}, market={self.market[0, t]}, retirement={self.retirement_account[0, t]}, wealth={self.financial_wealth[0, t]}"
            )
            if t < self.years - 1:
                print(
                    f"Updating t+1: cash={self.cash[0, t + 1]}, market={self.market[0, t + 1]}, retirement={self.retirement_account[0, t + 1]}"
                )
                print(
                    f"After update: cash={self.cash[0, t + 1]}, market={self.market[0, t + 1]}, retirement={self.retirement_account[0, t + 1]}, wealth={self.financial_wealth[0, t + 1]}"
                )

    def calculate_retirement_contribution(self, t, total_real_income, current_age):
        contribution = np.minimum(
            self.retirement_accounts.calculate_contribution(
                total_real_income, current_age, self.retirement_contribution_rate
            ),
            total_real_income * 0.9,
        )
        self.retirement_contributions[:, t] = contribution
        return contribution

    def calculate_after_tax_income(self, t, total_real_income):
        real_capital_gains = self.capital_gains[:, t]
        real_donations = self.charitable_donations[:, t]
        real_contributions = self.retirement_contributions[:, t]
        real_withdrawals = self.retirement_withdrawals[:, t]

        # Pre-tax contributions reduce taxable income; withdrawals from
        # pre-tax retirement accounts are taxed as income. Donations are
        # NOT subtracted here — TaxSystem applies the deduction (with the
        # AGI cap) itself.
        real_taxable_income = (
            total_real_income - real_contributions + real_withdrawals
        )
        # At most 85% of US Social Security benefits are taxable; the
        # provisional-income phase-in is not modeled, so this applies the
        # upper bound. UK State Pension is fully taxable.
        if self.tax_region != "UK":
            real_taxable_income = (
                real_taxable_income - 0.15 * self.pension_income[:, t]
            )
        self.real_taxable_income[:, t] = real_taxable_income

        # Payroll taxes apply to labor income only, not pension income or
        # retirement withdrawals
        self.tax_paid[:, t] = self.tax_system.calculate_tax(
            real_taxable_income,
            real_capital_gains,
            real_donations,
            wage_income=self.income[:, t],
        )

        # Spendable income: withdrawn dollars are available to spend;
        # taxed, contributed, and donated dollars are not.
        return (
            total_real_income
            + real_withdrawals
            - self.tax_paid[:, t]
            - real_contributions
            - real_donations
        )

    def adjust_cash_and_market(self, t):
        total_liquid = self.cash[:, t] + self.market[:, t]

        # Ensure minimum cash balance
        self.cash[:, t] = np.maximum(self.cash[:, t], self.min_cash_threshold)

        # If cash exceeds maximum, move excess to market
        excess_cash = np.maximum(self.cash[:, t] - self.max_cash_threshold, 0)
        self.cash[:, t] -= excess_cash
        self.market[:, t] += excess_cash

        # Ensure cash doesn't exceed total liquid assets
        self.cash[:, t] = np.minimum(self.cash[:, t], total_liquid)
        self.market[:, t] = total_liquid - self.cash[:, t]

    def calculate_future_income(self, t):
        return np.sum(self.income[:, t:], axis=1)

    def calculate_future_pension(self, t, current_age):
        years_until_claim = max(0, self.claim_age - current_age)
        future_pension = np.sum(self.pension_income[:, t + years_until_claim :], axis=1)
        return future_pension

    def calculate_non_financial_wealth(self):
        for t in range(self.years):
            current_age = self.current_age + t
            self.non_financial_wealth[:, t] = self.calculate_future_income(
                t
            ) + self.calculate_future_pension(t, current_age)

    def calculate_charitable_donations(self, t, total_real_income):
        donations = total_real_income * self.charitable_giving_rate
        return np.minimum(donations, self.charitable_giving_cap)

    def get_results(self):
        return {
            "income": self.income,
            "pension_income": self.pension_income,
            "inflation": self.inflation,
            "cash": self.cash,
            "market": self.market,
            "retirement_account": self.retirement_account,
            "financial_wealth": self.financial_wealth,
            "consumption": self.consumption,
            "savings": self.savings,
            "non_financial_wealth": self.non_financial_wealth,
            "total_wealth": self.total_wealth,
            "tax_paid": self.tax_paid,
            "capital_gains": self.capital_gains,
            "retirement_contributions": self.retirement_contributions,
            "retirement_withdrawals": self.retirement_withdrawals,
            "charitable_donations": self.charitable_donations,
            "real_after_tax_income": self.real_after_tax_income,
            "real_pre_tax_income": self.real_pre_tax_income,
        }
