import numpy as np
import squigglepy as sq
from models.income_paths import ARIncomePath, ConstantRealIncomePath, LinearGrowthIncomePath, ExponentialGrowthIncomePath

# Default three-asset portfolio (stocks, bonds, real estate) — shared by
# these defaults and the app's sliders so they can't drift apart
DEFAULT_ASSET_RETURNS = np.array([0.07, 0.03, 0.05])
DEFAULT_ASSET_VOLATILITIES = np.array([0.15, 0.05, 0.10])
DEFAULT_ASSET_CORRELATIONS = np.array([
    [1.0, 0.2, 0.5],
    [0.2, 1.0, 0.3],
    [0.5, 0.3, 1.0]
])

input_params = {
    "m": 1000,
    "years": 40,
    "cash_start": 10000,
    "market_start": 50000,
    "min_income": 30000,
    "years_until_retirement": 35,
    "years_until_death": 40,
    "claim_age": 70,
    "retirement_income": sq.to(30000, 50000),
    "inflation_rate": sq.to(0.01, 0.03),
    "ar_inflation_coefficients": [0.7],
    "ar_inflation_sd": 0.005,
    "r": 0.05,
    "income_fraction_consumed_before_retirement": 0.1,
    "income_fraction_consumed_after_retirement": 0.1,
    "wealth_fraction_consumed_before_retirement": 0.9,
    "wealth_fraction_consumed_after_retirement": 0.9,
    "min_cash_threshold": 5000,
    "max_cash_threshold": 20000,
    "tax_region": "California",
    "portfolio_weights": np.array([0.6, 0.3, 0.1]),  # Stocks, Bonds, Real Estate
    "asset_returns": DEFAULT_ASSET_RETURNS,
    "asset_volatilities": DEFAULT_ASSET_VOLATILITIES,
    "asset_correlations": DEFAULT_ASSET_CORRELATIONS,
    "income_path": LinearGrowthIncomePath(50000, 0.03, 5000),
    "retirement_account_start": 100000,
    "retirement_contribution_rate": 0.1,
}

