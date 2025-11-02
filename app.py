from utils.plot import plot_model_output
from models.personal_finance import PersonalFinanceModel
from models.income_paths import (
    ARIncomePath,
    ConstantRealIncomePath,
    ExponentialGrowthIncomePath,
    LinearGrowthIncomePath,
)
from models.analysis import get_risk_aversion, marginal_change_analysis
from config.parameters import input_params
import utilities as utils
import sys

import numpy as np
import pandas as pd
import squigglepy as sq
import streamlit as st

sys.path.append("src")

# Configurations
st.set_page_config(
    page_title="Financial Life App",
)

hide_default_format = """
       <style>
       #MainMenu {visibility: hidden; }
       footer {visibility: hidden;}
       </style>
       """
st.markdown(hide_default_format, unsafe_allow_html=True)

st.title("Financial Life Model")

st.markdown(
    """
    This app simulates the possibilities for an individual's financial life using an advanced personal finance model.
    """
)

st.markdown(
    """
    The repo for this app can be found [here](https://github.com/joel-becker/financial-life).
    If you have any questions or feedback, please contact me by [email](mailto:joelhbkr@gmail.com) or [twitter](https://twitter.com/joel_bkr).
    """
)

with st.expander("How does the model work?"):
    st.markdown(
        """
        The model simulates income, inflation, and market returns paths over time.
        It uses these paths to compute paths for consumption, savings, wealth, taxes paid, and retirement benefits.
        The model now includes different income path options, detailed tax calculations for UK and California (including US federal taxes),
        and estimates of retirement benefits based on lifetime earnings and years worked.
        """
    )

st.markdown("### Parameters")

st.markdown("#### Background Information")

col1, col2 = st.columns(2)

with col1:
    current_age = st.slider("Current age", 18, 100, 30,
                            help="Your current age.")

with col2:
    age_at_death = st.slider(
        "Expected age at death",
        18,
        120,
        90,
        help="The age at which you expect to pass away.",
    )

st.markdown("#### Tax and Benefit System")

tax_region = st.selectbox(
    "Select tax region",
    ["California", "Massachusetts", "New York", "DC", "Texas", "UK"],
    help="Choose the tax system you want to use for calculations.",
)

if tax_region == "UK":
    st.info(
        "UK tax system selected. The model will calculate income tax, National Insurance contributions, and estimate State Pension. Note that Gift Aid is not currently modeled for charitable donations."
    )
elif tax_region == "California":
    st.info(
        "California tax system selected. The model will calculate federal and state income taxes, Social Security, Medicare, and estimate Social Security benefits. Charitable donations will be considered as tax deductions."
    )
elif tax_region == "Massachusetts":
    st.info(
        "Massachusetts tax system selected. The model will calculate federal and state income taxes (flat rate), Social Security, Medicare, and estimate Social Security benefits. Charitable donations will be considered as tax deductions."
    )
elif tax_region == "New York":
    st.info(
        "New York tax system selected. The model will calculate federal and state income taxes, Social Security, Medicare, and estimate Social Security benefits. Charitable donations will be considered as tax deductions."
    )
elif tax_region == "DC":
    st.info(
        "Washington D.C. tax system selected. The model will calculate federal and D.C. income taxes, Social Security, Medicare, and estimate Social Security benefits. Charitable donations will be considered as tax deductions."
    )
else:  # Texas
    st.info(
        "Texas tax system selected. The model will calculate federal income taxes (no state income tax), Social Security, Medicare, and estimate Social Security benefits. Charitable donations will be considered as tax deductions."
    )

st.markdown("#### Income and Spending")

with st.expander("Income Options"):
    col1, col2 = st.columns(2)
    with col1:
        base_income = st.number_input(
            "Base income", min_value=0, value=50_000, step=1000, help="Your starting annual income."
        )
    with col2:
        min_income = st.slider(
            "Minimum income",
            0,
            50_000,
            15_000,
            help="The reservation income, a lower bound that you don't expect to dip below.",
        )

# Default to Linear Growth with 3% growth rate
default_income_path = LinearGrowthIncomePath(base_income, 0.03, 5000)

with st.expander("Advanced Income Path Options"):
    income_path_type = st.selectbox(
        "Select income path type",
        ["Linear Growth", "Autoregressive", "Constant Real", "Exponential Growth"],
        help="Choose the type of income growth model you want to use.",
    )

    if income_path_type == "Autoregressive":
        ar_coefficients = st.text_input(
            "AR coefficients (comma-separated)",
            "0.5,0.3",
            help="Autoregressive coefficients for income model.",
        )
        ar_coefficients = [float(x) for x in ar_coefficients.split(",")]
        income_sd = st.slider(
            "Income standard deviation",
            0,
            20000,
            5000,
            help="Standard deviation of income shocks.",
        )
        income_path = ARIncomePath(
            sq.to(base_income * 0.9, base_income *
                  1.1), ar_coefficients, income_sd
        )
    elif income_path_type == "Linear Growth":
        col1, col2 = st.columns(2)
        with col1:
            annual_growth_rate = st.slider(
                "Annual growth rate",
                0.0,
                0.1,
                0.03,
                help="Annual linear growth rate of your income.",
            )
        with col2:
            income_sd = st.slider(
                "Income standard deviation",
                0,
                50000,
                5000,
                help="Standard deviation of income shocks.",
            )
        income_path = LinearGrowthIncomePath(
            base_income, annual_growth_rate, income_sd)
    else:  # Exponential Growth
        col1, col2 = st.columns(2)
        with col1:
            annual_growth_rate = st.slider(
                "Annual growth rate",
                0.0,
                0.1,
                0.03,
                help="Annual exponential growth rate of your income.",
            )
        with col2:
            income_sd = st.slider(
                "Income standard deviation",
                0,
                50000,
                5000,
                help="Standard deviation of income shocks.",
            )
        income_path = ExponentialGrowthIncomePath(
            base_income, annual_growth_rate, income_sd
        )

with st.expander("Spending Options"):
    st.markdown("##### Consumption Patterns")

    col1, col2 = st.columns(2)
    with col1:
        income_fraction_consumed_before_retirement = st.slider(
            "Fraction of annual post-tax income consumed (before retirement)",
            0.0,
            2.0,
            0.0,
            help="This is only one component of consumption: the other is the fraction of your annualized total wealth.",
        )
        wealth_fraction_consumed_before_retirement = st.slider(
            "Fraction of annualized total wealth consumed (before retirement)",
            0.0,
            2.0,
            1.15,
            help="Annualized total wealth refers to your total wealth (including all future income, excluding income this year) divided by the number of years you have left to live.",
        )

    with col2:
        income_fraction_consumed_after_retirement = st.slider(
            "Fraction of annual post-tax income consumed (after retirement)",
            0.0,
            2.0,
            0.0,
            help="This is only one component of consumption: the other is the fraction of your annualized total wealth.",
        )
        wealth_fraction_consumed_after_retirement = st.slider(
            "Fraction of annualized total wealth consumed (after retirement)",
            0.0,
            2.0,
            1.05,
            help="Beyond this, it is assumed that you will spend all of your retirement income.",
        )

    st.markdown("##### Consumption Constraints")

    col1, col2 = st.columns(2)
    with col1:
        minimum_consumption = st.slider(
            "Minimum annual consumption",
            0,
            50_000,
            20_000,
            help="Minimum amount you need to consume each year.",
        )

    with col2:
        maximum_consumption_fraction = st.slider(
            "Maximum consumption as fraction of wealth",
            1.0,
            3.0,
            2.0,
            help="Maximum consumption as a fraction of your annualized wealth.",
        )

# Use the selected income path or the default
income_path = income_path if "income_path" in locals() else default_income_path

st.markdown("#### Investment and Savings")

with st.expander("Initial Assets"):
    col1, col2 = st.columns(2)
    with col1:
        cash_start = st.number_input(
            "Initial cash savings",
            min_value=0,
            value=10_000,
            step=1000,
            help="Amount of cash savings you start with.",
        )

    with col2:
        market_start = st.number_input(
            "Initial market wealth",
            min_value=0,
            value=50_000,
            step=5000,
            help="Amount of market investments you start with.",
        )

with st.expander("Portfolio Construction"):
    portfolio_type = st.selectbox(
        "Select portfolio type",
        ["Simple", "Custom"],
        help="Choose between a simple 3-asset portfolio or a custom multi-asset portfolio.",
    )

    if portfolio_type == "Simple":
        col1, col2, col3 = st.columns(3)
        with col1:
            stock_weight = st.slider(
                "Stock allocation",
                0.0,
                1.0,
                0.7,
                help="Proportion of your portfolio allocated to stocks.",
            )
        with col2:
            bond_weight = st.slider(
                "Bond allocation",
                0.0,
                1.0 - stock_weight,
                0.15,
                help="Proportion of your portfolio allocated to bonds.",
            )
        with col3:
            real_estate_weight = st.slider(
                "Real estate allocation",
                0.0,
                1.0 - stock_weight - bond_weight,
                0.15,
                help="Proportion of your portfolio allocated to real estate.",
            )

        portfolio_weights = np.array(
            [stock_weight, bond_weight, real_estate_weight])
        asset_returns = np.array([0.07, 0.03, 0.05])
        asset_volatilities = np.array([0.15, 0.05, 0.10])
        asset_correlations = np.array(
            [[1.0, 0.2, 0.5], [0.2, 1.0, 0.3], [0.5, 0.3, 1.0]]
        )
    else:  # Custom portfolio
        num_assets = st.number_input(
            "Number of assets", min_value=1, max_value=10, value=3
        )
        portfolio_weights = []
        asset_returns = []
        asset_volatilities = []
        asset_correlations = np.eye(num_assets)

        # Default values for equity, bonds, and property
        default_returns = [0.07, 0.03, 0.05]
        default_volatilities = [0.15, 0.05, 0.10]

        # Asset features
        for i in range(num_assets):
            st.markdown(f"#### Asset {i + 1}")
            col1, col2, col3 = st.columns(3)
            with col1:
                weight = st.slider(
                    f"Asset {i + 1} weight",
                    0.0,
                    1.0,
                    1.0 / num_assets,
                    key=f"weight_{i}",
                )
                portfolio_weights.append(weight)
            with col2:
                returns = st.slider(
                    f"Asset {i + 1} return",
                    -0.1,
                    0.2,
                    default_returns[i] if i < 3 else 0.05,
                    key=f"return_{i}",
                )
                asset_returns.append(returns)
            with col3:
                volatility = st.slider(
                    f"Asset {i + 1} volatility",
                    0.0,
                    0.5,
                    default_volatilities[i] if i < 3 else 0.1,
                    key=f"volatility_{i}",
                )
                asset_volatilities.append(volatility)

        # Correlations
        st.markdown("#### Asset Correlations")
        correlation_cols = st.columns(num_assets * (num_assets - 1) // 2)
        col_index = 0
        for i in range(num_assets):
            for j in range(i + 1, num_assets):
                with correlation_cols[col_index]:
                    correlation = st.slider(
                        f"Corr {i + 1}-{j + 1}", -1.0, 1.0, 0.2, key=f"corr_{i}_{j}"
                    )
                    asset_correlations[i, j] = correlation
                    asset_correlations[j, i] = correlation
                col_index += 1

        portfolio_weights = np.array(portfolio_weights)
        asset_returns = np.array(asset_returns)
        asset_volatilities = np.array(asset_volatilities)

with st.expander("Cash Management"):
    max_cash_threshold = st.slider(
        "Maximum cash on hand",
        0,
        50_000,
        30_000,
        help="Maximum amount of cash you want to hold at any given time.",
    )
    min_cash_threshold = st.slider(
        "Minimum cash on hand",
        0,
        50_000,
        3_000,
        help="Minimum amount of cash you want to hold at any given time.",
    )

st.markdown("#### Retirement")

with st.expander("Retirement Account Settings"):
    col1, col2 = st.columns(2)
    with col1:
        retirement_age = st.slider(
            "Retirement age", 18, 100, 70, help="The age at which you retire."
        )
    with col2:
        retirement_income = st.slider(
            "Retirement income",
            0,
            100_000,
            5_000,
            help="Annual income you expect to earn during retirement.",
        )

    retirement_account_start = st.slider(
        "Initial retirement account balance",
        0,
        1_000_000,
        0,
        help="Initial balance in your retirement accounts (e.g., 401(k), IRA).",
    )
    retirement_contribution_rate = st.slider(
        "Retirement contribution rate",
        0.0,
        0.5,
        0.05,
        help="Percentage of income contributed to retirement accounts each year.",
    )

st.markdown("#### Charitable Giving")

with st.expander("Charitable Giving Options"):
    col1, col2 = st.columns(2)

    with col1:
        charitable_giving_rate = st.slider(
            "Charitable giving rate (% of income)",
            0.0,
            1.0,
            0.0,
            help="Percentage of your income that you plan to donate to charity each year.",
        )

    with col2:
        charitable_giving_cap = st.slider(
            "Maximum annual charitable donation",
            0,
            100000,
            100000,
            help="Maximum amount you're willing to donate in a single year, regardless of income.",
        )

st.markdown("#### Advanced Settings")
col1, col2 = st.columns(2)
with col1:
    with st.expander("Simulation Settings"):
        m = st.slider("Number of simulated paths", 100, 10_000, 1000)
with col2:
    with st.expander("Economic Conditions"):
        r = st.slider(
            "Risk-free interest rate",
            0.0,
            0.1,
            0.02,
            help="The risk-free rate of return.",
        )
        inflation_rate = st.slider(
            "Inflation rate", 0.0, 0.1, 0.02, help="Expected annual inflation rate."
        )

st.markdown("#### AGI Scenario")
with st.expander("AGI Impact Parameters"):
    enable_agi = st.checkbox("Enable AGI scenario modeling", value=False)

    if enable_agi:
        st.markdown("##### AGI Timing")
        st.markdown(
            "Enter your cumulative probabilities for AGI arrival by each year:")

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            prob_2_years = st.slider(
                "2 years",
                0.0,
                1.0,
                0.05,
                help="Probability of AGI arriving within 2 years",
            )
        with col2:
            prob_5_years = st.slider(
                "5 years",
                0.0,
                1.0,
                0.2,
                help="Probability of AGI arriving within 5 years",
            )
        with col3:
            prob_10_years = st.slider(
                "10 years",
                0.0,
                1.0,
                0.5,
                help="Probability of AGI arriving within 10 years",
            )
        with col4:
            prob_20_years = st.slider(
                "20 years",
                0.0,
                1.0,
                0.8,
                help="Probability of AGI arriving within 20 years",
            )

        st.markdown("##### Economic Impact")
        col1, col2 = st.columns(2)
        with col1:
            wage_multiplier = st.slider(
                "Wage multiplier after AGI",
                0.0,
                10.0,
                0.5,
                help="Factor by which wages change after AGI (e.g., 0.5 means wages are halved)",
            )
        with col2:
            returns_multiplier = st.slider(
                "Investment returns multiplier after AGI",
                0.0,
                10.0,
                2.0,
                help="Factor by which investment returns change after AGI (e.g., 2.0 means returns double)",
            )

        agi_params = {
            "cdf_points": {
                2: prob_2_years,
                5: prob_5_years,
                10: prob_10_years,
                20: prob_20_years,
            },
            "wage_multiplier": wage_multiplier,
            "returns_multiplier": returns_multiplier,
        }
    else:
        agi_params = None

# Construct the input parameters dictionary
input_params = {
    "m": m,
    "years": age_at_death - current_age + 1,
    "cash_start": cash_start,
    "market_start": market_start,
    "retirement_account_start": retirement_account_start,
    "min_income": min_income,
    "years_until_retirement": retirement_age - current_age,
    "years_until_death": age_at_death - current_age,
    "claim_age": retirement_age,
    "current_age": current_age,
    "retirement_income": retirement_income,
    "inflation_rate": inflation_rate,
    "ar_inflation_coefficients": [0.7],
    "ar_inflation_sd": 0.005,
    "r": r,
    "income_fraction_consumed_before_retirement": income_fraction_consumed_before_retirement,
    "income_fraction_consumed_after_retirement": income_fraction_consumed_after_retirement,
    "wealth_fraction_consumed_before_retirement": wealth_fraction_consumed_before_retirement,
    "wealth_fraction_consumed_after_retirement": wealth_fraction_consumed_after_retirement,
    "min_cash_threshold": min_cash_threshold,
    "max_cash_threshold": max_cash_threshold,
    "tax_region": tax_region,
    "portfolio_weights": portfolio_weights,
    "asset_returns": asset_returns,
    "asset_volatilities": asset_volatilities,
    "asset_correlations": asset_correlations,
    "income_path": income_path,
    "retirement_contribution_rate": retirement_contribution_rate,
    "minimum_consumption": minimum_consumption,
    "maximum_consumption_fraction": maximum_consumption_fraction,
    "charitable_giving_rate": charitable_giving_rate,
    "charitable_giving_cap": charitable_giving_cap,
}

if agi_params:
    input_params["agi_params"] = agi_params

st.markdown("### Results")

variables_to_plot = st.multiselect(
    "Select variables to plot",
    options=[
        "consumption",
        "income",
        "pension_income",
        "savings",
        "cash",
        "tax_paid",
        "market",
        "retirement_account",
        "retirement_contributions",
        "retirement_withdrawals",
        "financial_wealth",
        "non_financial_wealth",
        "total_wealth",
        "charitable_donations",
        "capital_gains",
        "inflation",
    ],
    default=["consumption", "income", "financial_wealth"],
)

# Run the financial life model with the input parameters
model = PersonalFinanceModel(input_params)
model.simulate()
results = model.get_results()

# Display a plot
st.pyplot(plot_model_output(results, variables_to_plot))

with st.expander("How do I read these plots?"):
    st.markdown(
        """
        The plots show the simulated paths of selected variables over time. 
        Here's what each variable represents:
        - Income: Your pre-tax income each year (excluding pension income).
        - Pension Income: The retirement benefits you receive (e.g., State Pension or Social Security).
        - Inflation: The annual inflation rate.
        - Cash: Your cash savings each year.
        - Market: Your market investments each year.
        - Retirement Account: Your retirement account balance each year.
        - Financial Wealth: Your total financial wealth, including cash, market investments, and retirement accounts.
        - Consumption: The amount you consume each year.
        - Savings: The amount you save (or dissave) each year.
        - Non-Financial Wealth: The present value of your future income and pension benefits.
        - Total Wealth: Your total wealth, including financial wealth and non-financial wealth.
        - Tax Paid: The amount of income tax and other contributions you pay each year.
        - Capital Gains: The capital gains realized each year.
        - Retirement Contributions: The amount contributed to your retirement accounts each year.
        - Retirement Withdrawals: The amount withdrawn from your retirement accounts each year.
        - Charitable Donations: The amount you donate to charity each year.

        The solid line represents the median outcome, while the shaded areas represent different confidence intervals.
        """
    )

st.markdown("### Financial Advice")

st.info("""
The advice below is based on comparing your current financial plan to some nearby alternative financial plans, based on the risk aversion level you select.
""")

risk_aversion_options = {
    "Low": "Linear utility (risk-neutral)",
    "Medium": "Log utility",
    "High": "CRRA utility with parameter 3",
}
risk_aversion_choice = st.radio(
    "Select your risk aversion level:",
    options=list(risk_aversion_options.keys()),
    index=1,  # Default to "Medium"
    help="Low corresponds to risk-neutrality/linear utility, medium to log utility, and high to CRRA utility with parameter 3.",
)
risk_aversion = get_risk_aversion(risk_aversion_choice)

if st.button("Generate Financial Advice"):
    with st.spinner("Analyzing your financial scenario..."):
        changes = marginal_change_analysis(input_params, m, risk_aversion)

        st.success("Analysis complete!")
        if changes:
            st.write(
                "Here are some suggestions that could impact your financial outcomes:"
            )

            # Create a table for the changes
            table_data = []
            for change in changes[:5]:  # Show top 5 changes
                if change["parameter"] == "portfolio_weights":
                    change_0 = float(change["change"][0])
                    change_1 = float(change["change"][1])
                    change_2 = float(change["change"][2])
                    advice = (
                        f"Adjust your investment portfolio:\n"
                        f"stocks: {change_0:.1%},\n"
                        f"bonds: {change_1:.1%},\n"
                        f"real estate: {change_2:.1%}"
                    )
                elif change["parameter"] == "years_until_retirement":
                    change_value = float(change["change"])
                    if change_value > 0:
                        advice = f"Increase your retirement age by {abs(change_value):.0f} years."
                    else:
                        advice = f"Decrease your retirement age by {abs(change_value):.0f} years."
                else:
                    change_value = float(change["change"])
                    param_name = change["parameter"].replace("_", " ").title()
                    direction = "Increase" if change_value > 0 else "Decrease"
                    advice = f"{direction} your {param_name.lower()} by {abs(change_value):.1%}."

                impact = f"{change['percent_improvement']:.2f}%"
                table_data.append([advice, impact])

            # Display the table without index
            df = pd.DataFrame(
                table_data, columns=["Suggested Change",
                                     "Impact on Lifetime Utility"]
            )
            st.dataframe(
                df.style.applymap(
                    lambda x: "color: green"
                    if x.endswith("%") and float(x[:-1]) > 0
                    else "color: red",
                    subset=["Impact on Lifetime Utility"],
                ),
                hide_index=True,
            )
        else:
            st.write(
                "Your current financial plan looks optimal based on our analysis.")
