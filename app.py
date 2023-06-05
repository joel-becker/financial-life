import streamlit as st

import sys

sys.path.append("src")
import utilities as utils
import parameters as params

# Configurations
st.set_page_config(page_title="Financial Life App")

st.set_page_config(layout="wide")

st.markdown(
    """
<style>
body {
    color: #fff;
    background-color: #4F8BF9;
}
</style>
    """, 
    unsafe_allow_html=True
    )

st.markdown("""
<style>
    h1 {
        color: red;
    }
</style>
""", unsafe_allow_html=True)

st.title("Financial Life Model")

# Create columns for parameter inputs
left, center, right = st.columns(3)

# Sliders for the model's parameters
with left:
    m = st.slider("Number of paths", 100, 500, 300)
    years = st.slider("Number of years", 10, 50, 25)
    cash_start = st.slider("Initial cash", 0, 100_000, 10_000)
    market_start = st.slider("Initial market wealth", 0, 500_000, 30_000)

with center:
    # Income lifecycle
    st.markdown("### Income lifecycle")
    income_start = st.slider("Initial income", 20_000, 200_000, 40_000)
    st.markdown("### Income AR(1) coefficients")
    ar_income_coefficients = st.slider(
        "AR(1) coefficients", 0.0, 1.0, (0.5, 0.3, 0.1)
    )
    ar_income_sd = st.slider("AR(1) standard deviation", 0.0, 100_000.0, 15_000.0)

with right:
    min_income = st.slider(
        "Minimum income",
        0,
        50_000,
        10_000,
        help="The reservation income, a lower bound that you don't expect to dip below.",
    )
    inflation_rate = st.slider("Inflation rate", 0.0, 0.1, 0.02)
    r = st.slider(
        "Interest rate", 0.0, 0.1, 0.02, help="The rate of return on savings."
    )

variables_to_plot = st.multiselect(
    "Select variables to plot",
    options=[
        "income",
        "cash",
        "market",
        "consumption",
        "inflation",
        "savings",
        "financial_wealth",
    ],
    default=["income", "cash", "market", "consumption"],
)

# Construct the parameters dictionary
input_params = params.input_params
input_params["m"] = m
input_params["years"] = years
input_params["cash_start"] = cash_start
input_params["market_start"] = market_start
input_params["life_cycle_income"] = [
    income_start + (year * 3 * 1000) for year in range(years)
]
input_params["min_income"] = min_income
input_params["inflation_rate"] = inflation_rate
input_params["r"] = r

# Run the financial life model with the input parameters
model_output = utils.financial_life_model(input_params)

# Display a plot
st.pyplot(utils.plot_model_output(model_output, variables_to_plot))
