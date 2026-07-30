import numpy as np
import pytest

from models.personal_finance import PersonalFinanceModel
from tests.test_tax_flow import make_params


def make_retiree_model(**overrides):
    # retired from t=5 of a 10-year sim, min_income floor present
    params = make_params(
        m=1,
        years=10,
        years_until_retirement=5,
        years_until_death=10,
        retirement_income=40000,
        min_income=30000,
        claim_age=99,  # no SS within horizon; isolates retirement_income
    )
    params.update(overrides)
    model = PersonalFinanceModel(params)
    model.inflation = np.zeros((1, 10))
    return model


def test_min_income_floor_applies_to_working_years_only():
    # the floor is a reservation wage; it must not resurrect labor income
    # for retirees
    model = make_retiree_model()
    model.income = np.zeros((1, 10))
    model.income[0, :5] = 20000.0  # below the 30k floor
    model.apply_income_floor()
    assert np.all(model.income[0, :5] == 30000.0)
    assert np.all(model.income[0, 5:] == 0.0)


def test_retirement_income_received_after_retirement():
    model = make_retiree_model()
    model.income = np.zeros((1, 10))
    total = model.calculate_total_income(6, 36)
    assert total[0] == pytest.approx(40000.0)


def test_retirement_income_not_received_while_working():
    model = make_retiree_model()
    model.income = np.zeros((1, 10))
    model.income[0, :5] = 100000.0
    total = model.calculate_total_income(2, 32)
    assert total[0] == pytest.approx(100000.0)


def test_retirement_income_pays_no_payroll_tax():
    # retirement income is ordinary income but not wages
    model = make_retiree_model()
    model.income = np.zeros((1, 10))
    model.initialize_simulation()
    model.simulate_year(6, np.zeros((1, 10)))
    expected_tax = model.tax_system.calculate_tax(
        model.real_taxable_income[:, 6],
        np.array([0.0]),
        np.array([0.0]),
        wage_income=np.array([0.0]),
    )
    assert model.tax_paid[0, 6] == pytest.approx(expected_tax[0])


def test_years_until_retirement_clamped_to_zero():
    # retirement age below current age must mean "already retired", not a
    # negative array slice that floors phantom wages into most years
    model = make_retiree_model(years_until_retirement=-10, min_income=15000)
    np.random.seed(0)
    model.simulate()
    assert np.all(model.income == 0.0)
    assert np.all(model.retirement_contributions == 0.0)


def test_future_pension_counted_in_total_wealth_during_simulation():
    # pension_income must be populated for ALL years before the simulation
    # loop; previously calculate_total_wealth always saw zeros ahead of t
    # and omitted the entire pension annuity from total_wealth
    from tests.test_tax_flow import make_params
    params = make_params(
        m=1, years=40, years_until_retirement=30, years_until_death=40,
        claim_age=67, current_age=30,
    )
    seen = []
    model = PersonalFinanceModel(params)
    original = model.calculate_future_pension

    def probe(t, current_age):
        value = original(t, current_age)
        seen.append((t, float(value[0])))
        return value

    model.calculate_future_pension = probe
    np.random.seed(1)
    model.simulate()
    early_calls = [v for t, v in seen if t == 0]
    assert early_calls and early_calls[0] > 0
