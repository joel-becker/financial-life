import numpy as np
from models.personal_finance import PersonalFinanceModel

def calculate_crra_utility(consumption, risk_aversion):
    if risk_aversion == 1:  # Log utility
        return np.log(consumption)
    elif risk_aversion == 0:  # Linear utility
        return consumption
    else:
        return (consumption ** (1 - risk_aversion) - 1) / (1 - risk_aversion)

def calculate_utility(params, n_sims, risk_aversion, seed=0):
    # Fixed seed = common random numbers across candidate plans, so
    # utility differences reflect the parameter change, not sampling noise
    np.random.seed(seed)
    model = PersonalFinanceModel(params)
    model.simulate()
    results = model.get_results()
    consumption = results['consumption'][:n_sims]
    return np.mean(np.sum(calculate_crra_utility(consumption, risk_aversion), axis=1))

def marginal_change_analysis(base_params, n_sims, risk_aversion):
    base_utility = calculate_utility(base_params, n_sims, risk_aversion)
    changes = []

    # Define small changes for each parameter group
    param_changes = {
        'consumption': [
            ('wealth_fraction_consumed_before_retirement', [-0.1, 0.1]),
            ('wealth_fraction_consumed_after_retirement', [-0.1, 0.1])
        ],
        'portfolio': [
            ('portfolio_weights', [
                [0.1, -0.05, -0.05],  # Increase stocks
                [-0.05, 0.1, -0.05],  # Increase bonds
                [-0.05, -0.05, 0.1]   # Increase real estate
            ])
        ],
        'retirement': [
            ('years_until_retirement', [-2, 2]),
            ('retirement_contribution_rate', [-0.05, 0.05])
        ]
    }

    for group, group_changes in param_changes.items():
        for param, deltas in group_changes:
            for delta in deltas:
                new_params = base_params.copy()
                if param == 'portfolio_weights':
                    new_weights = [max(0, min(1, x + d)) for x, d in zip(new_params[param], delta)]
                    # Normalize weights to ensure they sum to 1
                    total = sum(new_weights)
                    new_params[param] = [w / total for w in new_weights]
                elif param.startswith('income_fraction') or param.startswith('wealth_fraction'):
                    new_params[param] = max(0, min(1, new_params[param] + delta))
                else:
                    new_params[param] = max(0, new_params[param] + delta)
                
                new_utility = calculate_utility(new_params, n_sims, risk_aversion)
                # abs() so the sign means better/worse even when CRRA
                # utility is negative (risk_aversion > 1)
                percent_change = (new_utility - base_utility) / abs(base_utility) * 100
                
                if param == 'portfolio_weights':
                    change_description = [f"{d:.2f}" for d in delta]
                else:
                    change_description = f"{delta:.2f}"
                
                changes.append({
                    'group': group,
                    'parameter': param,
                    'change': change_description,
                    'percent_improvement': percent_change
                })

    return sorted(changes, key=lambda x: abs(x['percent_improvement']), reverse=True)

def focused_what_if_analysis(base_params, changes, n_sims, risk_aversion):
    base_utility = calculate_utility(base_params, n_sims, risk_aversion)
    results = []

    for param, new_value in changes.items():
        new_params = base_params.copy()
        if param == 'portfolio_weights[0]':
            # Adjust other weights proportionally
            old_stock = base_params['portfolio_weights'][0]
            other_weights = base_params['portfolio_weights'][1:]
            other_total = sum(other_weights)
            if other_total > 0:
                factor = (1 - new_value) / other_total
                new_params['portfolio_weights'] = [new_value] + [w * factor for w in other_weights]
            else:
                new_params['portfolio_weights'] = [new_value, 1 - new_value, 0]
        elif param.startswith('income_fraction') or param.startswith('wealth_fraction'):
            new_params[param] = max(0, min(1, new_value))
        else:
            new_params[param] = new_value
        
        new_utility = calculate_utility(new_params, n_sims, risk_aversion)
        percent_change = (new_utility - base_utility) / abs(base_utility) * 100

        results.append({
            'parameter': param,
            'old_value': base_params[param] if param != 'portfolio_weights[0]' else base_params['portfolio_weights'][0],
            'new_value': new_value,
            'percent_change': percent_change
        })

    return results

# Helper function to convert risk aversion option to numerical value
def get_risk_aversion(option):
    if option == 'Low':
        return 0  # Linear utility (risk-neutral)
    elif option == 'Medium':
        return 1  # Log utility
    elif option == 'High':
        return 3  # High risk aversion
    else:
        raise ValueError("Invalid risk aversion option. Choose 'low', 'mid', or 'high'.")