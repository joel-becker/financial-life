import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import create_style_script as css

# Match figures to the Streamlit theme (backgrounds, text, tick colors).
# This styling was previously an import side effect of the deleted
# utilities.py module; it belongs here, next to the plotting it styles.
plt.rcParams.update(css.mplstyle)


def plot_model_output(model_output, variables=None, alpha=0.03, mean_line_alpha=1, mean_line_width=2,
                      show_grid=True, grid_alpha=0.1, grid_style='dashed', plot_credibility_interval=True,
                      credibility_intervals=[0.5, 0.8, 0.95]):
    if variables is not None:
        model_output = {k: v for k, v in model_output.items() if k in variables}

    ncols = 1
    nrows = len(model_output.keys())
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(6 * ncols, 3 * nrows))

    if isinstance(axs, np.ndarray):
        axs_iter = axs.flatten()
    else:
        axs_iter = np.array([axs])

    for ax in axs_iter:
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)

    formatter = ticker.FuncFormatter(lambda x, pos: "${:,.0f}".format(x))

    for ax, (key, value) in zip(axs_iter, model_output.items()):
        value = np.transpose(value)

        ax.plot(
            value.mean(axis=1),
            color=css.primary_color, 
            alpha=mean_line_alpha, 
            linewidth=mean_line_width
        )

        if plot_credibility_interval:
            sorted_values = np.sort(value, axis=1)
            for i in range(len(credibility_intervals)-1, -1, -1):
                lower_quantile = (1-credibility_intervals[i]) / 2
                upper_quantile = 1 - lower_quantile
                ax.fill_between(
                    range(value.shape[0]),
                    np.quantile(sorted_values, lower_quantile, axis=1),
                    np.quantile(sorted_values, upper_quantile, axis=1),
                    color=css.primary_color, 
                    alpha=0.3 * (i + 1) / len(credibility_intervals),
                    label=f'{int(credibility_intervals[i]*100)}% credibility interval'
                )
            ax.legend()
        else:
            ax.plot(
                value,
                color=css.primary_color,  
                alpha=alpha
            )

        ax.set_title(key.replace("_", " ").title())
        ax.yaxis.set_major_formatter(formatter)
        
        if show_grid:
            ax.grid(color=css.primary_color, linestyle=grid_style, linewidth=0.5, alpha=grid_alpha)

        if np.min(value) >= 0:
            ax.set_ylim(bottom=0)

    try:
        if len(model_output.keys()) % 2 != 0:
            fig.delaxes(axs[-1, -1])
    except:
        pass

    try:
        for ax in axs[-1, :]:
            ax.set_xlabel("Years from present")
    except:
        try:
            for i in range(len(axs_iter)):
                axs[i].set_xlabel("Years from present")
        except:
            axs.set_xlabel("Years from present")
    
    try:
        for ax in axs[:, -0]:
            ax.set_ylabel("2023 USD")
    except:
        try:
            axs[0].set_ylabel("2023 USD")
        except:
            axs.set_ylabel("2023 USD")
        
    plt.tight_layout()
    return fig

