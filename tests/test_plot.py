import numpy as np
import toml
from matplotlib.colors import to_hex

import utils.plot as plot_mod


def test_figures_use_streamlit_theme_background():
    # The app's figures must match the Streamlit theme background. This
    # styling used to be a side effect of importing the deleted
    # utilities.py module and silently reverted to white when that
    # import was removed.
    theme = toml.load(".streamlit/config.toml")["theme"]
    fig = plot_mod.plot_model_output(
        {"income": np.ones((5, 4)) * 1000.0}, variables=["income"]
    )
    assert to_hex(fig.get_facecolor()) == theme["backgroundColor"].strip().lower()
    for ax in fig.get_axes():
        assert to_hex(ax.get_facecolor()) == theme["backgroundColor"].strip().lower()
