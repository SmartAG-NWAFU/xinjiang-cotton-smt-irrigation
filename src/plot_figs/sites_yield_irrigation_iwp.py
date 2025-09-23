"""Additional panel: Sites yield/irrigation/IWP scatter or box plots.

Delegates to `PlotAllfigures.fig_sites_yield_irrigation_iwp`.
"""

import os
import sys

_here = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.abspath(os.path.join(_here, '..'))
if _project_root not in sys.path:
    sys.path.append(_project_root)

try:
    from .plot_all_figures import PlotAllfigures  # type: ignore
except Exception:
    from plot_all_figures import PlotAllfigures  # type: ignore


def plot() -> None:
    PlotAllfigures().fig_sites_yield_irrigation_iwp()
