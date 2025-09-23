"""Figure 1: Study areas map.

This module renders the study area figure by delegating to the legacy
implementation in `src/plot_all_figures.PlotAllfigures.fig1_study_areas`.
"""

import os
import sys

# Ensure project root on sys.path for legacy modules
_here = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.abspath(os.path.join(_here, '..'))
if _project_root not in sys.path:
    sys.path.append(_project_root)

try:
    # Prefer local legacy module within plot_figs
    from .plot_all_figures import PlotAllfigures  # type: ignore
except Exception:
    # Fallback for direct script execution
    from plot_all_figures import PlotAllfigures  # type: ignore


def plot() -> None:
    """Render Figure 1 (study areas)."""
    PlotAllfigures().fig1_study_areas()
