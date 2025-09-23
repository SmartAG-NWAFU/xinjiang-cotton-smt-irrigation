"""Figure plotting package.

Each module in this package renders one figure for the paper/supplement.
Use `src/plot_figures.py` (or the legacy `src/plot_all_figures.py`) to run them.
"""

# Re-export convenience functions for orchestration
from .fig1_study_areas import plot as plot_fig1_study_areas
from .fig2_calibration_validation import plot as plot_fig2_calibration_validation
from .fig3_irrigation_thresholds import plot as plot_fig3_irrigation_thresholds
from .baseline_vs_deficit import plot as plot_fig_baseline_vs_deficit
from .fig4_deficit_return import plot as plot_fig4_deficit_return
from .fig5_future_results import plot as plot_fig5_future_results
from .fig6_standardized_coefficients import plot as plot_fig6_standardized_coefficients
from .figs1_cotton_production import plot as plot_figs1_cotton_production
from .figs2_weather_change import plot as plot_figs2_weather_change
from .figs3_correlation_analysis import plot as plot_figs3_correlation_analysis
from .figs4_simulation_results import plot as plot_figs4_simulation_results
from .xinjiang_weather_maps import plot as plot_xinjiang_weather_maps
from .cotton_areas_clustered import plot as plot_cotton_areas_clustered
from .sites_yield_irrigation_iwp import plot as plot_sites_yield_irrigation_iwp
from .test_figure import plot as plot_test_figure

