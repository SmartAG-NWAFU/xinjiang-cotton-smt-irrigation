"""Figures Orchestrator

Provides a simple entry point to generate figures, each defined in
`src/plot_figs/` as a dedicated module. Usage examples:

- Generate study area map:            python -m src.plot_figures fig1
- Generate all main paper figures:    python -m src.plot_figures main
- Generate all supplementary figures: python -m src.plot_figures supp
"""

import argparse
from typing import Callable, Dict

from .plot_figs import (
    plot_fig1_study_areas,
    plot_fig2_calibration_validation,
    plot_fig3_irrigation_thresholds,
    plot_fig_baseline_vs_deficit,
    plot_fig4_deficit_return,
    plot_fig5_future_results,
    plot_fig6_standardized_coefficients,
    plot_figs1_cotton_production,
    plot_figs2_weather_change,
    plot_figs3_correlation_analysis,
    plot_figs4_simulation_results,
    plot_xinjiang_weather_maps,
    plot_cotton_areas_clustered,
    plot_sites_yield_irrigation_iwp,
    plot_test_figure,
)


def _registry() -> Dict[str, Callable[[], None]]:
    return {
        # Main figures
        "fig1": plot_fig1_study_areas,
        "fig2": plot_fig2_calibration_validation,
        "fig3": plot_fig3_irrigation_thresholds,
        "baseline": plot_fig_baseline_vs_deficit,
        "fig4": plot_fig4_deficit_return,
        "fig5": plot_fig5_future_results,
        "fig6": plot_fig6_standardized_coefficients,
        # Supplementary and extras
        "s1": plot_figs1_cotton_production,
        "s2": plot_figs2_weather_change,
        "s3": plot_figs3_correlation_analysis,
        "s4": plot_figs4_simulation_results,
        "wx_maps": plot_xinjiang_weather_maps,
        "clustered": plot_cotton_areas_clustered,
        "sites_panel": plot_sites_yield_irrigation_iwp,
        "test": plot_test_figure,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate figures")
    parser.add_argument(
        "which",
        help="Figure key or group (fig1, fig2, fig3, fig4, fig5, fig6, s1, s2, s3, s4, wx_maps, clustered, sites_panel, test, main, supp, all)",
    )
    args = parser.parse_args()

    reg = _registry()

    if args.which == "main":
        for key in ["fig1", "fig2", "fig3", "fig4", "fig5", "fig6"]:
            reg[key]()
        return
    if args.which == "supp":
        for key in ["s1", "s2", "s3", "s4"]:
            reg[key]()
        return
    if args.which == "all":
        for fn in reg.values():
            fn()
        return

    if args.which not in reg:
        raise SystemExit(f"Unknown figure key: {args.which}. Valid: {', '.join(sorted(reg))}")
    reg[args.which]()


if __name__ == "__main__":
    main()

