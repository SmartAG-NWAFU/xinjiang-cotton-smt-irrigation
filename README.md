# Water-Saving and Economic Benefits of Soil Moisture Threshold-based Irrigation Strategy for Cotton in Xinjiang under Climate Change

This repository provides the main code base supporting the study on soil moisture threshold-based irrigation strategy (SMTIS) for Xinjiang cotton under historical and future climate scenarios. It couples the AquaCrop model with a nonlinear optimization framework to identify stage-specific thresholds and evaluates impacts on irrigation demand, yield, irrigation water productivity, and economic returns across space and time.

In brief, the workflow: (i) prepares gridded weather and soil inputs; (ii) calibrates and validates AquaCrop against observed canopy cover, biomass, and yield; (iii) searches stage-specific thresholds; (iv) runs baseline/deficit/future simulations; and (v) summarizes results and produces publication-ready figures.

## Repository Structure

- `src/`
  - `weather/`: Weather preparation and analysis
    - `extract_weather.py`, `future_weather_extract.py`, `gee_extractor.py`
    - `process_era5_grid_weather_data.py`, `prepare_weather_data.py`
    - `calculate_weather_soil_correlation.py`, `climate_change_weather_trends.py`
  - `soil/`: Soil data preparation and extraction
    - `prepare_soil_data.py`, `extract_soil.py`, `extract_dem.py`
  - `cotton_units/`: Cotton unit grid creation and labelling
    - `create_cotton_units.py`, `set_cotton_units_labels.py`
  - `plot_figs/`: One module per figure; `plot_all_figures.py` holds legacy implementations used by wrappers
  - `plot_figures.py`: Orchestrator to run figures from the CLI
- `model/`
  - `simulation.py`: Main simulation driver (baseline/deficit/future, etc.)
  - `deficit_thresholds_Kriging.py`: Interpolate optimized thresholds to grid
  - `metrics.py`: Shared metrics (e.g., GDD)
- `utils/`: Utilities such as `plot_map_tools.py`, `csv_convert_raster.py`, `lookup_id_zones.py`
- `data/`: Data placeholders (not tracked for large files)
- `requirements.txt`: Python dependencies
- `gee.yaml`, `xinjiangcotton.yml`: Example configuration files

## Environment Setup

- Python 3.10 or 3.11 recommended
- Create a virtual environment and install dependencies:
  - `python -m venv .venv && source .venv/bin/activate`
  - `pip install -r requirements.txt`
- External dependencies:
  - AquaCrop (Python) for simulation
  - Google Earth Engine (if using the weather extraction scripts). Ensure proper credentials and project are configured.

## Running Simulations

- Configure input paths and scenario parameters in `model/simulation.py`.
- Execute the simulation (examples within the file show how to run baseline/deficit/future scenarios).

## Generating Figures

Figures are modularized under `src/plot_figs/` and can be run via the orchestrator:

- Single figure: `python -m src.plot_figures fig1`
- Main figures (fig1–fig6): `python -m src.plot_figures main`
- Supplementary figures (S1–S4): `python -m src.plot_figures supp`
- All: `python -m src.plot_figures all`

Each figure module saves output to the corresponding `figs/` subdirectory.

## Reproducibility

- Provide required input data (soil, weather) under the layout described above.
- Use the same Python version and dependencies as in `requirements.txt`.
- Run simulations then generate figures with `src/plot_figures.py` to reproduce paper plots.
- Randomness is not used in core simulation; runs are deterministic given the same inputs.

## Typical Workflow

- Optional data acquisition (GEE): `src/weather/gee_extractor.py`, `src/weather/extract_weather.py`
- Weather/soil preparation: `src/weather/*`, `src/soil/*`
- Simulation: `model/simulation.py`
- Analysis and visualization: `model/deficit_thresholds_Kriging.py`, `src/plot_figs/*`, `utils/plot_map_tools.py`

## Notes

- Verify data paths and time ranges (historical/future) before running.
- Adjust parallelism (workers/memory thresholds) in `model/simulation.py` as needed.
- This repository does not store large raw/intermediate data.

## License and Copyright

- Code is released under the MIT License (see `LICENSE`).
- Copyright © 2025 SmartAG@NWAFU.
- External datasets and third-party assets may have their own licenses. Ensure you have permission to use and redistribute them. Do not commit or redistribute proprietary data.

## Acknowledgements

- AquaCrop (Python) modeling framework
- Google Earth Engine (data acquisition components)

## Contact

For questions about the code or workflow, please open an issue or contact the corresponding authors listed in the manuscript.
