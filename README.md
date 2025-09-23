# Optimized Cotton Irrigation

Process-based modeling (AquaCrop) and data pipelines for preparing weather/soil inputs, calibrating/validating the crop model, running irrigation scenarios, and generating publication-quality figures for Xinjiang cotton.

This repository contains the core code used in the manuscript to support review and reproducibility.

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

Note: Large figures and manuscript assets are not included. Generate locally via the figure modules if needed.

## Environment Setup

- Python 3.10 or 3.11 recommended
- Create a virtual environment and install dependencies:
  - `python -m venv .venv && source .venv/bin/activate`
  - `pip install -r requirements.txt`
- External dependencies:
  - AquaCrop (Python) for simulation
  - Google Earth Engine (if using the weather extraction scripts). Ensure proper credentials and project are configured.

## Data Layout

Scripts read inputs from relative paths by default (adjust in code if needed):

- Soil and weather examples (see constants in `model/simulation.py`):
  - `../data/grid_10km/aquacrop_inputdata/soil/soil.csv`
  - `../data/grid_10km/aquacrop_inputdata/weather/2000-01-01_2022-12-31/`
  - `../data/grid_10km/aquacrop_inputdata/weather/2022-01-01_2081-12-31/<GCM>/`

If your layout differs, update the constants such as `SOIL_CSV_PATH`, `WEATHER_BASE_DIR`, `FUTURE_WEATHER_ROOT` accordingly.

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

## Typical Workflow

- Optional data acquisition (GEE): `src/weather/gee_extractor.py`, `src/weather/extract_weather.py`
- Weather/soil preparation: `src/weather/*`, `src/soil/*`
- Simulation: `model/simulation.py`
- Analysis and visualization: `model/deficit_thresholds_Kriging.py`, `src/plot_figs/*`, `utils/plot_map_tools.py`

## Notes

- Verify data paths and time ranges (historical/future) before running.
- Adjust parallelism (workers/memory thresholds) in `model/simulation.py` as needed.
- This repository does not store large raw/intermediate data.

## Acknowledgements

- AquaCrop (Python) modeling framework
- Google Earth Engine (data acquisition components)

## Citation

If you use this code or workflow in your research, please cite the associated manuscript (details to be added after publication).
