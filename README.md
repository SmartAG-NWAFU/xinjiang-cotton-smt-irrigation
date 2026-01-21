# Water-Saving and Economic Benefits of Soil Moisture Threshold-based Irrigation Strategy for Cotton in Xinjiang under Climate Change

This repository provides the main code base supporting the study on soil moisture threshold-based irrigation strategy (SMTIS) for Xinjiang cotton under historical and future climate scenarios. It couples the AquaCrop model with a nonlinear optimization framework to identify stage-specific thresholds and evaluates impacts on irrigation demand, yield, irrigation water productivity, and economic returns across space and time.

In brief, the workflow: (i) prepares gridded weather and soil inputs; (ii) calibrates and validates AquaCrop against observed canopy cover, biomass, and yield; (iii) searches stage-specific thresholds; (iv) runs baseline/deficit/future simulations; and (v) summarizes results and produces publication-ready figures.

## What is included vs. expected

This repo includes pipeline wiring, utilities, and a small set of reference geodata. Full reproduction requires additional internal modules and large datasets that are not tracked here.

Missing modules (expected on PYTHONPATH):
- `gee_extractor.py` (GEE helper used by soil/dem/weather extraction)
- `management_scenarios.py` and `crop_varieties.py` (AquaCrop management and cultivar parameters)
- `src/plot_figs/plot_all_figures.py` (legacy figure implementation used by wrappers)

## Repository Structure

- `model/`
  - `simulation.py`: AquaCrop batch runner (baseline/deficit/future)
- `src/`
  - `cotton_units/`: Build cotton unit grid and zone labels
  - `soil/`: GEE soil/dem extraction and soil CSV preparation
  - `weather/`: CMIP6 extraction and climate trend summary
  - `plot_figs/`: Figure wrappers that call `PlotAllfigures`
  - `plot_figures.py`: CLI entry to run figure wrappers
- `utils/`: Plot helpers, raster conversion, and zone lookup scripts
- `data/`: Reference shapefiles and zone raster (large inputs not tracked)
- `fig/`: Sample output figure(s)
- `requirements.txt`, `gee.yaml`, `xinjiangcotton.yml`: Dependency specs

## Environment Setup

- Python 3.10+ recommended
- Minimal pip install:
  - `python -m venv .venv && source .venv/bin/activate`
  - `pip install -r requirements.txt`
- Additional packages used by specific scripts:
  - `aquacrop`, `matplotlib`, `geopandas`, `rasterio`, `cartopy`, `cbgeo`
- Conda env snapshots used during development:
  - `conda env create -f xinjiangcotton.yml`
  - `conda env create -f gee.yaml`
- GEE-based scripts require Earth Engine authentication and a local `gee_extractor.py`.

## Expected Data Layout

Most scripts use hard-coded relative paths. The expected layout is:

```
data/
  grid_10km/
    xinjiang_cotton_units.csv
    soil/
      sand_mean.csv
      silt_mean.csv
      clay_mean.csv
      soc_mean.csv
      bdod_mean.csv
    aquacrop_inputdata/
      soil/soil.csv
      weather/2000-01-01_2022-12-31/*.txt
      weather/2022-01-01_2081-12-31/<gcm_model>/*.txt
      units_labels.csv
  xinjiang_zones/
    xinjiang.shp
    xinjiang_zones.tif
  study_area/
    spam2020_v1r0_global_A_COTT_A.tif
```

## Data Preparation

- Cotton units: `python src/cotton_units/create_cotton_units.py`
- Zone labels: `python src/cotton_units/set_cotton_units_labels.py`
- Soil extraction/prep: `python src/soil/extract_soil.py`, then `python src/soil/prepare_soil_data.py`
- DEM extraction: `python src/soil/extract_dem.py`
- Future weather extraction (CMIP6): `python src/weather/future_weather_extract.py`
- Climate trend summary: `python src/weather/climate_change_weather_trends.py`

## Running Simulations

- Configure input paths and scenario parameters in `model/simulation.py`.
- Provide `management_scenarios.py` and `crop_varieties.py` on `PYTHONPATH`.
- Run: `python model/simulation.py` (edit the scenario in `__main__`).
- Tune `MAX_WORKERS_PER_POOL`, `TOTAL_POOLS`, and `MEMORY_THRESHOLD` for your hardware.

## Generating Figures

Figure wrappers live under `src/plot_figs/` and are driven by the orchestrator, but they require the legacy implementation in `src/plot_figs/plot_all_figures.py` (not tracked here):

- Single figure: `python -m src.plot_figures fig1`
- Main figures (fig1-fig6): `python -m src.plot_figures main`
- Supplementary figures (s1-s4): `python -m src.plot_figures supp`
- All: `python -m src.plot_figures all`

## Reproducibility

- Provide required input data (soil, weather) under the layout described above.
- Use the same Python version and dependencies as in `requirements.txt` or the conda env files.
- Run simulations then generate figures with `src/plot_figures.py` to reproduce paper plots.
- Randomness is not used in core simulation; runs are deterministic given the same inputs.

## Typical Workflow

- Cotton unit grid: `src/cotton_units/create_cotton_units.py`, `src/cotton_units/set_cotton_units_labels.py`
- Optional data acquisition (GEE): `src/soil/extract_soil.py`, `src/soil/extract_dem.py`, `src/weather/future_weather_extract.py`
- Weather/soil preparation: `src/soil/prepare_soil_data.py`, `src/weather/climate_change_weather_trends.py`
- Simulation: `model/simulation.py`
- Analysis and visualization: `src/plot_figs/*`, `utils/plot_map_tools.py`, `utils/csv_convert_raster.py`

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
