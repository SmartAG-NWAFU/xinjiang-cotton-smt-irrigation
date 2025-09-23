"""Assign cotton unit labels (north/south) by sampling a zones raster.

Renamed from `set_cotton_units_lables.py` to fix a typo in the filename
(`labels`).
"""

import numpy as np
import pandas as pd
import rasterio

def lookup_id_zones():
    unit_grid = pd.read_csv('../../data/grid_10km/xinjiang_cotton_units.csv')
    units_zones_tiff_file_path = '../../data/xinjiang_zones/xinjiang_zones.tif'
    with rasterio.open(units_zones_tiff_file_path) as src:
        # Read the raster data and sample by lon/lat
        units_zones = src.read(1)
        indices = [src.index(lon, lat) for lon, lat in zip(unit_grid['lon'].values, unit_grid['lat'].values)]
        unit_grid['labels'] = [units_zones[x, y] for x, y in indices]
        # Normalize to two classes: 1=south, 2=north
        unit_grid['labels'] = np.where(unit_grid['labels'] == 1, 1, 2)
    return unit_grid.to_csv('../../data/grid_10km/aquacrop_inputdata/units_labels.csv', index=False)

def main():
    lookup_id_zones() 

if __name__ == "__main__":  
    main()
