import numpy as np
import pandas as pd
import rasterio


def lookup_id_zones():
    unit_grid = pd.read_csv('../data/grids/units.csv')
    units_zones_tiff_file_path = '../data/xinjiang_zones/xinjiang_zones.tif'
    with rasterio.open(units_zones_tiff_file_path) as src:
        # Read the raster data
        units_zones = src.read(1)
        indices = [src.index(lon, lat) for lon, lat in zip(unit_grid['lon'].values, unit_grid['lat'].values)]
        unit_grid['labels'] = [units_zones[x, y] for x, y in indices]
    return unit_grid.to_csv('../data/grids/aquacrop_inputdata/units_labels.csv', index=False)

def main():
    lookup_id_zones() 

if __name__ == "__main__":  
    main()
