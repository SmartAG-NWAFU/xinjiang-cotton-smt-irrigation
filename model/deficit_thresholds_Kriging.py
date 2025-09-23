import os
import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import xy
from pyproj import CRS, Transformer
from pykrige.ok import OrdinaryKriging

# ----------------------------
# 1) Read sites + SMT values
# ----------------------------
def prepare_sites_optimized_smts():
    experiment_sites = pd.read_csv('../data/xinjiang_zones/experimental_sites.csv')
    sites_optimized_smts = pd.read_csv('../results/simulation10km/optimized_smts_results_7.csv')
    experiment_sites['ID'] = experiment_sites['ID'].astype(str)
    sites_optimized_smts['ID'] = sites_optimized_smts['ID'].astype(str)
    df = sites_optimized_smts.merge(
        experiment_sites[['ID', 'lon', 'lat']],
        on='ID', how='left'
    )
    return df[['ID', 'lon', 'lat', 'SMT1', 'SMT2', 'SMT3', 'SMT4']]

# ----------------------------
# 2) Read base raster (grid, CRS, mask, etc.)
# ----------------------------
def load_raster_meta():
    tiff_path = '../data/study_area/xinjiang_cotton_percentage.tif'
    with rasterio.open(tiff_path) as src:
        band = src.read(1)  # Values (only for shape/checks)
        transform = src.transform
        crs = src.crs
        height, width = src.height, src.width
        nodata = src.nodata  # Here -1.0
        # Use dataset mask (0=invalid, >0=valid)
        # mask = src.read_masks(1)  # uint8
        mask = band > 0
    return band, transform, crs, height, width, nodata, mask

# ----------------------------
# 3) Optional: reproject site coords -> raster CRS
# ----------------------------
def reproject_points_if_needed(sites_df, target_crs):
    """Assume site lon/lat are WGS84. If base raster CRS is not EPSG:4326, reproject."""
    if target_crs is None:
        # Fallback: treat as lon/lat
        return sites_df['lon'].values, sites_df['lat'].values

    target = CRS.from_user_input(target_crs)
    wgs84 = CRS.from_epsg(4326)

    if target == wgs84:
        return sites_df['lon'].values, sites_df['lat'].values

    transformer = Transformer.from_crs(wgs84, target, always_xy=True)
    x, y = transformer.transform(sites_df['lon'].values, sites_df['lat'].values)
    return np.array(x), np.array(y)

# ----------------------------
# 4) Build grid axes (x→right, y→down; PyKrige needs ascending)
# ----------------------------
def build_grid_axes(transform, width, height):
    cols = np.arange(width)
    rows = np.arange(height)

    # x-axis (columns): x of row 0, for all columns
    x_coords, _ = xy(transform, np.zeros_like(cols), cols)
    x_coords = np.array(x_coords)  # 1D, usually ascending

    # y-axis (rows): y of column 0, for all rows
    _, y_coords = xy(transform, rows, np.zeros_like(rows))
    y_coords = np.array(y_coords)  # North-up rasters are often descending (N→S)

    # Ensure ascending for PyKrige; if descending, reverse and mark flip
    y_ascending = np.all(np.diff(y_coords) > 0)
    if not y_ascending:
        y_coords_sorted = y_coords[::-1]  # ascending
        flip_y = True
    else:
        y_coords_sorted = y_coords
        flip_y = False

    return x_coords, y_coords_sorted, flip_y

# ----------------------------
# 5) Kriging + mask + write output
# ----------------------------
def krige_and_save_all_bands(sites, band, transform, crs, height, width, nodata, mask, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    # Reproject site coords to raster CRS
    x_pts, y_pts = reproject_points_if_needed(sites, crs)

    # Build grid axes (ensure ascending for PyKrige)
    gridx, gridy_sorted, flip_y = build_grid_axes(transform, width, height)

    # Fields to process
    fields = ['SMT1', 'SMT2', 'SMT3', 'SMT4']

    for fld in fields:
        print(f'>>> Processing {fld} ...')
        z_vals = sites[fld].values.astype(float)

        # Ordinary Kriging; you may test other variograms 'spherical'/'exponential'/'gaussian'
        OK = OrdinaryKriging(
            x_pts, y_pts, z_vals,
            variogram_model='spherical',
            verbose=False, enable_plotting=False
        )

        # Interpolate (gridy_sorted is ascending)
        z_interp, _ = OK.execute('grid', gridx, gridy_sorted)  # shape: (len(gridy), len(gridx))

        # If y was reversed for kriging, flip back to top→down order
        if flip_y:
            z_interp = z_interp[::-1, :]

        # Apply dataset mask: mask==0 indicates invalid region
        out = np.full((height, width), nodata, dtype='float32')
        out[mask] = z_interp[mask].astype('float32')

        # Write GeoTIFF with nodata preserved
        out_path = os.path.join(out_dir, f'{fld}.tif')
        profile = {
            'driver': 'GTiff',
            'height': height,
            'width': width,
            'count': 1,
            'dtype': 'float32',
            'crs': crs,
            'transform': transform,
            'nodata': nodata,
            'compress': 'LZW'
        }
        with rasterio.open(out_path, 'w', **profile) as dst:
            dst.write(out, 1)

        print(f'>>> {fld} done: {out_path}')

def create_summary_csv(smt_files, output_csv):
    """Read the 4 SMT raster files and generate a CSV with ID, lon, lat, SMT1-4.
    smt_files: dict, e.g., {'SMT1': 'SMT1.tif', 'SMT2': 'SMT2.tif', ...}
    """
    data_dict = {}

    # Use the first raster as the base
    first_key = list(smt_files.keys())[0]
    with rasterio.open(smt_files[first_key]) as src:
        transform = src.transform
        data = src.read(1)
        rows, cols = data.shape

        lon_list = []
        lat_list = []
        for row in range(rows):
            for col in range(cols):
                lon, lat = rasterio.transform.xy(transform, row, col)
                lon_list.append(lon)
                lat_list.append(lat)

        data_dict['ID'] = range(0, len(lon_list))
        data_dict['lon'] = lon_list
        data_dict['lat'] = lat_list
        data_dict[first_key] = data.flatten().round(2)

    # Read remaining SMT rasters
    for key, filepath in smt_files.items():
        if key == first_key:
            continue
        with rasterio.open(filepath) as src:
            data = src.read(1)
            data_dict[key] = data.flatten().round(2)

    # Build DataFrame and save
    df = pd.DataFrame(data_dict)

    df = df[(df['SMT1'] > 0) & (df['SMT2'] > 0) & (df['SMT3'] > 0) & (df['SMT4'] > 0)].reset_index(drop=True)
    df['ID'] = df.index  # Re-generate ID starting from 0

    df.to_csv(output_csv, index=False)
    print(f"Created CSV file: {output_csv}")
    print(f"Total rows: {len(df)}")
    print("\nPreview:")
    print(df.head())


# ----------------------------
# main
# ----------------------------
if __name__ == '__main__':
    # kriging
    sites = prepare_sites_optimized_smts()
    band, transform, crs, height, width, nodata, mask = load_raster_meta()
    output_dir = "../results/simulation10km/kriging_smt"
    krige_and_save_all_bands(sites, band, transform, crs, height, width, nodata, mask, output_dir)
    # create summary csv
    smt_files = {
        'SMT1': '../results/simulation10km/kriging_smt/SMT1.tif',
        'SMT2': '../results/simulation10km/kriging_smt/SMT2.tif',
        'SMT3': '../results/simulation10km/kriging_smt/SMT3.tif',
        'SMT4': '../results/simulation10km/kriging_smt/SMT4.tif'
    }
    create_summary_csv(smt_files, "../results/simulation10km/SMT_summary.csv")
