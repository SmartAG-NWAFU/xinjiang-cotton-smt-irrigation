import os
import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import xy
from pyproj import CRS, Transformer
from pykrige.ok import OrdinaryKriging

# ----------------------------
# 1) 读取站点 + SMT
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
# 2) 读取底图（用于网格坐标、CRS、掩膜等）
# ----------------------------
def load_raster_meta():
    tiff_path = '../data/study_area/xinjiang_cotton_percentage.tif'
    with rasterio.open(tiff_path) as src:
        band = src.read(1)  # 数值本身（仅用于行列尺寸/检查）
        transform = src.transform
        crs = src.crs
        height, width = src.height, src.width
        nodata = src.nodata  # 这里是 -1.0
        # 使用数据集 mask（0=无效，>0=有效）
        # mask = src.read_masks(1)  # uint8
        mask = band > 0
    return band, transform, crs, height, width, nodata, mask

# ----------------------------
# 3) 可选：重投影站点坐标 -> 底图 CRS
# ----------------------------
def reproject_points_if_needed(sites_df, target_crs):
    """假设站点 lon/lat 是 WGS84，经检查如果底图不是 EPSG:4326，就重投影。"""
    if target_crs is None:
        # 保险起见，直接当作经纬度用
        return sites_df['lon'].values, sites_df['lat'].values

    target = CRS.from_user_input(target_crs)
    wgs84 = CRS.from_epsg(4326)

    if target == wgs84:
        return sites_df['lon'].values, sites_df['lat'].values

    transformer = Transformer.from_crs(wgs84, target, always_xy=True)
    x, y = transformer.transform(sites_df['lon'].values, sites_df['lat'].values)
    return np.array(x), np.array(y)

# ----------------------------
# 4) 生成网格坐标（x 向右，y 向下；注意 PyKrige 需要升序）
# ----------------------------
def build_grid_axes(transform, width, height):
    cols = np.arange(width)
    rows = np.arange(height)

    # x 方向（列）——取第 0 行的所有列中心点的 x
    x_coords, _ = xy(transform, np.zeros_like(cols), cols)
    x_coords = np.array(x_coords)  # 1D, 通常升序

    # y 方向（行）——取第 0 列的所有行中心点的 y
    _, y_coords = xy(transform, rows, np.zeros_like(rows))
    y_coords = np.array(y_coords)  # 注意北上的栅格通常是降序（由北到南）

    # PyKrige 需要升序；若 y_coords 是降序，则升序排序并记录需要翻转
    y_ascending = np.all(np.diff(y_coords) > 0)
    if not y_ascending:
        y_coords_sorted = y_coords[::-1]  # 升序
        flip_y = True
    else:
        y_coords_sorted = y_coords
        flip_y = False

    return x_coords, y_coords_sorted, flip_y

# ----------------------------
# 5) Kriging + 掩膜 + 写出
# ----------------------------
def krige_and_save_all_bands(sites, band, transform, crs, height, width, nodata, mask, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    # 站点坐标重投影到底图 CRS
    x_pts, y_pts = reproject_points_if_needed(sites, crs)

    # 生成网格轴（确保传入 PyKrige 的是升序）
    gridx, gridy_sorted, flip_y = build_grid_axes(transform, width, height)

    # 要处理的字段
    fields = ['SMT1', 'SMT2', 'SMT3', 'SMT4']

    for fld in fields:
        print(f'>>> 处理 {fld} ...')
        z_vals = sites[fld].values.astype(float)

        # 建立 OK；变差函数你也可尝试 'spherical' / 'exponential' / 'gaussian'
        OK = OrdinaryKriging(
            x_pts, y_pts, z_vals,
            variogram_model='spherical',
            verbose=False, enable_plotting=False
        )

        # 执行插值（注意 gridy_sorted 是升序）
        z_interp, _ = OK.execute('grid', gridx, gridy_sorted)  # shape: (len(gridy), len(gridx))

        # 若我们把 y 反向过给 kriging，需要翻转回来，使之与原始行顺序一致（top->down）
        if flip_y:
            z_interp = z_interp[::-1, :]

        # 使用数据集 mask 掩膜：mask==0 的位置是无效区
        out = np.full((height, width), nodata, dtype='float32')
        out[mask] = z_interp[mask].astype('float32')

        # 写 GeoTIFF，保留 nodata 标记
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

        print(f'>>> {fld} 完成：{out_path}')

def create_summary_csv(smt_files, output_csv):
    """
    读取4个SMT插值后的tif文件，生成一个CSV文件，包含ID、经纬度、SMT1-4
    smt_files: dict, {'SMT1': 'SMT1.tif', 'SMT2': 'SMT2.tif', ...}
    """
    data_dict = {}

    # 先读取第一个栅格作为基准
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

    # 依次读取其他SMT栅格
    for key, filepath in smt_files.items():
        if key == first_key:
            continue
        with rasterio.open(filepath) as src:
            data = src.read(1)
            data_dict[key] = data.flatten().round(2)

    # 生成DataFrame并保存
    df = pd.DataFrame(data_dict)

    df = df[(df['SMT1'] > 0) & (df['SMT2'] > 0) & (df['SMT3'] > 0) & (df['SMT4'] > 0)].reset_index(drop=True)
    df['ID'] = df.index  # 重新生成从 0 开始的 ID

    df.to_csv(output_csv, index=False)
    print(f"已创建CSV文件: {output_csv}")
    print(f"总行数: {len(df)}")
    print("\n数据预览:")
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
