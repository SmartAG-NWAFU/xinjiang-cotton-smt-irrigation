# -*- coding: utf-8 -*-
# @Time    : 2025/06/08 10:00:00
# @Author  : Bin Chen
# @Description: Process cotton data for Xinjiang region
# @Reference: 
# @Note: 

import os
import pandas as pd
import geopandas as gpd
import rasterio
import numpy as np
import matplotlib.pyplot as plt
import cbgeo
from rasterio.transform import Affine

def setup_paths():
    """设置文件路径"""
    src_path = os.path.dirname(os.path.abspath(__file__))
    paths = {
        'xinjiang_shp': os.path.join(src_path, '../../data/xinjiang_zones/xinjiang.shp'),
        'cotton_tif': os.path.join(src_path, '../../data/study_area/spam2020_v1r0_global_A_COTT_A.tif'),
        'xinjiang_cotton_tif': os.path.join(src_path, '../../data/study_area/xinjiang_cotton_units.tif'),
        'xinjiang_cotton_percentage_tif': os.path.join(src_path, '../../data/study_area/xinjiang_cotton_percentage.tif'),
        'xinjiang_cotton_csv': os.path.join(src_path, '../../data/grid_10km/xinjiang_cotton_units.csv')
    }
    return paths

def crop_cotton_data(xinjiang_shp, cotton_tif, output_tif):
    """裁剪棉花数据到新疆边界"""
    xinjiang = gpd.read_file(xinjiang_shp)
    cbgeo.crop_raster(xinjiang, cotton_tif, mask_outside=True, output_path=output_tif)
    print(f"已裁剪棉花数据到新疆边界: {output_tif}")

def calculate_percentage(input_tif, output_tif):
    """计算棉花种植百分比"""
    with rasterio.open(input_tif) as src:
        data = src.read(1)
        transform = src.transform

    # 计算栅格面积
    x_res = transform[0]
    y_res = transform[4]
    pixel_areas = np.zeros_like(data, dtype=float)

    rows, cols = data.shape
    for row in range(rows):
        for col in range(cols):
            lon, lat = rasterio.transform.xy(transform, row, col)
            lat_dist = 111320 * abs(y_res)
            lon_dist = 111320 * abs(x_res) * np.cos(np.radians(lat))
            pixel_areas[row, col] = lat_dist * lon_dist

    # 转换为公顷并计算百分比
    pixel_areas_ha = pixel_areas / 10000
    cotton_percentage = np.where(pixel_areas_ha > 0, 
                               (data / pixel_areas_ha) * 100, 
                               0)

    # 保存百分比数据
    with rasterio.open(output_tif, 'w',
                      driver='GTiff',
                      height=cotton_percentage.shape[0],
                      width=cotton_percentage.shape[1],
                      count=1,
                      dtype=cotton_percentage.dtype,
                      crs=src.crs,
                      transform=transform,
                      nodata=0) as dst:
        dst.write(cotton_percentage, 1)

    print(f"已计算并保存棉花种植百分比: {output_tif}")
    print(f"最大百分比: {np.max(cotton_percentage):.2f}%")
    print(f"平均百分比: {np.mean(cotton_percentage[cotton_percentage > 0]):.2f}%")

def create_csv(input_tif, output_csv):
    """创建包含经纬度和值的CSV文件"""
    with rasterio.open(input_tif) as src:
        data = src.read(1)
        transform = src.transform

    rows, cols = data.shape
    lon_list = []
    lat_list = []
    value_list = []

    for row in range(rows):
        for col in range(cols):
            lon, lat = rasterio.transform.xy(transform, row, col)
            value = data[row, col]
            if value > 0:
                lon_list.append(lon)
                lat_list.append(lat)
                value_list.append(value)

    df = pd.DataFrame({
        'ID': range(0, len(lon_list)),
        'lon': lon_list,
        'lat': lat_list,
        'percentage': value_list
    })

    df.to_csv(output_csv, index=False)
    print(f"已创建CSV文件: {output_csv}")
    print(f"总行数: {len(df)}")
    print("\n数据预览:")
    print(df.head())

def plot_distribution(input_tif, title):
    """绘制数据分布直方图"""
    with rasterio.open(input_tif) as src:
        data = src.read(1)

    plt.figure(figsize=(10, 6))
    valid_data = data[data > 0]
    plt.hist(valid_data.flatten(), bins=50, alpha=0.7, color='green', edgecolor='black')
    plt.title(title)
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()

def main():
    """主函数"""
    # 设置路径
    paths = setup_paths()

    # 1. 裁剪棉花数据
    # crop_cotton_data(paths['xinjiang_shp'], paths['cotton_tif'], paths['xinjiang_cotton_tif'])

    # 2. 计算百分比并保存
    # calculate_percentage(paths['xinjiang_cotton_tif'], paths['xinjiang_cotton_percentage_tif'])

    # 3. 创建CSV文件
    create_csv(paths['xinjiang_cotton_percentage_tif'], paths['xinjiang_cotton_csv'])

    # 4. 绘制分布图
    # plot_distribution(paths['xinjiang_cotton_percentage_tif'], 'Distribution of Cotton Percentage')

if __name__ == "__main__":
    main()

