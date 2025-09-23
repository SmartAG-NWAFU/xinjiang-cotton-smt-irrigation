import os
import sys
script_directory = os.path.dirname(os.path.abspath(__file__))
# 获取项目根目录的路径（假设model和utils是同级目录）
project_root = os.path.dirname(script_directory)
# 将项目根目录添加到sys.path
sys.path.append(project_root)
from utils.plot_map_tools import add_north, add_scalebar

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from matplotlib.colors import ListedColormap
import cartopy.feature as cfeature
import cartopy.io.shapereader as shapereader
import matplotlib.ticker as mticker
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import geopandas as gpd
import rasterio
import cartopy.crs as ccrs
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.patches as patches
import matplotlib.lines as mlines
from matplotlib.colorbar import ColorbarBase
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.ticker import MultipleLocator
from matplotlib.ticker import MaxNLocator
from matplotlib.patches import Patch
from sklearn.linear_model import LinearRegression
from statsmodels.regression.mixed_linear_model import MixedLM
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")
import cbgeo

XINJIANG = gpd.read_file('../data/xinjiang_zones/xinjiang.shp')
XINJIANG_CITY = gpd.read_file('../data/xinjiang_zones/city.shp')
XINJIANG_ZONES2 = gpd.read_file('../data/xinjiang_zones/xinjiang_zones2.shp')

class PlotAllfigures:
    def __init__(self):
        pass
    def fig1_study_areas(self):
        # Set font to Times New Roman
        plt.rcParams['font.family'] = 'Times New Roman'
        plt.rcParams['font.serif'] = ['Times New Roman']
        
        # Define unified font sizes
        FONT_SIZE_SMALL = 6
        FONT_SIZE_MEDIUM = 8
        FONT_SIZE_LARGE = 10
        
        # Create figure and subplots
        fig = plt.figure(figsize=(6.89, 6.89))
        gs = fig.add_gridspec(4, 2, wspace=-0.15, hspace=0.15)
        ax1 = fig.add_subplot(gs[:2, 0], projection=ccrs.PlateCarree())
        ax2 = fig.add_subplot(gs[2:, 0], projection=ccrs.PlateCarree())
        ax3 = fig.add_subplot(gs[0, 1], projection=ccrs.PlateCarree())
        ax4 = fig.add_subplot(gs[1, 1], projection=ccrs.PlateCarree())
        ax5 = fig.add_subplot(gs[2, 1], projection=ccrs.PlateCarree())
        ax6 = fig.add_subplot(gs[3, 1], projection=ccrs.PlateCarree())

        # Load geographic data from JSON file using geopandas
        xinjiang = gpd.read_file('../data/xinjiang_zones/xinjiang.shp')
        xinjiang_city = gpd.read_file('../data/xinjiang_zones/city.shp')
        north_xinjiang = gpd.read_file('../data/xinjiang_zones/xinjiang_north.shp')
        south_xinjiang = gpd.read_file('../data/xinjiang_zones/xinjiang_south.shp')
        # zones2_xinjiang = gpd.read_file('../data/xinjiang_zones/xinjiang_zones2.shp')
        experiment_sites = pd.read_csv('../data/xinjiang_zones/experimental_sites.csv')
        china = gpd.read_file('../data/xinjiang_zones/china.json')

        # ax1
        cmap = ListedColormap(["#33A02C", "#B2DF8A", "#FDBF6F", "#1F78B4", "#999999", "#E31A1C", "#E6E6E6", "#A6CEE3"])
        with rasterio.open('../data/study_area/xinjiang_dem5km.tif') as src:
            data = src.read(1)
            left, bottom, right, top = src.bounds

        data_mask = np.ma.masked_less_equal(data, 0)
        extent = [left, right, bottom, top]
        # print(extent)
        im = ax1.imshow(
            data_mask, 
            origin='upper',
            transform=ccrs.PlateCarree(),  
            cmap=cmap,
            zorder=2,
            extent=extent        
        )

        # Create a colorbar
        cax = fig.add_axes([0.44, 0.54, 0.02, 0.08])
        cbar = fig.colorbar(im, cax=cax, orientation='vertical')
        cbar.set_ticks([0,2500,5000])
        cbar.set_label('Elevation (m)', size=FONT_SIZE_MEDIUM)

        xinjiang.plot(ax=ax1, facecolor='none', edgecolor='black', lw=0.3, zorder=3, alpha=0.5)
        xinjiang_city.plot(ax=ax1, facecolor='none', edgecolor='black', lw=0.3, zorder=3, alpha=0.5)
        add_north(ax1, labelsize=FONT_SIZE_LARGE)
        add_scalebar(ax1, y=34.5, x=73, length_km=300, lw=2, size=FONT_SIZE_SMALL, lat_range=(34, 50))

        # Plot province and city boundaries with adjusted latitude and longitude range
        chinaax1 = fig.add_axes([0.125, 0.74, 0.15, 0.15], projection=ccrs.PlateCarree())
        chinaax1.set_extent([-180, 180, -90, 90], crs=ccrs.PlateCarree())  # Adjust the latitude and longitude range
        chinaax1.add_feature(cfeature.LAND)
        chinaax1.add_feature(cfeature.COASTLINE, lw=0.3)
        # chinaax1.add_feature(cfeature.OCEAN)
        china.plot(ax=chinaax1, color='white', edgecolor='black')
        xinjiang.plot(ax=chinaax1, color='red', edgecolor='red')
        chinaax1.set_xticks([])  # Remove x-axis ticks
        chinaax1.set_yticks([])  # Remove y-axis ticks

        ax1.set_xticks(np.arange(np.floor(left), np.ceil(right) + 1, step=(np.ceil(right) - np.floor(left)) // 4), crs=ccrs.PlateCarree())
        ax1.set_yticks(np.arange(np.floor(bottom), np.ceil(top) + 1, step=(np.ceil(top) - np.floor(bottom)) // 4), crs=ccrs.PlateCarree())
        ax1.xaxis.set_major_formatter(LONGITUDE_FORMATTER)
        ax1.yaxis.set_major_formatter(LATITUDE_FORMATTER)
        ax1.tick_params(axis='x', labelsize=FONT_SIZE_MEDIUM)
        ax1.tick_params(axis='y', labelsize=FONT_SIZE_MEDIUM)
        
        ## ax2
        # Load and mask raster data
        with rasterio.open('../data/study_area/xinjiang_cotton_percentage.tif') as src:
            data = src.read(1)
            bounds = src.bounds
        data_mask = np.ma.masked_less_equal(data, 0)
        extent = [bounds.left, bounds.right, bounds.bottom, bounds.top]
        # Plot cotton ratio map
        ax2.imshow(data_mask, origin='upper', transform=ccrs.PlateCarree(), cmap='Greens', zorder=2, extent=extent, vmin=0, vmax=100)
        xinjiang.plot(ax=ax2, transform=ccrs.PlateCarree(), facecolor='none', edgecolor='black', lw=0.7, zorder=3, alpha=0.8)
        xinjiang_city.plot(ax=ax2, transform=ccrs.PlateCarree(), facecolor='none', edgecolor='black', lw=0.3, zorder=3, alpha=0.5)
        north_xinjiang.plot(ax=ax2, transform=ccrs.PlateCarree(), facecolor='none', edgecolor='orange', lw=0.5, zorder=3, alpha=0.8)    
        south_xinjiang.plot(ax=ax2, transform=ccrs.PlateCarree(), facecolor='none', edgecolor='blue', lw=0.5, zorder=3, alpha=0.8)


        # Add site markers and annotations
        def add_experiment_sites(ax, experiment_sites):
            for _, row in experiment_sites.iterrows():
                ax.plot(row['lon'], row['lat'], marker='^', color='red', markersize=3, transform=ccrs.PlateCarree())
                # add annotate label
                if row['name'] == 'Aksu':
                    ax.annotate(row['name'], xy=(row['lon'], row['lat']), xytext=(row['lon'] + 0.5, row['lat'] + 0.5),
                                color='black', fontsize=FONT_SIZE_SMALL, ha='center')
                elif row['name'] == 'Fukang':
                    ax.annotate(row['name'], xy=(row['lon'], row['lat']), xytext=(row['lon'] + 2, row['lat']),
                                arrowprops=dict(arrowstyle='->', color='black'), color='black', fontsize=FONT_SIZE_SMALL, ha='center')
                elif row['name'] == 'Qira':
                    ax.annotate(row['name'], xy=(row['lon'], row['lat']), xytext=(row['lon'] + 0.5, row['lat'] + 0.5),
                                color='black', fontsize=FONT_SIZE_SMALL, ha='center')
                elif row['name'] == 'WeiliCountry':
                    ax.annotate('Weili County', xy=(row['lon'], row['lat']), xytext=(row['lon'] + 2, row['lat'] + 1),
                                arrowprops=dict(arrowstyle='->', color='black'), color='black', fontsize=FONT_SIZE_SMALL, ha='center')
                elif row['name'] == 'Shihezi':
                    ax.annotate(row['name'], xy=(row['lon'], row['lat']), xytext=(row['lon'] - 0.5, row['lat'] - 1.5),
                                arrowprops=dict(arrowstyle='->', color='black'), color='black', fontsize=FONT_SIZE_SMALL, ha='center')
                elif row['name'] == 'Tumushuke':
                    ax.annotate(row['name'], xy=(row['lon'], row['lat']), xytext=(row['lon'] - 2, row['lat'] + 1),
                                arrowprops=dict(arrowstyle='->', color='black'), color='black', fontsize=FONT_SIZE_SMALL, ha='center')
                elif row['name'] == 'Korla':
                    ax.annotate(row['name'], xy=(row['lon'], row['lat']), xytext=(row['lon'] + 1, row['lat'] - 2),
                                arrowprops=dict(arrowstyle='->', color='black'), color='black', fontsize=FONT_SIZE_SMALL, ha='center')
                elif row['name'] == 'Huyanghe':
                    ax.annotate(row['name'], xy=(row['lon'], row['lat']), xytext=(row['lon'] - 1.9, row['lat'] - 1.3),
                                arrowprops=dict(arrowstyle='->', color='black'), color='black', fontsize=FONT_SIZE_SMALL, ha='center')
                elif row['name'] == 'Urumqi':
                    ax.annotate(row['name'], xy=(row['lon'], row['lat']), xytext=(row['lon'] + 1.5, row['lat'] + 1.5),
                                arrowprops=dict(arrowstyle='->', color='black'), color='black', fontsize=FONT_SIZE_SMALL, ha='center')
                elif row['name'] == 'ShawanCountry':
                    ax.annotate('Shawan County', xy=(row['lon'], row['lat']), xytext=(row['lon'] + 1.5, row['lat'] + 2),
                                arrowprops=dict(arrowstyle='->', color='black'), color='black', fontsize=FONT_SIZE_SMALL, ha='center')

        # legend_elements = [
        #     Patch(edgecolor='orange', facecolor='none', label='Northe Xinjiang', linewidth=0.8, alpha=0.8),
        #     Patch(edgecolor='blue', facecolor='none', label='Southe Xinjiang', linewidth=0.8, alpha=0.8),
        #     mlines.Line2D([], [], color='red', marker='^',label='Experimental sites', linestyle='None', markersize=6),
        #     mlines.Line2D([], [], color='black', marker='o', label='Weather stations', linestyle='None', markersize=5)
        # ]
        # ax2.legend(handles=legend_elements, 
        #         loc='upper left',
        #         title="",
        #         frameon=False,
        #         fontsize=FONT_SIZE_MEDIUM)

        # cax2 = fig.add_axes([0.37, 0.14, 0.16, 0.02])
        # cbar2 = plt.colorbar(ax2.images[0], cax=cax2, orientation='horizontal')
        # cbar2.ax.xaxis.set_ticks_position('top')
        # cbar2.ax.xaxis.set_label_position('top')
        # cbar2.set_label('Planting frequency(%)', fontsize=FONT_SIZE_MEDIUM)
        # cbar2.set_ticks([0, 20, 40, 60, 80, 100])
        # cbar2.set_ticklabels([0, 20, 40, 60, 80, 100], fontsize=FONT_SIZE_SMALL)

        # ax2.set_xticks(np.arange(np.floor(left), np.ceil(right) + 1, step=(np.ceil(right) - np.floor(left)) // 4), crs=ccrs.PlateCarree())
        # ax2.set_yticks(np.arange(np.floor(bottom), np.ceil(top) + 1, step=(np.ceil(top) - np.floor(bottom)) // 4), crs=ccrs.PlateCarree())
        # ax2.xaxis.set_major_formatter(LONGITUDE_FORMATTER)
        # ax2.yaxis.set_major_formatter(LATITUDE_FORMATTER)
        # ax2.tick_params(axis='x', labelsize=FONT_SIZE_MEDIUM)
        # ax2.tick_params(axis='y', labelsize=FONT_SIZE_MEDIUM)

        # Define helper functions for additional data layers
        def add_tmean():
            with rasterio.open('../results/xinjiang_weather10km/mean_tmean.tif') as src:
                zones_data = src.read(1)
                zones_bounds = src.bounds
            data_mask_tman = np.ma.masked_invalid(zones_data)
            vmin = 15
            vmax = 25
            ax3.imshow(
                data_mask_tman,
                origin='upper',
                transform=ccrs.PlateCarree(),
                extent=[zones_bounds.left, zones_bounds.right, zones_bounds.bottom, zones_bounds.top],
                cmap='YlOrRd',
                norm=Normalize(vmin=vmin, vmax=vmax),
                zorder=2
            )
            xinjiang.plot(ax=ax3, transform=ccrs.PlateCarree(), facecolor='none', edgecolor='gray', lw=0.3, zorder=3, alpha=0.5)
            # xinjiang_city.plot(ax=ax3, transform=ccrs.PlateCarree(), facecolor='none', edgecolor='gray', lw=0.3, zorder=3, alpha=0.5)
            cax3 = fig.add_axes([0.8, 0.71, 0.01, 0.16])
            cbar3 = ColorbarBase(cax3, cmap='YlOrRd', norm=Normalize(vmin=vmin, vmax=vmax), extend='both')
            cbar3.ax.set_ylabel('Temperature (℃)', fontsize=FONT_SIZE_MEDIUM)

            # 在ax3的右下角添加直方图
            hist_ax = ax3.inset_axes([0.62, 0.15, 0.35, 0.35])  # [left, bottom, width, height]
            hist_ax.hist(data_mask_tman.compressed(), bins=30, color='gray', alpha=0.7)
            hist_ax.grid(True, linestyle='--', alpha=0.3)
            hist_ax.tick_params(axis='both', which='major', labelsize=FONT_SIZE_MEDIUM)

        def add_prec():
            with rasterio.open('../results/xinjiang_weather10km/mean_prcp.tif') as src:
                prcp_data = src.read(1)
                prcp_bounds = src.bounds
            data_mask_pre =np.ma.masked_invalid(prcp_data)
            vmin = 0
            vmax = 300
            ax4.imshow(data_mask_pre, origin='upper', transform=ccrs.PlateCarree(),
                        extent=[prcp_bounds.left, prcp_bounds.right, prcp_bounds.bottom, prcp_bounds.top], 
                        cmap='YlGnBu',  norm=Normalize(vmin=vmin, vmax=vmax),zorder=2)
            xinjiang.plot(ax=ax4, transform=ccrs.PlateCarree(), facecolor='none', edgecolor='gray', lw=0.3, zorder=3, alpha=0.5)
            # xinjiang_city.plot(ax=ax4, transform=ccrs.PlateCarree(), facecolor='none', edgecolor='gray', lw=0.3, zorder=3, alpha=0.5)
            cax4 = fig.add_axes([0.8, 0.51, 0.01, 0.16])
            cbar4 = ColorbarBase(cax4, cmap='YlGnBu', norm=Normalize(vmin=vmin, vmax=vmax), extend='both')
            cbar4.ax.set_ylabel('Precipitation (mm)', fontsize=FONT_SIZE_MEDIUM)
            # 在ax4的右下角添加直方图
            hist_ax = ax4.inset_axes([0.62, 0.15, 0.35, 0.35])  # [left, bottom, width, height]
            hist_ax.hist(data_mask_pre.compressed(), bins=30, color='gray', alpha=0.7)
            hist_ax.grid(True, linestyle='--', alpha=0.3)
            hist_ax.tick_params(axis='both', which='major', labelsize=FONT_SIZE_MEDIUM)

        def add_ete0():
            with rasterio.open('../results/xinjiang_weather10km/mean_et.tif') as src:
                et0_data = src.read(1)
                et0_bounds = src.bounds
            data_mask_et0 =np.ma.masked_invalid(et0_data)
            vmin = 750
            vmax = 1000
            ax5.imshow(data_mask_et0, origin='upper', transform=ccrs.PlateCarree(), 
                    extent=[et0_bounds.left, et0_bounds.right, et0_bounds.bottom, et0_bounds.top],
                    cmap='Blues', norm=Normalize(vmin=vmin, vmax=vmax), zorder=2)
            xinjiang.plot(ax=ax5, transform=ccrs.PlateCarree(), facecolor='none', edgecolor='gray', lw=0.3, zorder=3, alpha=0.5)
            # xinjiang_city.plot(ax=ax5, transform=ccrs.PlateCarree(), facecolor='none', edgecolor='gray', lw=0.3, zorder=3, alpha=0.5)
            cax5 = fig.add_axes([0.8, 0.32, 0.01, 0.16])
            cbar5 = ColorbarBase(cax5, cmap='Blues', norm=Normalize(vmin=vmin, vmax=vmax), extend='both')
            # cbar5.locator = MultipleLocator(base=100)
            # cbar5.update_ticks()
            cbar5.ax.set_ylabel('$ET_{0}$ (mm)', fontsize=FONT_SIZE_MEDIUM)

            # 在ax5的右下角添加直方图
            hist_ax = ax5.inset_axes([0.62, 0.15, 0.35, 0.35])  # [left, bottom, width, height]
            hist_ax.hist(data_mask_et0.compressed(), bins=30, color='gray', alpha=0.7)
            hist_ax.grid(True, linestyle='--', alpha=0.3)
            hist_ax.tick_params(axis='both', which='major', labelsize=FONT_SIZE_MEDIUM)

        def add_gdd10():
            with rasterio.open('../results/xinjiang_weather10km/mean_gdd.tif') as src:
                thermal_data = src.read(1)
                thermal_bounds = src.bounds
            data_mask_thermal = np.ma.masked_invalid(thermal_data)
            vmin = 1000
            vmax = 2500
            ax6.imshow(data_mask_thermal, origin='upper', transform=ccrs.PlateCarree(), 
                    extent=[thermal_bounds.left, thermal_bounds.right, thermal_bounds.bottom, thermal_bounds.top],
                      cmap='Oranges',norm=Normalize(vmin=vmin, vmax=vmax), zorder=2)
            xinjiang.plot(ax=ax6, transform=ccrs.PlateCarree(), facecolor='none', edgecolor='gray', lw=0.3, zorder=3, alpha=0.5)
            # xinjiang_city.plot(ax=ax6, transform=ccrs.PlateCarree(), facecolor='none', edgecolor='gray', lw=0.3, zorder=3, alpha=0.5)
            cax6 = fig.add_axes([0.8, 0.12, 0.01, 0.16])
            cbar6 = ColorbarBase(cax6, cmap='Oranges', norm=Normalize(vmin=vmin, vmax=vmax), extend='both')
            # cbar6.locator = MultipleLocator(base=500)
            # cbar6.update_ticks()
            cbar6.ax.set_ylabel('Growing degree days(℃·day)', fontsize=FONT_SIZE_MEDIUM)

            # 在ax6的右下角添加直方图
            hist_ax = ax6.inset_axes([0.62, 0.15, 0.35, 0.35])  # [left, bottom, width, height]
            hist_ax.hist(data_mask_thermal.compressed(), bins=30, color='gray', alpha=0.7)
            hist_ax.grid(True, linestyle='--', alpha=0.3)
            hist_ax.tick_params(axis='both', which='major', labelsize=FONT_SIZE_MEDIUM)

        def add_character_number():
            # ax1.set_position([0.06, 0.06, 0.6, 0.9])
            # ax2.set_position([0.057, 0.73, 0.23, 0.23])
            # ax3.set_position([0.57, 0.68, 0.28, 0.28])
            # ax4.set_position([0.57, 0.37, 0.28, 0.28])
            # ax5.set_position([0.57, 0.06, 0.28, 0.28])
            ax1.text(0.5, 0.98, '(a)', transform=ax1.transAxes, verticalalignment='top', horizontalalignment='left', fontsize=FONT_SIZE_LARGE)
            ax2.text(0.5, 0.98, '(b)', transform=ax2.transAxes, verticalalignment='top', horizontalalignment='left', fontsize=FONT_SIZE_LARGE)
            ax3.text(0.05, 0.95, '(c)', transform=ax3.transAxes, verticalalignment='top', horizontalalignment='left', fontsize=FONT_SIZE_LARGE)
            ax4.text(0.05, 0.95, '(d)', transform=ax4.transAxes, verticalalignment='top', horizontalalignment='left', fontsize=FONT_SIZE_LARGE)
            ax5.text(0.05, 0.95, '(e)', transform=ax5.transAxes, verticalalignment='top', horizontalalignment='left', fontsize=FONT_SIZE_LARGE)
            ax6.text(0.05, 0.95, '(f)', transform=ax6.transAxes, verticalalignment='top', horizontalalignment='left', fontsize=FONT_SIZE_LARGE)

        def add_weather_stations(ax2):
            df = pd.read_csv('../data/study_area/st_all_new.csv', encoding='gbk')
            xinjiang_coords = df[df['province'] == '新疆'][['lon', 'lat']]
            ax2.scatter(xinjiang_coords['lon'], xinjiang_coords['lat'], marker='o', color='black', 
                        s=1, transform=ccrs.PlateCarree(), zorder=10)
            
        def plot_study_area():
            # add_weather_stations(ax2)
            add_experiment_sites(ax2, experiment_sites)
            add_prec()
            add_ete0()
            add_gdd10()
            add_tmean()
            # add_character_number()
            
            plt.savefig('../figs/paper_figs/fig1_study_areas_flower.jpg', dpi=300, bbox_inches='tight')

        plot_study_area()

    def fig_sites_yield_irrigation_iwp(self):
        df = pd.read_csv(os.path.join(script_directory, '../results/sites_yield_irrigation_statisic.csv'))
        df.columns = [c.strip() for c in df.columns]
        df['years'] = df['years'].astype(int)
        df = df[df['years'].isin(range(2005, 2022))]
        grouped = df.groupby('years').agg({
            'yield_sta(t/ha)': ['mean', 'std'],
            'sum_depth(mm)': ['mean', 'std'],
            'iwp(kg/m3)': ['mean', 'std']
        })
        grouped.columns = ['_'.join(col).strip() for col in grouped.columns.values]
        grouped = grouped.reset_index()
        plt.rcParams['font.family'] = 'Times New Roman'
        plt.rcParams['font.size'] = 11
        sns.set_style('white')
        color_list = ['#1f77b4', '#ff7f0e', '#2ca02c']
        label_list = [
            'Yield (t/ha)',
            'Irrigation (mm)',
            'IWP (kg/m³)'
        ]
        y_keys = [
            ('yield_sta(t/ha)_mean', 'yield_sta(t/ha)_std'),
            ('sum_depth(mm)_mean', 'sum_depth(mm)_std'),
            ('iwp(kg/m3)_mean', 'iwp(kg/m3)_std')
        ]
        fig, axes = plt.subplots(3, 1, figsize=(6.89, 6), sharex=True)
        x_ticks = [2005, 2007, 2009, 2011, 2013, 2015, 2017, 2019, 2021]
        # 计算全局均值
        global_means = [
            grouped['yield_sta(t/ha)_mean'].mean(),
            grouped['sum_depth(mm)_mean'].mean(),
            grouped['iwp(kg/m3)_mean'].mean()
        ]
        for i, ax in enumerate(axes):
            y_mean, y_std = y_keys[i]
            ax.plot(grouped['years'], grouped[y_mean], color=color_list[i], lw=2, marker='o', markersize=4, label=label_list[i])
            ax.fill_between(grouped['years'],
                            grouped[y_mean] - grouped[y_std],
                            grouped[y_mean] + grouped[y_std],
                            color=color_list[i], alpha=0.06, linewidth=0)
            # 添加全局均值虚线
            ax.axhline(global_means[i], color=color_list[i], linestyle='--', linewidth=1.3, label='Mean')
            # 添加均值数值标注
            if i == 0:
                yy_val = global_means[i] + 0.5
            elif i == 1:
                yy_val = global_means[i] + 300
            elif i == 2:
                yy_val = global_means[i] + 0.1
            x_min = grouped['years'].min()
            # 右上角稍上方
            ax.text(x_min, yy_val, f"{global_means[i]:.2f}", color=color_list[i], fontsize=11, va='bottom', ha='left', fontweight='bold', backgroundcolor='white')
            ax.set_ylabel(label_list[i], fontsize=12)
            ax.tick_params(axis='y', labelsize=11, direction='out', length=4, width=1.1, right=False, left=True)
            ax.yaxis.set_ticks_position('left')
            if i < 2:
                ax.tick_params(axis='x', which='both', bottom=False, labelbottom=False)
            else:
                ax.tick_params(axis='x', labelsize=11, direction='out', length=4, width=1.1, top=False, bottom=True)
                ax.xaxis.set_ticks_position('bottom')
                ax.set_xticks(x_ticks)
        plt.tight_layout()
        outputfile = os.path.join(script_directory, '../figs/paper_figs/')
        os.makedirs(outputfile, exist_ok=True)
        fig.savefig(os.path.join(outputfile, 'fig2_sites_yield_irrigation_iwp_timeseries.jpg'), dpi=300, bbox_inches='tight')
        plt.close(fig)

    def fig2_calibration(self):
        metrics_data = pd.read_csv(os.path.join(script_directory, '../results/calibration/metrics.csv'))
       
        # 设置全局样式
        sns.set_style("white")
        plt.rcParams.update({
            'font.family': 'Times New Roman',
            'font.size': 12
        })

        # 显示名称与子图标题
        cal_names = ['canopy_cover', 'biomass', 'yield_t_ha']
        titles = [
            r'(a) Canopy cover (%)', r'(b) Canopy cover (%)',
            r'(c) Biomass (kg ha$^{-1}$)', r'(d) Biomass (kg ha$^{-1}$)',
            r'(e) Yield (Tonne ha$^{-1}$)', r'(f) Yield (Tonne ha$^{-1}$)'
        ]
        # 颜色设置：校准=浅蓝，验证=浅橙
        colors = {'cal': '#AED6F1', 'val': '#F9E79F'}

        # 创建画布
        fig, ax = plt.subplots(3, 2, figsize=(4.86, 6.9))
        ax = ax.flatten()

        for i, name in enumerate(cal_names):
            df = pd.read_csv(os.path.join(script_directory, f'../results/calibration/{name}.csv'))
            metric_name = 'Dry yield (tonne/ha)' if name == 'yield_t_ha' else name
            df_metrics = metrics_data[metrics_data['calibrated_names'] == metric_name]

            for j, cal_label in enumerate(['cal', 'val']):
                index = i * 2 + j
                ax_ij = ax[index]

                df_filtered = df[df['cal_label'] == cal_label]
                df_metrics_filtered = df_metrics[df_metrics['cal_label'] == cal_label]

                # 散点图
                ax_ij.scatter(df_filtered['obs_values'], df_filtered['sim_values'],
                            edgecolor='black', facecolor=colors[cal_label], s=30)

                # 1:1 线
                max_range = max(df_filtered['obs_values'].max(), df_filtered['sim_values'].max())
                ax_ij.plot([0, max_range], [0, max_range], 'k--', linewidth=1, label='1:1 line')

                # 轴刻度
                if i == 0:
                    ax_ij.set_yticks([0, 25, 50, 75, 100])
                    ax_ij.set_xticks([0, 25, 50, 75, 100])
                elif i == 1:
                    ax_ij.set_yticks([0, 10000, 20000])
                    ax_ij.set_xticks([0, 10000, 20000])
                elif i == 2:
                    ax_ij.set_yticks([0, 2.5, 5.0, 7.5])
                    ax_ij.set_xticks([0, 2.5, 5.0, 7.5])

                is_left_col = index % 2 == 0
                is_bottom_row = index >= 4

                if is_left_col:
                    ax_ij.set_ylabel("Simulated value")
                else:
                    ax_ij.set_ylabel("")
                    ax_ij.tick_params(axis='y', labelleft=False)

                if is_bottom_row:
                    ax_ij.set_xlabel("Observed value")
                else:
                    ax_ij.set_xlabel("")
                    ax_ij.tick_params(axis='x', labelbottom=True)

                ax_ij.tick_params(axis='both', which='both', direction='out', length=3, width=0.8, bottom=True, left=True)

                # 添加评估指标
                if not df_metrics_filtered.empty:
                    r2 = df_metrics_filtered['r2'].values[0]
                    rmse = df_metrics_filtered['rmse'].values[0]
                    d = df_metrics_filtered['d'].values[0]
                    textstr = f"$R^2$: {r2:.2f}\nRMSE: {rmse:.2f}\nd: {d:.2f}"
                    ax_ij.text(0.05, 0.95, textstr, transform=ax_ij.transAxes,
                            fontsize=10, verticalalignment='top',
                            bbox=dict(boxstyle='round', facecolor='white', alpha=0.1))

                # 设置子图标题
                ax_ij.set_title(titles[index], loc='left', fontsize=12)

        # 添加顶部大标题
        # fig.text(0.35, 0.94, 'Calibration', fontsize=10, ha='center', fontfamily='Times New Roman')
        # fig.text(0.75, 0.94, 'Validation', fontsize=10, ha='center', fontfamily='Times New Roman')

        # 构建图例元素
        line_handle = mlines.Line2D([], [], color='black', linestyle='--', label='1:1 line')
        cal_handle = mlines.Line2D([], [], marker='o', color='black', label='Calibration',
                                markerfacecolor='#AED6F1', markersize=6, linestyle='None')
        val_handle = mlines.Line2D([], [], marker='o', color='black', label='Validation',
                                markerfacecolor='#F9E79F', markersize=6, linestyle='None')

        # 添加图例到右下角子图（第三行第二列）
        ax[5].legend(
            handles=[line_handle, cal_handle, val_handle],
            loc='lower right',
            fontsize=11,
            frameon=False
        )
        # 布局调整
        plt.subplots_adjust(wspace=0.25, hspace=0.25)
        # plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.tight_layout()
        # 保存图像
        outputfile = os.path.join(script_directory, '../figs/paper_figs/')
        os.makedirs(outputfile, exist_ok=True)
        fig.savefig(os.path.join(outputfile, 'fig2_calibration_validation_metrics.jpg'), dpi=300, bbox_inches='tight')
        plt.close(fig)

    def fig3_irrigation_threshoulds(self):
        # 绘图配置
        PLOT_CONFIG = {
            'SMT1': {
                'cbar': {'label': 'Emergence (SMT1)', 'vmin': 55, 'vmax': 100, 'step': 5}
            },
            'SMT2': {
                'cbar': {'label': 'Canopy growth (SMT2)', 'vmin': 55, 'vmax': 100, 'step': 5}
            },
            'SMT3': {
                'cbar': {'label': 'Maximum canopy (SMT3)', 'vmin': 55, 'vmax': 100, 'step': 5}
            },
            'SMT4': {
                'cbar': {'label': 'Canopy senescence (SMT4)', 'vmin': 55, 'vmax': 100, 'step': 5}
            }
        }
        def plot_geodata(ax):
            xinjiang = gpd.read_file('../data/xinjiang_zones/xinjiang.shp')
            xinjiang_city = gpd.read_file('../data/xinjiang_zones/city.shp')
            xinjiang.plot(ax=ax, transform=ccrs.PlateCarree(),
                        facecolor='none', edgecolor='gray', lw=0.3, zorder=3, alpha=0.5)
            # xinjiang_city.plot(ax=ax, transform=ccrs.PlateCarree(),
                            # facecolor='none', edgecolor='gray', lw=0.3, zorder=3, alpha=0.5)
            
        def plot_raster_layer(ax, file_path, data_type):
            config = PLOT_CONFIG[data_type]
            with rasterio.open(file_path) as src:
                data = src.read(1)
                bounds = src.bounds
            masked_data = np.ma.masked_less_equal(data, 0)
            im = ax.imshow(
                masked_data, origin='upper', transform=ccrs.PlateCarree(),
                extent=[bounds.left, bounds.right, bounds.bottom, bounds.top],
                cmap='viridis', 
                norm=Normalize(vmin=config['cbar']['vmin'], vmax=config['cbar']['vmax']),
                zorder=2
            )
            # 在ax的右下角添加直方图
            # hist_masked_data = np.ma.masked_outside(data, vmin, vmax)
            hist_masked_data = masked_data
            hist_ax = ax.inset_axes([0.6, 0.098, 0.35, 0.35])  # [left, bottom, width, height]
            hist_ax.hist(hist_masked_data.compressed(), bins=50, color='gray', alpha=0.9)
            # 添加平均值竖线
            mean_val = hist_masked_data.compressed().mean()
            hist_ax.axvline(mean_val, color='red', linestyle='--', linewidth=1)
            hist_ax.tick_params(axis='both', which='major', labelsize=10, colors='black')
            for spine in hist_ax.spines.values():
                spine.set_edgecolor('black')
                spine.set_linewidth(0.6)  
            return im, bounds, mean_val
        
        def setup_axis(ax, bounds, position):
            left, right = bounds.left, bounds.right
            bottom, top = bounds.bottom, bounds.top
            ax.set_xticks(np.arange(np.floor(left), np.ceil(right) + 1, step=(np.ceil(right) - np.floor(left)) // 4), crs=ccrs.PlateCarree())
            ax.set_yticks(np.arange(np.floor(bottom), np.ceil(top) + 1, step=(np.ceil(top) - np.floor(bottom)) // 4), crs=ccrs.PlateCarree())
            ax.xaxis.set_major_formatter(LONGITUDE_FORMATTER)
            ax.yaxis.set_major_formatter(LATITUDE_FORMATTER)
            ax.tick_params(axis='x', labelsize=10)
            ax.tick_params(axis='y', labelsize=10)
            if position in (2, 4):
                ax.tick_params(left=False, labelleft=False)
            else:
                ax.yaxis.set_major_formatter(LATITUDE_FORMATTER)

        def create_geo_axis(fig, position):
            ax = fig.add_subplot(2, 2, position, projection=ccrs.PlateCarree())
            ax.set_facecolor('none')
            return ax
        
        def create_colorbar(fig, cax_position, cbar_config):
            cax = fig.add_axes(cax_position)
            norm = Normalize(vmin=cbar_config['vmin'], vmax=cbar_config['vmax'])
            cbar = ColorbarBase(cax, cmap='viridis', norm=norm,  orientation='horizontal')
            cbar.locator = MultipleLocator(base=cbar_config['step'])
            cbar.update_ticks()
            return cbar
        
        def irrigation_threshoulds():
            fig = plt.figure(figsize=(6.89, 7.2)) 
            subplot_configs = [
                ('SMT1', f'../results/simulation10km/kriging_smt/SMT1.tif', 1),
                ('SMT2', f'../results/simulation10km/kriging_smt/SMT2.tif', 2),
                ('SMT3', f'../results/simulation10km/kriging_smt/SMT3.tif', 3),
                ('SMT4', f'../results/simulation10km/kriging_smt/SMT4.tif', 4)
            ]

            for data_type, file_path, position in subplot_configs:
                ax = create_geo_axis(fig, position)
                im, bounds, mean_val = plot_raster_layer(ax, file_path, data_type)
                plot_geodata(ax)
                setup_axis(ax, bounds, position)

                # ---- 在每个子图上方添加colorbar ----
                config = PLOT_CONFIG[data_type]['cbar']
                bbox = ax.get_position()   # 获取子图的位置 [x0, y0, width, height]
                # colorbar 的位置：紧贴子图上方
                if position in [1,2]:
                    up = 0.11
                else:
                    up = 0.04
                cax_position = [
                    bbox.x0,        # 左边对齐
                    bbox.y1 + up, # 在子图顶部稍微上移
                    bbox.width + 0.01,     # 宽度与子图相同
                    0.02           # 高度
                ]
                cbar = create_colorbar(fig, cax_position, config)
                # 设置标题在 colorbar 上方
                cbar.ax.set_title(config['label'], fontsize=12, pad=4)

                ax.text(0.05, 0.95, f'({chr(96 + position)})',  # chr(97) = 'a', i start from 0
                    transform=ax.transAxes, fontsize=12,
                    va='top', ha='left',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='none'),
                    zorder=12)
                            # 添加文字标注
                ax.text(0.05, 0.80,
                    f"Mean = {mean_val:.1f}", 
                    transform=ax.transAxes,
                    color='black', fontsize=11, ha='left', va='top'
                )


            plt.tight_layout()
            plt.subplots_adjust(wspace=0.14, hspace=0.05)  # hspace 稍微加大，避免挤在一起
            plt.savefig('../figs/paper_figs/fig3_irrigation_threshoulds.jpg', dpi=300, bbox_inches='tight')
            plt.close(fig)

        irrigation_threshoulds()

    def fig_baseline(self):
        # 地理数据加载
        XINJIANG = gpd.read_file('../data/xinjiang_zones/xinjiang.shp')
        XINJIANG_CITY = gpd.read_file('../data/xinjiang_zones/city.shp')

        # 绘图配置
        PLOT_CONFIG = {
            'yield': {
                'cmap': 'Oranges',
                'cbar': {'label': 'Yield (tonne/ha)', 'vmin': 4.5, 'vmax': 7.5, 'step': 0.5}
            },
            'irrigation': {
                'cmap': 'Blues',
                'cbar': {'label': 'Irrigation (mm)', 'vmin': 200, 'vmax': 800, 'step': 100}
            }
        }

        def create_geo_axis(fig, position):
            ax = fig.add_subplot(3, 2, position, projection=ccrs.PlateCarree())
            ax.set_facecolor('none')
            return ax

        def plot_geodata(ax):
            XINJIANG.plot(ax=ax, transform=ccrs.PlateCarree(),
                        facecolor='none', edgecolor='gray', lw=0.3, zorder=3, alpha=0.5)
            XINJIANG_CITY.plot(ax=ax, transform=ccrs.PlateCarree(),
                            facecolor='none', edgecolor='gray', lw=0.3, zorder=3, alpha=0.5)

        def create_colorbar(fig, cax_position, cmap, cbar_config):
            cax = fig.add_axes(cax_position)
            norm = Normalize(vmin=cbar_config['vmin'], vmax=cbar_config['vmax'])
            cbar = ColorbarBase(cax, cmap=cmap, norm=norm, extend='both')
            cbar.locator = MultipleLocator(base=cbar_config['step'])
            cbar.update_ticks()
            cbar.ax.set_ylabel(cbar_config['label'], fontsize=12)
            return cbar

        def plot_raster_layer(ax, file_path, data_type):
            config = PLOT_CONFIG[data_type]
            with rasterio.open(file_path) as src:
                data = src.read(1)
                bounds = src.bounds
            masked_data = np.ma.masked_less_equal(data, 0)
            im = ax.imshow(
                masked_data, origin='upper', transform=ccrs.PlateCarree(),
                extent=[bounds.left, bounds.right, bounds.bottom, bounds.top],
                cmap=config['cmap'], 
                norm=Normalize(vmin=config['cbar']['vmin'], vmax=config['cbar']['vmax']),
                zorder=2
            )
            return im, bounds, masked_data.compressed()

        def setup_axis(ax, bounds, position):
            left, right = bounds.left, bounds.right
            bottom, top = bounds.bottom, bounds.top
            ax.set_xticks(np.arange(np.floor(left), np.ceil(right) + 1, step=(np.ceil(right) - np.floor(left)) // 4), crs=ccrs.PlateCarree())
            ax.set_yticks(np.arange(np.floor(bottom), np.ceil(top) + 1, step=(np.ceil(top) - np.floor(bottom)) // 4), crs=ccrs.PlateCarree())
            ax.xaxis.set_major_formatter(LONGITUDE_FORMATTER)
            ax.yaxis.set_major_formatter(LATITUDE_FORMATTER)
            ax.tick_params(axis='x', labelsize=12)
            ax.tick_params(axis='y', labelsize=12)
            if position in (2, 4):
                ax.tick_params(left=False, labelleft=False)
            else:
                ax.yaxis.set_major_formatter(LATITUDE_FORMATTER)

        def baseline_yield_irrigation(label):
            fig = plt.figure(figsize=(6.89, 8.3)) 
            
            raster_data_storage = {'yield': {}, 'irrigation': {}}
            axes_dict = {}  # 保存每个subplot轴对象

            # 4幅地图
            subplot_configs = [
                ('irrigation', f'../results/analysis10km/history/baseline/Seasonal_irrigation_mm_{label}.tif', 1,  'CI'),
                ('yield', f'../results/analysis10km/history/baseline/Dry_yield_tonne_ha_{label}.tif', 2, 'CI'),
                ('irrigation', '../results/analysis10km/history/deficit/Seasonal_irrigation_mm_90-0-0.tif', 3,'DI'),
                ('yield', '../results/analysis10km/history/deficit/Dry_yield_tonne_ha_90-0-0.tif', 4, 'DI')
            ]

            for data_type, file_path, position, scenario in subplot_configs:
                ax = create_geo_axis(fig, position)
                axes_dict[position] = ax  # 保存轴
                im, bounds, data_vals = plot_raster_layer(ax, file_path, data_type)
                raster_data_storage[data_type][scenario] = data_vals
                plot_geodata(ax)
                setup_axis(ax, bounds, position)
                if position == 1:
                    create_colorbar(fig, [0.49, 0.38, 0.01, 0.60], PLOT_CONFIG[data_type]['cmap'], PLOT_CONFIG[data_type]['cbar'])
                elif position ==2:
                    create_colorbar(fig, [0.99, 0.38, 0.01, 0.60], PLOT_CONFIG[data_type]['cmap'], PLOT_CONFIG[data_type]['cbar'])

            # 概率密度图（ax5, ax6）
            ax5 = fig.add_subplot(3, 2, 5)  
            ax6 = fig.add_subplot(3, 2, 6)  
            axes_dict[5] = ax5
            axes_dict[6] = ax6

            for scenario, style in zip(['CI', 'DI'], ['blue', 'orange']):
                sns.histplot(raster_data_storage['irrigation'][scenario], ax=ax5, label=scenario, color=style, bins=30, kde=True, stat="percent")
                sns.histplot(raster_data_storage['yield'][scenario], ax=ax6, label=scenario, color=style, bins=30, kde=True, stat="percent")
                # sns.kdeplot(raster_data_storage['irrigation'][scenario], ax=ax5, label=scenario, color=style, lw=2)
                # sns.kdeplot(raster_data_storage['yield'][scenario], ax=ax6, label=scenario, color=style, lw=2)

            ax5.set_xlabel("Irrigation (mm)")
            ax6.set_xlabel("Yield (tonne/ha)")
            # ax5.set_yticks([0,0.002,0.004,0.006])
            ax5.set_ylabel("Percent (%)")
            # ax6.set_yticks([0,0.4,0.8,1.2,1.6])
            ax6.set_ylabel("Percent (%)")
            ax6.legend()

            for i in range(1, 7):
                ax = axes_dict[i]
                ax.text(0.05, 0.95, f'({chr(96 + i)})',  # chr(97) = 'a'
                        transform=ax.transAxes, fontsize=12,
                        va='top', ha='left',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='none'),
                        zorder=12)
            
            plt.tight_layout()
            plt.subplots_adjust(wspace=0.35, hspace=0.15)
            plt.savefig('../figs/paper_figs/fig4_baseline_vs_deficit.png', dpi=300, bbox_inches='tight')

        # 调用绘图函数
        baseline_yield_irrigation(label='90-7-0')

    def fig4_deficit_return(self):
        PLOT_CONFIG = {
            'yield_change': {
                'cmap': 'RdYlGn_r',
                'cbar': {'label': 'ΔYield (t ha⁻¹)', 'vmin': -0.5, 'vmax': 1, 'step': 0.5}
            },
            'irr_change': {
                'cmap': 'Blues_r',
                'cbar': {'label': 'ΔIrrigation (mm)', 'vmin': -300, 'vmax': 0, 'step': 100}
            },
            'iwp_change': {
                'cmap': 'Oranges',
                'cbar': {'label': 'ΔIWP (kg m⁻³)', 'vmin': 0.5, 'vmax': 1, 'step': 0.1}
            },
            'potential_profit': {
                'cmap': 'coolwarm',
                'cbar': {'label': 'EB (10^3 CNY)', 'vmin': -1.5, 'vmax': 7.5, 'step': 1.5}
            }
        }
        # period 和 metric 映射
        all_periods = ['history', '2040s_ssp245', '2040s_ssp585', '2070s_ssp245', '2070s_ssp585']
        metrics = [
            ('yield_change', 'Dry_yield_diff.tif'),
            ('irr_change', 'Seasonal_irrigation_diff.tif'),
            ('iwp_change', 'iwp_diff.tif'),
            ('potential_profit', 'potential_profit.tif'),
        ]
        
        def create_axis(fig, row, col):
            ax = fig.add_subplot(5, 4, row * 4 + col + 1, projection=ccrs.PlateCarree())
            ax.set_facecolor('none')
            return ax

        def plot_geodata(ax):
            XINJIANG.plot(ax=ax, transform=ccrs.PlateCarree(),
                        facecolor='none', edgecolor='black', lw=0.3, zorder=3, alpha=0.5)
            # XINJIANG_CITY.plot(ax=ax, transform=ccrs.PlateCarree(),
            #                 facecolor='none', edgecolor='gray', lw=0.3, zorder=3, alpha=0.5)
            XINJIANG_ZONES2.plot(ax=ax, transform=ccrs.PlateCarree(),
                        facecolor='none', edgecolor='black', lw=0.3, zorder=3, alpha=0.5)

        def create_colorbar(fig, ax_position, cmap, cbar_config, vmin, vmax, step):
            cax = fig.add_axes(ax_position)
            norm = Normalize(vmin=vmin, vmax=vmax)
            cbar = ColorbarBase(
                cax, cmap=cmap, norm=norm, extend='both', orientation='horizontal'
            )
            cbar.locator = MultipleLocator(base=step)
            cbar.update_ticks()
            cbar.ax.set_xlabel(cbar_config['label'], fontsize=8)
            cbar.ax.xaxis.set_label_position('top')       # 标签放上面
            cbar.ax.xaxis.set_ticks_position('top')        # 刻度放上面
            cbar.ax.tick_params(labelsize=6, labelrotation=0)  # 刻度水平

        def plot_raster_layer(ax, file_path, data_type):
            config = PLOT_CONFIG[data_type]
            vmin = config['cbar']['vmin']
            vmax = config['cbar']['vmax']
            step = config['cbar']['step']
            with rasterio.open(file_path) as src:
                if data_type == 'potential_profit':
                    data = src.read(1) / 1000
                else:
                    data = src.read(1)
                bounds = src.bounds
            masked_data = np.ma.masked_invalid(data)
            ax.imshow(masked_data, origin='upper',transform=ccrs.PlateCarree(),
                        extent=[bounds.left, bounds.right, bounds.bottom, bounds.top],
                        cmap=config['cmap'], zorder=2,
                        norm=Normalize(vmin=vmin, vmax=vmax))
            ax.set_xticks([])  # 不显示x刻度
            ax.set_yticks([])  # 不显示y刻度
            ax.spines['geo'].set_visible(False)  # 移除地理图的边框

            # 在ax的右下角添加直方图
            hist_masked_data = np.ma.masked_outside(data, vmin, vmax)
            # hist_masked_data = masked_data
            hist_ax = ax.inset_axes([0.54, 0.18, 0.25, 0.25])  # [left, bottom, width, height]
            hist_ax.hist(hist_masked_data.compressed(), bins=50, color='gray', alpha=0.9)
            hist_ax.tick_params(axis='both', which='major', labelsize=4, colors='black')
            for spine in hist_ax.spines.values():
                spine.set_edgecolor('black')
                spine.set_linewidth(0.6)  

            return vmin, vmax, step, masked_data

        def plot_grid():
            all_raster_data = {metric[0]: {period: [] for period in all_periods} for metric in metrics}
            fig = plt.figure(figsize=(6.89, 8.4))
            for row_idx, period in enumerate(all_periods):
                for col_idx, (data_type, metric_filename) in enumerate(metrics):
                    file_path = f'../results/analysis10km/future/return/{period}/{metric_filename}'
                    ax = create_axis(fig, row_idx, col_idx)
                    vmin, vmax, step, raster_data = plot_raster_layer(ax, file_path, data_type)
                    # 保存数据用于 kde 分布图
                    masked_data = np.ma.masked_less_equal(raster_data, 0)
                    all_raster_data[data_type][period] = masked_data.compressed()  # 保留正值
                    plot_geodata(ax)

                    # 在第一行添加 colorbar
                    if row_idx == 0:
                        cbar_x = 0.19 * col_idx + 0.15
                        create_colorbar(fig, [cbar_x, 0.87, 0.17, 0.01],
                                        PLOT_CONFIG[data_type]['cmap'],
                                        PLOT_CONFIG[data_type]['cbar'],
                                        vmin, vmax, step)

                period_labels = ['History', '2040s(SSP2-4.5)', '2040s(SSP5-8.5)', '2070s(SSP2-4.5)', '2070s(SSP5-8.5)']
                # # 在每一行左侧添加 period 标签
                period_label = period_labels[row_idx]
                fig.text(0.1, 0.78 - row_idx * 0.15, period_label, fontsize=8, rotation=90,
                        va='center', ha='left')
            
            fig.subplots_adjust(wspace=-0.2, hspace=-0.2)
            plt.savefig('../figs/paper_figs/fig4_deficit_return_grid_t.jpg', dpi=300, bbox_inches='tight')
        # 执行绘图函数
        plot_grid()

    def fig5_future_results(self):
        # === 映射字典和顺序 ===
        period_map = {
            'history': 'Historical',
            '2040s_ssp245': '2040s(SSP2-4.5)',
            '2040s_ssp585': '2040s(SSP5-8.5)',
            '2070s_ssp245': '2070s(SSP2-4.5)',
            '2070s_ssp585': '2070s(SSP5-8.5)'
        }
        period_labels = list(period_map.values())
        irrigation_methods_order = ['90-7-0', '90-0-0']
        # === 数据导入 ===
        df = pd.read_csv('../results/analysis10km/future/return/different_periods_results.csv')
        df_profit = pd.read_csv('../results/analysis10km/future/return/static_future_return.csv')
        # 替换并排序
        for d in [df, df_profit]:
            d['periods'] = d['periods'].replace(period_map)
            d['periods'] = pd.Categorical(d['periods'], categories=period_labels, ordered=True)
        df['scenario_params'] = pd.Categorical(df['scenario_params'],
                                            categories=irrigation_methods_order,
                                            ordered=True)
        # === 样式设置 ===
        sns.set(style="ticks")
        plt.rcParams.update({
            'font.family': 'Times New Roman',
            'font.size': 12
        })
        irrcolors = ['blue', 'orange']
        # === GCM marker ===
        gcm_list = df['gcms'].unique()
        gcm_markers = ['o', 's', '^', 'D', 'v', '*', 'X', 'P']
        gcm_marker_dict = {gcm: gcm_markers[i % len(gcm_markers)] for i, gcm in enumerate(gcm_list)}

        # === 辅助函数 ===
        def add_gcm_points(ax, data, x, y, hue=None, dodge=True, size=3):
            """添加 GCM 散点"""
            for gcm in gcm_list:
                subset = data[data['gcms'] == gcm]
                sns.stripplot(
                    data=subset, x=x, y=y, hue=hue,
                    ax=ax, dodge=dodge, jitter=True,
                    marker=gcm_marker_dict[gcm], size=size,
                    alpha=0.9, edgecolor='gray', linewidth=0.3,
                    palette='dark:.3', legend=False
                )

        def add_means(ax, data, x, y, hue=None, offset_map=None, color='red', size=3):
            """添加均值红点"""
            means = data.groupby([x] + ([hue] if hue else []))[y].mean().reset_index()
            for _, row in means.iterrows():
                xpos = period_labels.index(row[x])
                if hue and offset_map:
                    xpos += offset_map.get(row[hue], 0)
                ax.plot(xpos, row[y], 'o', color=color, markersize=size, zorder=5)

        def set_yaxis_right(ax):
            """把 y 轴移动到右边"""
            ax.yaxis.set_label_position("right")
            ax.yaxis.tick_right()

        # === 主绘图函数 ===
        def plot_panel(ax, data, y, hue=None, ylabel='', show_legend=False, show_ticks=False, y_right=False):
            sns.boxplot(
                data=data, x='periods', y=y, hue=hue,
                ax=ax, showfliers=False, palette=irrcolors, linewidth=1
            )
            add_gcm_points(ax, data, 'periods', y, hue=hue)

            if hue:
                offset_map = {irrigation_methods_order[0]: -0.2, irrigation_methods_order[1]: 0.2}
            else:
                offset_map = None
            add_means(ax, data, 'periods', y, hue=hue, offset_map=offset_map)

            ax.set_xlabel('')
            ax.set_ylabel(ylabel, size=12)
            if show_ticks:
                ax.tick_params(axis='x', rotation=45, labelsize=11)
            else:
                ax.tick_params(axis='x', bottom=False, top=False, labelbottom=False, labelsize=11)
            ax.get_legend().remove()

            if y_right:
                set_yaxis_right(ax)

        # === 创建画布 ===
        fig, axs = plt.subplots(2, 2, figsize=(6.89, 5.8))
        # 子图1
        plot_panel(axs[0, 0], df, 'Dry yield (tonne/ha)',
                hue='scenario_params', ylabel='Yield (t ha⁻¹)')
        # 子图2（右边 y 轴）
        plot_panel(axs[0, 1], df, 'Seasonal irrigation (mm)',
                hue='scenario_params', ylabel='Seasonal irrigation (mm)', y_right=True)
        # 子图3
        plot_panel(axs[1, 0], df, 'iwp(kg/m3)',
                hue='scenario_params', ylabel='IWP (kg m⁻³)', show_ticks=True)
        # 子图4（右边 y 轴）
        ax4 = axs[1, 1]
        paired_colors = sns.color_palette("Paired")[:4]
        box_palette = {
            'Historical': paired_colors[1],
            '2040s(SSP2-4.5)': paired_colors[2],
            '2040s(SSP5-8.5)': paired_colors[2],
            '2070s(SSP2-4.5)': paired_colors[3],
            '2070s(SSP5-8.5)': paired_colors[3]
        }
        sns.boxplot(data=df_profit, x='periods', y='profit', ax=ax4,
                    showfliers=False, palette=box_palette, linewidth=1)
        add_gcm_points(ax4, df_profit, 'periods', 'profit', hue=None, dodge=False)
        add_means(ax4, df_profit, 'periods', 'profit')
        ax4.set_ylabel('Economic benefits (CNY)', size=12)
        ax4.tick_params(axis='x', rotation=45, labelsize=12)
        ax4.set_xlabel('')
        set_yaxis_right(ax4)

        # 子图标签
        for i, ax in enumerate(axs.flat):
            ax.text(0.05, 0.95, f'({chr(97 + i)})',
                    transform=ax.transAxes, fontsize=12,
                    va='top', ha='left',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='none'),
                    zorder=12)

        # === 图例 ===
        irrigation_handles = [
            mlines.Line2D([], [], color=irrcolors[0], label='Conventional Irrigation', marker='s', linestyle='', markersize=6),
            mlines.Line2D([], [], color=irrcolors[1], label='Stage-specific deficit Irrigation', marker='s', linestyle='')
        ]
        gcm_handles = [
            mlines.Line2D([], [], color='k', marker=gcm_marker_dict[gcm], linestyle='', markersize=6, label=gcm)
            for gcm in [g for g in gcm_list if g != 'historical']
        ]
        mean_handle = mlines.Line2D([], [], color='red', marker='o', linestyle='', label='Mean', markersize=6)
        legend1 = fig.legend(handles=irrigation_handles,
                            loc='lower left', frameon=False,
                            bbox_to_anchor=(0.12, -0.19),
                            fontsize=9, title='Irrigation Methods', title_fontsize=12)
        legend2 = fig.legend(handles=gcm_handles + [mean_handle],
                            loc='lower right', ncol=3, frameon=False,
                            bbox_to_anchor=(0.99, -0.21),
                            fontsize=9, title='Climate Models', title_fontsize=12,
                            columnspacing=0.8)
        fig.add_artist(legend1)
        fig.add_artist(legend2)

        # plt.tight_layout(rect=[0, 0.08, 1, 1])
        fig.subplots_adjust(wspace=0.1, hspace=0.1)
        fig.savefig('../figs/paper_figs/fig5_future_simulation_results_with_gcm_marker.jpg',
                    dpi=300, bbox_inches='tight')
        plt.close(fig)

    def fig6_plot_standardized_coefficients(self):
        plt.rcParams.update({
            'font.family': 'Times New Roman',
            'font.size': 12
        })
        df = pd.read_csv('../results/analysis10km/correlation/standardized_coefficients_by_period.csv')
        feature_rename = {
            'mean_tmean': 'Tavg',
            'mean_prcp': 'Prcp',
            'mean_et': 'ET0',
            'mean_gdd': 'GDD',
            'silt': 'Silt',
            'clay': 'Clay',
            'soc': 'SOC',
            'bdod': 'Bdod'
        }
        df['feature'] = df['feature'].replace(feature_rename)

        desired_feature_order = ['Tavg', 'Prcp', 'ET0', 'GDD', 'Silt', 'Clay', 'Bdod', 'SOC']
        features = [f for f in desired_feature_order if f in df['feature'].unique()]
        targets = df['target'].unique()
        periods = ['all', 'history', '2040s', '2070s']
        right_labels = ["", "Historical", "2040s", "2070s"]

        # 设置画布和 GridSpec
        fig = plt.figure(figsize=(6.89, 6.89))
        gs = gridspec.GridSpec(nrows=4, ncols=2, width_ratios=[18, 1], wspace=0.01, hspace=0.3)

        # 色彩映射设置
        vmin, vmax = -1.1, 1.1
        cmap = sns.diverging_palette(220, 20, as_cmap=True)
        # 存储最后一个 ax，用于显示 x 轴标签
        last_ax = None

        # 循环绘图
        for i, (period, label) in enumerate(zip(periods, right_labels)):
            ax = fig.add_subplot(gs[i, 0], sharex=last_ax if last_ax else None)

            sub_df = df[df['period'] == period]
            pivot_df = sub_df.pivot(index='target', columns='feature', values='values')
            pivot_df = pivot_df.reindex(index=targets, columns=features)
            # colorbar 位置仅用于第一个图
            if i == 0:
                cax = fig.add_subplot(gs[i, 1])
            else:
                cax = None
            sns.heatmap(
                pivot_df,
                ax=ax,
                cbar=(i == 0),
                cbar_ax=cax if i == 0 else None,
                annot=True,
                fmt=".2f",
                cmap=cmap,
                center=0,
                vmin=vmin,
                vmax=vmax,
                linewidths=0.5,
                linecolor='grey',
                annot_kws={"size": 11}
            )
            if i == 0:
                cax.set_title("Coefficient", fontsize=12)
            if i == 0:
                ax.set_title("Average coefficients", fontsize=12, loc='center', pad=6)
            if i == 1:
                ax.set_title("Coefficients modified by simulation period", fontsize=12, loc='center', pad=6)

            # 设置标签可见性
            if i < len(periods) - 1:
                ax.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
            else:
                ax.set_xticklabels(ax.get_xticklabels(), ha='center', fontsize=11)
                last_ax = ax  # 最后一个 ax 显示 x 轴刻度

            ax.text(0, 1.19, f'({chr(96 + i + 1)})',  # chr(97) = 'a', i start from 0
                    transform=ax.transAxes, fontsize=12,
                    va='top', ha='left',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='none'),
                    zorder=12)
            ax.set_ylabel('')
            ax.set_xlabel('')
            ax.tick_params(axis='y', labelsize=9)
            # 添加右侧标签
            if label:
                ax.text(1.01, 0.5, label, va='center', ha='left', fontsize=12,
                        transform=ax.transAxes, rotation=0)
        # fig.subplots_adjust(wspace=0.2, hspace=0.5)
        # plt.tight_layout()
        plt.subplots_adjust(left=0.10, right=0.94, top=0.95, bottom=0.05)
        plt.savefig('../figs/paper_figs/fig6_standardized_coefficients_by_period.jpg', dpi=300)

    def fig_s1_cotton_production(self):
        df = pd.read_csv('../data/cotton_production.csv')
        df["Rate (%)"] = df["Rate"] * 100
        fig, ax1 = plt.subplots(figsize=(6.89, 3.5))
        fontsize = 11
        line1, = ax1.plot(df["Years"], df["China(MT)"], label="China cotton output", color="tab:blue", marker="o")
        line2, = ax1.plot(df["Years"], df["Xinjiang (MT)"], label="Xinjiang cotton output", color="tab:green", marker="s")
        # ax1.set_xlabel("Year", fontsize=fontsize)
        ax1.set_ylabel("Cotton production (10,000 metric tons)", fontsize=fontsize)
        ax1.tick_params(axis='y', labelcolor="black", labelsize=fontsize)
        ax1.tick_params(axis='x', labelsize=fontsize)
        ax1.xaxis.set_major_locator(MaxNLocator(integer=True))

        # right y scale
        ax2 = ax1.twinx()
        line3, = ax2.plot(df["Years"], df["Rate (%)"], label="Xinjiang share (%)", color="tab:red", linestyle='--', marker="^")
        ax2.set_ylabel("Xinjiang share (%)", fontsize=fontsize)
        ax2.tick_params(axis='y', labelcolor="black", labelsize=fontsize)

        # merge legends in lower right
        lines = [line1, line2, line3]
        labels = [line.get_label() for line in lines]
        ax1.legend(lines, labels, loc='lower right', fontsize=fontsize)
        ax1.grid(True)
        plt.tight_layout()
        # save figure
        outputfile = os.path.join(script_directory, '../figs/supplementary_figs/')
        if not os.path.exists(outputfile):
            os.makedirs(outputfile)
        fig.savefig(os.path.join(outputfile, 'figs1_cotton_production.jpg'), dpi=300, bbox_inches='tight')
        plt.close(fig)

    def fig_s2_weather_change(self):
        df = pd.read_csv(os.path.join(script_directory, '../results/xinjiang_weather10km/weather_stats_results_2000-2081.csv'))
        df['Years'] = df['Years'].astype(int)
        df_hist = df[df['SSP'] == 'Historical']
        df_245 = df[df['SSP'] == 'ssp245']
        df_585 = df[df['SSP'] == 'ssp585']

        fig, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
        var_names = [('Tavg', 'Average temperature (°C)'), ('Prcp', 'Precipitation (mm)')]
        colors = {'Historical': 'black', 'ssp245': 'royalblue', 'ssp585': 'firebrick'}
        labels = {'Historical': 'Historical', 'ssp245': 'SSP2-4.5', 'ssp585': 'SSP5-8.5'}

        for i, (var, ylabel) in enumerate(var_names):
            ax = axes[i]
            ax.plot(df_hist['Years'], df_hist[var], color=colors['Historical'], label=labels['Historical'], linewidth=2)
            ax.plot(df_245['Years'], df_245[var], color=colors['ssp245'], label=labels['ssp245'], linewidth=2)
            ax.plot(df_585['Years'], df_585[var], color=colors['ssp585'], label=labels['ssp585'], linewidth=2)
            ax.set_ylabel(ylabel, fontsize=12)
            ax.grid(True, linestyle='--', alpha=0.5)
            if i == 0:
                ax.legend(loc='upper left', fontsize=11)
        # axes[-1].set_xlabel('Year', fontsize=12)
        plt.tight_layout()
        outputfile = os.path.join(script_directory, '../figs/supplementary_figs/')
        if not os.path.exists(outputfile):
            os.makedirs(outputfile)
        fig.savefig(os.path.join(outputfile, 'figs2_weather_change_2000-2076.jpg'), dpi=300, bbox_inches='tight')
        plt.close(fig)

    def fig_s3_plot_correlation_analysis(self):
        df = pd.read_csv('../results/analysis10km/correlation/weather_soil_potential_return_correlation.csv')
        df.rename(columns={
            'mean_tmean': 'Tavg',
            'mean_prcp': 'Prcp',
            'mean_et': 'ET0',
            'mean_gdd': 'GDD',
            'sand': 'Sand',
            'silt': 'Silt',
            'clay': 'Clay',
            'soc': 'Soc',
            'bdod': 'Bdod',
            'potential_profit': 'Gain',
            'Seasonal irrigation_diff': 'ΔIrrigation',
            'Dry yield_diff': 'ΔYield',
            'iwp_diff': 'ΔIWP'
        }, inplace=True)
        # targets = ['Yield_SDI','Yield_CI', 'Irr_SDI','Irr_CI','IWP_SDI','IWP_CI',"ΔYield", "ΔIrrigation", "ΔIWP", "Gain"]
        targets = ["ΔYield", "ΔIrrigation", "ΔIWP", "Gain"]
        features = ["Tavg",  "Prcp", "ET0", "GDD","Sand", "Silt", "Clay", "Soc", "Bdod"]
        # features = ["Tavg",  "Prcp", "ET0", "GDD", "Silt", "Clay", "Soc", "Bdod"]
        # 合并特征和目标
        combined_df = df[features + targets].dropna()
        # # 绘制特征 + 目标变量的联合分布图
        sns.pairplot(combined_df[features + targets], diag_kind='kde', corner=True)
        plt.tight_layout()
        plt.savefig('../figs/correlation/pairplot_features_vs_target.jpg', dpi=300)

        # 计算相关性矩阵并可视化
        sns.set_theme()
        corr_matrix = combined_df[features].corr()
        plt.figure(figsize=(6.89, 6.89))
        sns.heatmap(
            corr_matrix,
            annot=True,
            fmt=".2f",
            cmap='vlag',
            center=0,
            annot_kws={"size": 10},
            cbar_kws={"shrink": 0.8}
        )
        plt.xticks(fontsize=11, rotation=45, ha='center')
        plt.yticks(fontsize=11)
        plt.tight_layout()
        plt.savefig('../figs/supplementary_figs/figs3_correlation_heatmap_features_vs_target.jpg', dpi=300)
   
    def fig_s4_simulation_results(self):
        # 图例配置
        PLOT_CONFIG = {
            'yield': {
                'cmap': 'Oranges',
                'cbar': {'label': 'Yield (tonne/ha)', 'vmin': 5.5, 'vmax': 8, 'step': 0.5}
            },
            'irr': {
                'cmap': 'Blues',
                'cbar': {'label': 'Irrigation (mm)', 'vmin': 200, 'vmax': 800, 'step': 200}
            },
            'iwp': {
                'cmap': 'Greens',
                'cbar': {'label': 'IWP (kg/m³)', 'vmin': 0.5, 'vmax': 2, 'step': 0.5}
            },
        }

        # period 和 metric 映射
        all_periods = ['history', '2040s_ssp245', '2040s_ssp585', '2070s_ssp245', '2070s_ssp585']
        metrics = [
            ('yield', 'Dry_yield_tonne_ha.tif'),
            ('irr', 'Seasonal_irrigation_mm.tif'),
            ('iwp', 'iwpkg_m3.tif'),
        ]
        
        def create_axis(fig, row, col):
            ax = fig.add_subplot(5, 3, row * 3 + col + 1, projection=ccrs.PlateCarree())
            ax.set_facecolor('none')
            return ax

        def plot_geodata(ax):
            XINJIANG.plot(ax=ax, transform=ccrs.PlateCarree(),
                        facecolor='none', edgecolor='black', lw=0.3, zorder=3, alpha=0.5)
            # XINJIANG_CITY.plot(ax=ax, transform=ccrs.PlateCarree(),
            #                 facecolor='none', edgecolor='gray', lw=0.3, zorder=3, alpha=0.5)
            XINJIANG_ZONES2.plot(ax=ax, transform=ccrs.PlateCarree(),
                        facecolor='none', edgecolor='black', lw=0.3, zorder=3, alpha=0.5)

        def create_colorbar(fig, ax_position, cmap, cbar_config, vmin, vmax, step):
            cax = fig.add_axes(ax_position)
            norm = Normalize(vmin=vmin, vmax=vmax)
            cbar = ColorbarBase(
                cax, cmap=cmap, norm=norm, extend='both', orientation='horizontal'
            )
            cbar.locator = MultipleLocator(base=step)
            cbar.update_ticks()
            cbar.ax.set_xlabel(cbar_config['label'], fontsize=8)
            cbar.ax.xaxis.set_label_position('top')       # 标签放上面
            cbar.ax.xaxis.set_ticks_position('top')        # 刻度放上面
            cbar.ax.tick_params(labelsize=6, labelrotation=0)  # 刻度水平

        def plot_raster_layer(ax, file_path, data_type):
            config = PLOT_CONFIG[data_type]
            vmin = config['cbar']['vmin']
            vmax = config['cbar']['vmax']
            step = config['cbar']['step']
            with rasterio.open(file_path) as src:
                data = src.read(1)
                bounds = src.bounds
            masked_data = np.ma.masked_invalid(data)
            ax.imshow(masked_data, origin='upper',transform=ccrs.PlateCarree(),
                        extent=[bounds.left, bounds.right, bounds.bottom, bounds.top],
                        cmap=config['cmap'], zorder=2,
                        norm=Normalize(vmin=vmin, vmax=vmax))
            ax.set_xticks([])  # 不显示x刻度
            ax.set_yticks([])  # 不显示y刻度
            ax.spines['geo'].set_visible(False)  # 移除地理图的边框

            # 在ax的右下角添加直方图
            hist_masked_data = np.ma.masked_outside(data, vmin, vmax)
            # hist_masked_data = masked_data
            hist_ax = ax.inset_axes([0.54, 0.20, 0.30, 0.30])  # [left, bottom, width, height]
            hist_ax.hist(hist_masked_data.compressed(), bins=50, color='gray', alpha=0.7)
            hist_ax.tick_params(axis='both', which='major', labelsize=6, colors='black')
            for spine in hist_ax.spines.values():
                spine.set_edgecolor('black')
                spine.set_linewidth(0.6)
            return vmin, vmax, step

        def plot_grid():
            for label in ['90-0-0','90-7-0']:
                fig = plt.figure(figsize=(6.39, 8.4))
                for row_idx, period in enumerate(all_periods):
                    for col_idx, (data_type, metric_filename) in enumerate(metrics):
                        file_path = f'../results/analysis10km/future/{label}/{period}/{metric_filename}'
                        ax = create_axis(fig, row_idx, col_idx)
                        vmin, vmax, step = plot_raster_layer(ax, file_path, data_type)
                        plot_geodata(ax)
                        # 在第一行添加 colorbar
                        if row_idx == 0:
                            cbar_x = 0.24 * col_idx + 0.19
                            create_colorbar(fig, [cbar_x, 0.88, 0.18, 0.01],
                                            PLOT_CONFIG[data_type]['cmap'],
                                            PLOT_CONFIG[data_type]['cbar'],
                                            vmin, vmax, step)
                    period_labels = ['History', '2040s(SSP2-4.5)', '2040s(SSP5-8.5)', '2070s(SSP2-4.5)', '2070s(SSP5-8.5)']
                    # # 在每一行左侧添加 period 标签
                    period_label = period_labels[row_idx]
                    fig.text(0.1, 0.78 - row_idx * 0.15, period_label, fontsize=8, rotation=90,
                            va='center', ha='left')
                fig.subplots_adjust(wspace=-0.2, hspace=-0.2)
                plt.savefig(f'../figs/supplementary_figs/figs4_simulation_results_grid_{label}.jpg', dpi=300, bbox_inches='tight')
        # 执行绘图函数
        plot_grid()

    def fig_test(self):
        file_path = '../results/analysis10km/future/baseline/iwp_diff.tif'
        with rasterio.open(file_path) as src:
            data = src.read(1)
            bounds = src.bounds
        fig, ax = plt.subplots(figsize=(6.89, 3.5))
        # data_masked = np.ma.masked_invalid(data)
        data_masked = np.ma.masked_outside(data, 0, 1)
        ax.hist(data_masked.compressed(), bins=50, color='gray', alpha=0.5)
        ax.grid(True, linestyle='--', alpha=0.3)
        ax.tick_params(axis='both', which='major', labelsize=4)
        plt.tight_layout()
        plt.show()

    def fig_xinjiang_weather(self):
        # Define helper function to plot raster data and save each figure
        def plot_raster(file_path, cmap, title, vmin=None, vmax=None, output_path=None):
            fig, ax = plt.subplots(figsize=(6, 6), subplot_kw={'projection': ccrs.PlateCarree()})
            with rasterio.open(file_path) as src:
                data = src.read(1)
                bounds = src.bounds
            data_mask = np.ma.masked_less_equal(data, 0)
            im = ax.imshow(data_mask, origin='upper', transform=ccrs.PlateCarree(),
                            extent=[bounds.left, bounds.right, bounds.bottom, bounds.top], cmap=cmap, zorder=2, vmin=vmin, vmax=vmax)
            xinjiang.plot(ax=ax, transform=ccrs.PlateCarree(), facecolor='none', edgecolor='black', lw=0.7, zorder=3, alpha=0.8)
            ax.set_title(title, fontsize=12)
            cbar = fig.colorbar(im, ax=ax, orientation='vertical', fraction=0.046, pad=0.04)
            cbar.ax.set_ylabel(title, fontsize=10)
            if output_path:
                plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close(fig)

        # Load geographic data
        xinjiang = gpd.read_file('../data/xinjiang_zones/xinjiang.shp')

        # Plot each TIF file and save each figure
        plot_raster('../results/xinjiang_weather10km/mean_et.tif', 'Blues', 'Mean ET', output_path='../figs/weather10km/xinjiang_weather_mean_et.png')
        plot_raster('../results/xinjiang_weather10km/mean_gdd.tif', 'Oranges', 'Mean GDD', output_path='../figs/weather10km/xinjiang_weather_mean_gdd.png')
        plot_raster('../results/xinjiang_weather10km/mean_prcp.tif', 'Greens', 'Mean Precipitation', output_path='../figs/weather10km/xinjiang_weather_mean_prcp.png')
        plot_raster('../results/xinjiang_weather10km/mean_tmax.tif', 'Reds', 'Mean Tmax', output_path='../figs/weather10km/xinjiang_weather_mean_tmax.png')
        plot_raster('../results/xinjiang_weather10km/mean_tmin.tif', 'Purples', 'Mean Tmin', output_path='../figs/weather10km/xinjiang_weather_mean_tmin.png')

    def fig_cotton_areas_clusted(self):
        # Load geographic data
        province = gpd.read_file('../data/base/省级/省级.shp')
        xinjiang = province[province['省'] == '新疆维吾尔自治区']
        file_path = '../results/cotton_area_clustered/Cluster.tif'
        
        fig, ax = plt.subplots(figsize=(6,6), subplot_kw={'projection': ccrs.PlateCarree()})
        with rasterio.open(file_path) as src:
            data = src.read(1)
            bounds = src.bounds
        data_mask = np.ma.masked_invalid(data)
        unique_values = np.unique(data_mask.compressed())  # Get unique values from the masked array
        num_categories = len(unique_values)  # Determine the number of unique categories
        cmap = plt.get_cmap('Set1', num_categories)  # Use a colormap with the number of unique categories
        im = ax.imshow(data_mask, origin='upper', transform=ccrs.PlateCarree(), 
                    extent=[bounds.left, bounds.right, bounds.bottom, bounds.top], cmap=cmap, zorder=2)
        xinjiang.plot(ax=ax, transform=ccrs.PlateCarree(), facecolor='none', edgecolor='black', lw=0.7, zorder=3, alpha=0.8)
        # Create a legend for the categorical values
        handles = [mpatches.Patch(color=cmap(i), label=f'Cluster {int(val)}') for i, val in enumerate(unique_values)]
        ax.legend(handles=handles, title='Clusters', loc='upper right')
        plt.savefig(os.path.join('../figs', 'cotton_areas_clusters.png'), dpi=300, bbox_inches='tight')
        plt.close(fig)


def main():
    paper_fig = PlotAllfigures()
    paper_fig.fig1_study_areas()
    # paper_fig.fig_sites_yield_irrigation_iwp()
    # paper_fig.fig2_calibration()
    # paper_fig.fig3_irrigation_threshoulds()
    # paper_fig.fig_baseline()
    # paper_fig.fig4_deficit_return()
    # paper_fig.fig5_future_results()
    # paper_fig.fig6_plot_standardized_coefficients()
    # paper_fig.fig_s1_cotton_production()
    # paper_fig.fig_s2_weather_change()
    # paper_fig.fig_s3_plot_correlation_analysis()
    # paper_fig.fig_s4_simulation_results()
    # paper_fig.fig_xinjiang_weather()
    # paper_fig.fig_cotton_areas_clusted()
    # paper_fig.fig_test()

if __name__ == "__main__":
    main()

    