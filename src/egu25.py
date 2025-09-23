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
from matplotlib.ticker import MaxNLocator

def calibration():
    plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 12,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10
})
    metrics_data = pd.read_csv(os.path.join(script_directory, '../results/calibration/metrics.csv'))
    cal_names = ['canopy_cover', 'biomass', 'yield_t_ha']
    display_names = ['Canopy cover (%)', 'Biomass (kg/ha)', 'Dry yield (tonne/ha)']

    fig, ax = plt.subplots(2, 3, figsize=(9, 6))  # 2 rows, 3 columns
    ax = ax.flatten()

    # 设置每个变量对应的y轴刻度
    ticks_dict = {
        'canopy_cover': np.arange(0, 101, 20),
        'biomass': np.arange(0, 20001, 5000),
        'Dry yield (tonne/ha)': np.arange(0, 9, 2),
    }

    for i, (label_name, display_name) in enumerate(zip(cal_names, display_names)):
        df = pd.read_csv(os.path.join(script_directory, f'../results/calibration/{label_name}.csv'))
        if label_name == 'yield_t_ha':
            label_name = 'Dry yield (tonne/ha)'
        # 读取metrics数据
        df_metrics = metrics_data[metrics_data['calibrated_names'] == label_name]

        for j, cal_label in enumerate(['cal', 'val']):
            plot_index = j * 3 + i  # First row: cal (j=0), second row: val (j=1)
            df_filtered = df[df['cal_label'] == cal_label]
            df_metrics_filtered = df_metrics[df_metrics['cal_label'] == cal_label]

            sns.set_style("white")
            sns.scatterplot(x='sim_values', y='obs_values', data=df_filtered, s=80, ax=ax[plot_index])

            if not df_metrics_filtered.empty:
                r2 = df_metrics_filtered['r2'].values[0]
                rmse = df_metrics_filtered['rmse'].values[0]
                d = df_metrics_filtered['d'].values[0]
                texts = f"R2: {r2:.2f}\nRMSE: {rmse:.2f}\nd: {d:.2f}"
                ax[plot_index].text(0.05, 0.95, texts, ha='left', va='top', transform=ax[plot_index].transAxes, color='black')

            max_range = max(df_filtered['sim_values'].max(), df_filtered['obs_values'].max())
            ax[plot_index].set_xlim(0, max_range)
            ax[plot_index].set_ylim(0, max_range)
            ax[plot_index].plot([0, max_range], [0, max_range], 'r--', linewidth=2)

            # 设置y轴刻度
            ax[plot_index].set_yticks(ticks_dict[label_name])
            ax[plot_index].set_xticks(ticks_dict[label_name])

            # 只在第一行（cal）显示标题
            if j == 0:
                ax[plot_index].set_title(display_name)

            # 设置统一的轴标签
            ax[plot_index].set_ylabel("Observation values")
            ax[plot_index].set_xlabel("Simulation values" if j == 1 else "")

    plt.tight_layout()
    outputfile = os.path.join(script_directory, '../egu25/')
    if not os.path.exists(outputfile):
        os.makedirs(outputfile)
    
    fig.savefig(os.path.join(outputfile, 'calibration_validation_metrics.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)


def cotton_production():
    df = pd.read_csv('../data/cotton_production.csv')

    # 如果Rate是小数形式，乘以100显示百分比
    df["Rate (%)"] = df["Rate"] * 100

    # 创建图形
    fig, ax1 = plt.subplots(figsize=(10, 5))

    # 设置统一字体大小
    fontsize = 14

    # 左边Y轴：产量
    line1, = ax1.plot(df["Years"], df["China(MT)"], label="China Cotton Output", color="tab:blue", marker="o")
    line2, = ax1.plot(df["Years"], df["Xinjiang (MT)"], label="Xinjiang Cotton Output", color="tab:green", marker="s")
    ax1.set_xlabel("Year", fontsize=fontsize)
    ax1.set_ylabel("Cotton Production (10,000 Metric Tons)", fontsize=fontsize)
    ax1.tick_params(axis='y', labelcolor="black", labelsize=fontsize)
    ax1.tick_params(axis='x', labelsize=fontsize)
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))

    # 右边Y轴：比例
    ax2 = ax1.twinx()
    line3, = ax2.plot(df["Years"], df["Rate (%)"], label="Xinjiang Share (%)", color="tab:red", linestyle='--', marker="^")
    ax2.set_ylabel("Xinjiang Share (%)", fontsize=fontsize)
    ax2.tick_params(axis='y', labelcolor="black", labelsize=fontsize)

    # 合并图例并放在右下角
    lines = [line1, line2, line3]
    labels = [line.get_label() for line in lines]
    ax1.legend(lines, labels, loc='lower right', fontsize=fontsize)

    # 标题
    plt.title("China vs Xinjiang Cotton Output and Share Over Years", fontsize=18, fontweight='bold')

    # 网格和布局
    ax1.grid(True)
    plt.tight_layout()

    # 保存图片
    script_directory = os.path.dirname(os.path.abspath(__file__))
    outputfile = os.path.join(script_directory, '../egu25/')
    if not os.path.exists(outputfile):
        os.makedirs(outputfile)
    fig.savefig(os.path.join(outputfile, 'cotton_production.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)

def cotton_pie():
    # 设置统一字体大小
    fontsize = 14
    df = pd.read_excel('../data/Global Cotton Production by Country in 2023 - Global Cotton Production by Country.xlsx')
        # 排序并提取前5个国家
    df_sorted = df.sort_values(by="Metric Tons", ascending=False)
    top5 = df_sorted.head(5)
    others = pd.DataFrame([{
        "Country": "Other",
        "Metric Tons": df_sorted.iloc[5:]["Metric Tons"].sum()
    }])


    # 合并成一个饼图数据框
    pie_df = pd.concat([top5, others], ignore_index=True)

    # 设置颜色：自定义更柔和的配色
    colors = ['#4C72B0', '#55A868', '#C44E52', '#8172B3', '#CCB974', '#64B5CD']
    explode = [0.05]*5 + [0.1]  # top5 微突出，Other 更突出

    # 自定义百分比标签函数（国家名 + 百分比）
    def make_autopct(values):
        def my_autopct(pct):
            total = sum(values)
            val = int(round(pct * total / 100.0))
            return f"{pct:.1f}%"
        return my_autopct

    # 绘图
    fig, ax = plt.subplots(figsize=(5, 5))
    wedges, texts, autotexts = ax.pie(
        pie_df["Metric Tons"],
        labels=pie_df["Country"],
        autopct=make_autopct(pie_df["Metric Tons"]),
        startangle=140,
        colors=colors,
        explode=explode,
        shadow=True,
        textprops={'fontsize': fontsize}
    )

    # 设置标题
    plt.title("Global Cotton Production Share (2023)", fontsize=18, fontweight='bold')
    plt.tight_layout()

    # 保存图像
    script_directory = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_directory, '../egu25/')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    fig.savefig(os.path.join(output_dir, 'cotton_pie_2023.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)



if __name__ == "__main__":
    calibration()
    # cotton_production()
    # cotton_pie()
