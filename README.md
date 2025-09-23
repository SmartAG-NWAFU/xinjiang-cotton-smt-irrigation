# OptimizedCottonIrrigation

Optimizing cotton irrigation strategies in Xinjiang with process-based crop modeling (AquaCrop) and data pipelines for weather/soil preparation, model calibration/validation, and scenario simulations.

本仓库公开展示论文相关的核心代码与流程，以便审稿与复现。包含数据准备、模型参数初始化与校准、情景模拟（基线/亏缺/专家/未来气候）以及结果汇总绘图等模块。

## Repository Structure

- `src/`
  - `download_data/`: 数据下载与提取（如 GEE 提取 DEM/土壤/天气示例脚本）
  - `data_prepare_and_explore/`: 天气/土壤/观测等数据整理，特征统计与检查
  - `cotton_units/`: 研究区单元格/分区标注与构建
  - `cotton_zone_weather/`: 分区天气、聚类与相关性分析
  - 其他：图件绘制工具、用于稿件复现的脚本与笔记本
- `model/`
  - `simulation*.py`: 基线/亏缺/专家/未来等场景模拟主程序
  - `management_scenarios.py`: 管理情景（播期/灌溉策略等）生成与组合
  - `crop_varieties.py`: 分区/品种参数（由校准结果生成/读取）
  - `results_analysis.py`: 模拟结果汇总与统计
  - `main.py`: 示例入口（优化阈值与亏缺模拟）
- `utils/`: 常用工具（栅格/CSV 转换、地图绘图等）
- `requirements.txt`: Python 依赖
- `gee.yaml`: GEE 相关配置示例
- `xinjiangcotton.yml`: 项目级别配置（示例）

说明：为保持简洁，未纳入大体积的图件与稿件文档（figs/、paper/ 等）；需用时可在本地生成或单独提供。

## Environment Setup

- Python 3.10/3.11
- 推荐新环境安装依赖：
  ```bash
  python -m venv .venv && source .venv/bin/activate
  pip install -r requirements.txt
  ```
- 主要外部依赖：
  - `aquacrop`（Python 版 AquaCrop 模型）
  - 若使用 `src/download_data/gee_extractor.py` 等脚本，需要可用的 Google Earth Engine 账号、项目与认证

## Data Requirements & Layout

模型脚本默认从相对路径读取输入数据（可在代码内修改）：

- 天气与土壤数据（示例，见 `model/simulation.py` 顶部常量）：
  - `../data/grid_10km/aquacrop_inputdata/soil/soil.csv`
  - `../data/grid_10km/aquacrop_inputdata/weather/2000-01-01_2022-12-31/`
  - `../data/grid_10km/aquacrop_inputdata/weather/2022-01-01_2081-12-31/<GCM>/`
- 若目录不同，请在对应脚本中修改常量路径（如 `SOIL_CSV_PATH`、`WEATHER_BASE_DIR`、`FUTURE_WEATHER_ROOT`）。

## Quick Start

1) 准备环境并安装依赖：见上文。
2) 放置或链接所需的土壤与天气数据到预期目录结构。
3) 运行示例（亏缺阈值优化 + 亏缺模拟）：
   ```bash
   cd model
   python main.py
   ```
   - 或按需直接调用：
     ```bash
     # 历史/基线/亏缺/专家/未来气候模拟（参考 simulation.py 末尾示例）
     python simulation.py
     ```

## Typical Workflow

- 数据下载与提取（可选）：`src/download_data/`（如 GEE）
- 数据准备与特征统计：`src/data_prepare_and_explore/`
- 参数初始化与校准：`src/calibrate_validate/`
- 情景设置：`model/management_scenarios.py`
- 模拟执行：`model/simulation*.py`
- 结果汇总与可视化：`model/results_analysis.py`、`src/plot_all_figures.py`、`utils/plot_map_tools.py`

## Notes

- 运行前请确认数据路径与年份设置正确（历史/未来情景）。
- 并行参数（进程/内存阈值等）在 `model/simulation.py` 顶部可调整。
- 代码中示例路径基于内部数据布局，公共仓库不包含大体积原始/中间数据。

## Acknowledgements

- AquaCrop Python implementation (modeling core)
- Google Earth Engine (data acquisition when applicable)

## Citation

如您在研究中使用了本仓库的代码或流程，请引用本文相关工作（待论文公开后补充正式引用）。

