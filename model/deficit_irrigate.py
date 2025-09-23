import os
import sys
import pandas as pd
import numpy as np
from scipy.optimize import minimize
import hashlib
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
from tqdm import tqdm
import warnings

# 禁用AquaCrop的日志和警告
warnings.filterwarnings('ignore')

# 路径设置
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.extend([project_root, os.path.join(project_root, 'src')])

# 导入其他模块
from src.calibrate_validate.load_data_for_calibration import CalibrationData
from src.calibrate_validate.prepare_aquacrop_input import *
from crop_varieties import ZoneCropVarieties

class SiteDeficIrrigateOptimize():
    def __init__(self, id, load_data):
        self.ID = id
        self.load_data = load_data
        self.baseline_yield = None
        self.baseline_irrigation = None
        self._get_baseline_results()

    def _get_baseline_results(self):
        """获取基准产量和灌溉量"""
        weather = pre_weather(self.ID, self.load_data.weather_file_paths)
        soil = pre_soil(self.ID, self.load_data.soil_data)
        varieties = self.load_data.optimized_crop_parameters[
            (self.load_data.optimized_crop_parameters.ID == self.ID)].varieties.unique()
        
        output = []
        for variety in varieties:
            crop_params = ZoneCropVarieties(self.ID).create_aquacrop_parameters(variety)
            model = AquaCropModel(
                sim_start_time=f'2000/{crop_params.planting_date}',
                sim_end_time=f'2020/12/30',
                weather_df=weather,
                soil=soil,
                crop=crop_params,
                initial_water_content=InitialWaterContent(method='Depth', depth_layer=[1], value=['FC']),
                field_management=FieldMngt(mulches=True, mulch_pct=90, f_mulch=0.9),
                irrigation_management=IrrigationManagement(irrigation_method=2, IrrInterval=14, depth=200),
            )
            model.run_model(till_termination=True)
            output.append(model.get_simulation_results())
        
        df_output = pd.concat(output)
        self.baseline_yield = df_output['Dry yield (tonne/ha)'].mean().round(2)
        self.baseline_irrigation = df_output['Seasonal irrigation (mm)'].mean().round(2)

    def _run_single_optimization(self, x0, max_irr_season):
        """执行单个优化任务"""
        bounds = [(30, 100), (70, 100), (70, 100), (30, 100)]
        res = minimize(
            fun=self._evaluate,
            x0=x0,
            args=(max_irr_season,),
            method="SLSQP",
            bounds=bounds,
            constraints={
                "type": "ineq",
                "fun": self._constraint_func,
                "args": (max_irr_season,)
            },
            options={"maxiter": 5000}
        )
        
        if res.success:
            current_smts = np.clip(res.x.round().astype(int), 30, 100)
            current_efficiency = -res.fun
            return current_smts, current_efficiency
        return None, -np.inf

    def _evaluate(self, smts, max_irr_season):
        """评估函数（保持类内部）"""
        smts = np.round(smts).astype(int)
        out = self._run_defic_model(smts, max_irr_season)
        yld = out['Dry yield (tonne/ha)'].mean()
        tirr = out['Seasonal irrigation (mm)'].mean()
        efficiency = yld / tirr if tirr > 0 else -np.inf
        return -efficiency

    def _constraint_func(self, smts, max_irr_season):
        """约束函数（保持类内部）"""
        smts = np.round(smts).astype(int)
        out = self._run_defic_model(smts, max_irr_season)
        yld = out['Dry yield (tonne/ha)'].mean()
        obs = np.abs((yld - self.baseline_yield)) / self.baseline_yield
        return 0.1 - obs

    def _run_defic_model(self, smts, max_irr_season):
        """运行亏缺灌溉模型"""
        weather = pre_weather(self.ID, self.load_data.weather_file_paths)
        soil = pre_soil(self.ID, self.load_data.soil_data)
        varieties = self.load_data.optimized_crop_parameters[
            (self.load_data.optimized_crop_parameters.ID == self.ID)].varieties.unique()
        
        output = []
        for variety in varieties:
            crop_params = ZoneCropVarieties(self.ID).create_aquacrop_parameters(variety)
            model = AquaCropModel(
                sim_start_time=f'2000/{crop_params.planting_date}',
                sim_end_time=f'2020/12/30',
                weather_df=weather,
                soil=soil,
                crop=crop_params,
                initial_water_content=InitialWaterContent(method='Depth', depth_layer=[1], value=['FC']),
                field_management=FieldMngt(mulches=True, mulch_pct=90, f_mulch=0.9),
                irrigation_management=IrrigationManagement(
                    irrigation_method=1,
                    SMT=smts,
                    MaxIrrSeason=max_irr_season
                )
            )
            model.run_model(till_termination=True)
            output.append(model.get_simulation_results())
        return pd.concat(output)

def generate_initial_points(id, num_searches):
    """为指定ID生成初始点集合"""
    base_seed = int(hashlib.sha256(id.encode()).hexdigest()[:8], 16) % (2**32)
    np.random.seed(base_seed)
    x0_list = []
    for i in range(num_searches):
        search_seed = base_seed + i
        np.random.seed(search_seed)
        x0 = np.array([
        np.random.randint(30, 101),
        np.random.randint(70, 101),
        np.random.randint(70, 101),
        np.random.randint(30, 101)
        ])
        x0_list.append(x0)
    return x0_list

def process_single_task(id, x0, max_irr_season):
    """处理单个优化任务"""
    try:
        # 每个进程独立初始化
        np.random.seed(os.getpid())
        load_data = CalibrationData(use_initial_crop_parameters=True)
        optimizer = SiteDeficIrrigateOptimize(id, load_data)
        smts, efficiency = optimizer._run_single_optimization(x0, max_irr_season)
        return {
            'ID': id,
            'x0': x0.tolist(),
            'OptimizedSMTs': smts.tolist() if smts is not None else None,
            'Efficiency': efficiency,
            'BaselineYield': optimizer.baseline_yield,
            'BaselineIrrigation': optimizer.baseline_irrigation
        }
    except Exception as e:
        print(f"ID {id} 优化失败: {str(e)}")
        return {
            'ID': id,
            'x0': x0.tolist(),
            'OptimizedSMTs': None,
            'Efficiency': -np.inf,
            'BaselineYield': None,
            'BaselineIrrigation': None
        }

def main():
    # 参数配置
    MAX_IRR_SEASON = 300
    NUM_SEARCHES = 8  # 每个ID的初始点数量
    MAX_WORKERS = 8  # 保留部分核心给系统
    
    # 加载实验站点
    experiment_sites = pd.read_csv('../data/xinjiang_zones/experimental_sites.csv')
    ids = experiment_sites.ID.unique()[3:4]  # 取前10个ID
    
    # 生成所有任务
    tasks = []
    for id in ids:
        initial_points = generate_initial_points(id, NUM_SEARCHES)
        tasks.extend([(id, x0) for x0 in initial_points])
    
    # 执行并行计算
    results = []
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(process_single_task, id, x0, MAX_IRR_SEASON) for id, x0 in tasks]
        
        # 使用tqdm显示进度条
        for future in tqdm(as_completed(futures), total=len(futures), desc="全局优化进度"):
            results.append(future.result())
    
    # 处理结果
    results_df = pd.DataFrame(results)
    print(results_df)
    best_results = results_df.loc[results_df.groupby('ID')['Efficiency'].idxmax()]
    
    # 展开优化参数
    smt_cols = ['SMT1', 'SMT2', 'SMT3', 'SMT4']
    best_results[smt_cols] = pd.DataFrame(
        best_results['OptimizedSMTs'].apply(
            lambda x: x if x else [np.nan]*4).tolist(),
        index=best_results.index
    )
    
    # 保存结果
    output_path = '../results/simulation/optimized_smts_results_300mm_test2.csv'
    best_results.drop(columns=['x0', 'OptimizedSMTs']).to_csv(output_path, index=False)
    print(f"优化完成！结果已保存至 {output_path}")

if __name__ == '__main__':
    main()