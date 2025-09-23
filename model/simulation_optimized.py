import os
import pandas as pd
import time
import numpy as np
import multiprocessing
import traceback
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import psutil
import warnings
import pickle
import gc
from functools import lru_cache
import threading
from pathlib import Path

from aquacrop import AquaCropModel, Soil, InitialWaterContent
from aquacrop.utils import prepare_weather
from management_scenarios import ManagementScenarios
from crop_varieties import ZoneCropVarieties

# 配置路径与参数
SOIL_CSV_PATH = '../data/grid_10km/aquacrop_inputdata/soil/soil.csv'
WEATHER_BASE_DIR = '../data/grid_10km/aquacrop_inputdata/weather/2000-01-01_2021-12-31'
FUTURE_WEATHER_ROOT = '../data/grid_10km/aquacrop_inputdata/weather/2022-01-01_2076-12-31'

FUTURE_WEATHER_MODELS = {
    name: os.path.join(FUTURE_WEATHER_ROOT, name)
    for name in os.listdir(FUTURE_WEATHER_ROOT)
    if os.path.isdir(os.path.join(FUTURE_WEATHER_ROOT, name)) and name not in ['2000-01-01_2021-12-31']
}

# 针对256核CPU和512GB内存优化配置
MAX_WORKERS_PER_POOL = 64  # 4个池，每个64个worker = 256核
TOTAL_POOLS = 4
MEMORY_THRESHOLD = 95  # 充分利用512GB内存
CACHE_SIZE = 10000

os.environ['DEVELOPMENT'] = 'True'
warnings.simplefilter(action='ignore', category=FutureWarning)


class OptimizedGlobalData:
    """高度优化的全局数据管理，支持预加载和缓存"""
    _soil_df = None
    _weather_cache = {}
    _varieties_cache = {}
    _scenarios_cache = {}
    _lock = multiprocessing.Lock()
    _cache_file = '../cache/global_data.pkl'
    
    @classmethod
    def init_global_data(cls):
        """初始化全局数据，支持缓存持久化"""
        with cls._lock:
            if cls._soil_df is None:
                # 尝试从缓存加载
                if os.path.exists(cls._cache_file):
                    try:
                        with open(cls._cache_file, 'rb') as f:
                            cached_data = pickle.load(f)
                            cls._soil_df = cached_data.get('soil_df')
                            cls._weather_cache = cached_data.get('weather_cache', {})
                            cls._varieties_cache = cached_data.get('varieties_cache', {})
                            cls._scenarios_cache = cached_data.get('scenarios_cache', {})
                            print(f"从缓存加载数据: 土壤{len(cls._soil_df)}行, 天气{len(cls._weather_cache)}个")
                            return
                    except Exception as e:
                        print(f"缓存加载失败: {e}")
                
                # 加载土壤数据
                if not os.path.exists(SOIL_CSV_PATH):
                    raise FileNotFoundError(f"土壤数据文件不存在: {SOIL_CSV_PATH}")
                cls._soil_df = pd.read_csv(SOIL_CSV_PATH)
                if cls._soil_df.empty:
                    raise ValueError("土壤数据为空")
                
                # 预加载所有天气数据
                cls._preload_all_weather()
                
                # 保存到缓存
                cls._save_cache()

    @classmethod
    def _preload_all_weather(cls):
        """预加载所有天气数据到内存"""
        print("预加载天气数据...")
        weather_dirs = [WEATHER_BASE_DIR] + list(FUTURE_WEATHER_MODELS.values())
        
        for weather_dir in weather_dirs:
            if not os.path.exists(weather_dir):
                continue
            weather_files = [f for f in os.listdir(weather_dir) if f.endswith('.txt')]
            
            for weather_file in tqdm(weather_files, desc=f"加载 {os.path.basename(weather_dir)}"):
                try:
                    id = int(weather_file.replace('.txt', ''))
                    path = os.path.join(weather_dir, weather_file)
                    key = f"{weather_dir}_{id}"
                    if key not in cls._weather_cache:
                        cls._weather_cache[key] = prepare_weather(path)
                except Exception as e:
                    print(f"加载天气文件失败 {weather_file}: {e}")

    @classmethod
    def _save_cache(cls):
        """保存数据到缓存文件"""
        os.makedirs(os.path.dirname(cls._cache_file), exist_ok=True)
        cache_data = {
            'soil_df': cls._soil_df,
            'weather_cache': cls._weather_cache,
            'varieties_cache': cls._varieties_cache,
            'scenarios_cache': cls._scenarios_cache
        }
        with open(cls._cache_file, 'wb') as f:
            pickle.dump(cache_data, f)
        print(f"数据已缓存到: {cls._cache_file}")

    @classmethod
    def get_soil_df(cls):
        if cls._soil_df is None:
            cls.init_global_data()
        return cls._soil_df

    @classmethod
    def get_weather(cls, id, weather_base_dir):
        key = f"{weather_base_dir}_{id}"
        if key not in cls._weather_cache:
            path = os.path.join(weather_base_dir, f'{id}.txt')
            cls._weather_cache[key] = prepare_weather(path)
        return cls._weather_cache[key]

    @classmethod
    def get_varieties(cls, id):
        key = f"varieties_{id}"
        if key not in cls._varieties_cache:
            zone_varieties = ZoneCropVarieties(id)
            cls._varieties_cache[key] = zone_varieties.create_varieties_parameters()
        return cls._varieties_cache[key]

    @classmethod
    def get_scenarios(cls, id, scenario_func):
        key = f"scenarios_{id}_{scenario_func}"
        if key not in cls._scenarios_cache:
            management_scenarios = ManagementScenarios(id)
            cls._scenarios_cache[key] = getattr(management_scenarios, scenario_func)()
        return cls._scenarios_cache[key]


class OptimizedZoneSimulator:
    """优化的区域模拟器"""
    
    def __init__(self, id, scenario_func, weather_base_dir):
        self.id = id
        self.soil = self._create_soil()
        self.weather = OptimizedGlobalData.get_weather(id, weather_base_dir)
        self.varieties = OptimizedGlobalData.get_varieties(id)
        self.scenarios = OptimizedGlobalData.get_scenarios(id, scenario_func)

    def _create_soil(self):
        df = OptimizedGlobalData.get_soil_df()
        tempt_soil = df[df.ID == self.id].copy()
        depth_order = ['0-5cm', '5-15cm', '15-30cm', '30-60cm', '60-100cm', '100-200cm']
        tempt_soil['depth'] = pd.Categorical(tempt_soil['depth'], categories=depth_order, ordered=True)
        tempt_soil = tempt_soil.sort_values('depth').reset_index(drop=True)

        soil = Soil('custom', dz=[0.1] * 10)
        layer_thickness = [0.05, 0.1, 0.15, 0.3, 0.4, 1]
        for i in range(len(layer_thickness)):
            soil.add_layer_from_texture(
                thickness=layer_thickness[i],
                Sand=tempt_soil["sand"].iloc[i],
                Clay=tempt_soil["clay"].iloc[i],
                OrgMat=tempt_soil["soc"].iloc[i],
                penetrability=100
            )
        return soil


def generate_optimized_tasks(valid_ids, scenario_func, weather_base_dir):
    """生成优化后的任务列表"""
    tasks = []
    for id in valid_ids:
        try:
            simulator = OptimizedZoneSimulator(id, scenario_func, weather_base_dir)
            for variety, params in simulator.varieties.items():
                for scenario in simulator.scenarios:
                    tasks.append((id, simulator.weather, simulator.soil, variety, params, scenario, scenario_func))
        except Exception as e:
            print(f"任务生成失败 ID {id}: {str(e)}")
            continue
    return tasks


def optimized_worker_process(task):
    """优化的worker进程"""
    id, weather, soil, variety, crop_params, scenario, scenario_func = task
    try:
        if '_future_' in scenario_func:
            start_year, end_year = 2022, 2070
        else:
            start_year, end_year = 2000, 2020

        model = AquaCropModel(
            sim_start_time=f'{start_year}/{crop_params.planting_date}',
            sim_end_time=f'{end_year}/12/30',
            weather_df=weather,
            soil=soil,
            crop=crop_params,
            initial_water_content=InitialWaterContent(method='Depth', depth_layer=[1], value=['FC']),
            field_management=scenario[-2],
            irrigation_management=scenario[-1],
        )
        
        model.run_model(till_termination=True)
        result = model.get_simulation_results()
        result['ID'] = id
        result['variety'] = variety
        result['scenario_params'] = '-'.join(map(str, scenario[:3]))
        
        # 强制垃圾回收
        del model
        gc.collect()
        
        return result.to_dict('records')
    except Exception as e:
        return {'error': f"ID {id} | Variety {variety} | Scenario {scenario[:3]}\n{str(e)}\n{traceback.format_exc()}"}


def memory_monitor():
    """内存监控器"""
    while True:
        memory_percent = psutil.virtual_memory().percent
        if memory_percent > MEMORY_THRESHOLD:
            print(f"内存使用率过高: {memory_percent}%，强制垃圾回收")
            gc.collect()
            if memory_percent > MEMORY_THRESHOLD + 5:
                print("内存使用率持续过高，退出进程")
                os._exit(1)
        time.sleep(30)


def optimized_batch_processor(task_batch, result_collector, error_log, batch_id):
    """优化的批处理器"""
    try:
        OptimizedGlobalData.init_global_data()
    except Exception as e:
        error_log.append(f"进程 {os.getpid()} 初始化失败: {str(e)}")
        return

    with ProcessPoolExecutor(
        max_workers=MAX_WORKERS_PER_POOL,
        initializer=OptimizedGlobalData.init_global_data
    ) as executor:
        futures = {executor.submit(optimized_worker_process, task): task for task in task_batch}

        with tqdm(total=len(futures), desc=f"批次{batch_id}进度", leave=False) as pbar:
            for future in as_completed(futures):
                try:
                    result = future.result()
                    if isinstance(result, dict) and 'error' in result:
                        error_log.append(result['error'])
                    else:
                        df = pd.DataFrame(result)
                        try:
                            id_value = df['ID'].iloc[0]
                        except Exception as e:
                            error_log.append(f"提取ID失败: {str(e)}")
                            continue
                        if id_value in result_collector:
                            result_collector[id_value].append(df)
                except Exception as e:
                    error_log.append(f"Future异常: {str(e)}")
                finally:
                    pbar.update(1)


def optimized_save_results(collector, errors, output_dir):
    """优化的结果保存"""
    if errors:
        error_path = os.path.join(output_dir, "error_log.txt")
        with open(error_path, 'a', encoding='utf-8') as f:
            f.write('\n\n'.join(errors) + '\n')
        del errors[:]

    for id, data_list in collector.items():
        if data_list:
            try:
                combined_df = pd.concat([pd.DataFrame(df) for df in data_list], ignore_index=True)
                output_path = os.path.join(output_dir, f"{id}.csv")
                
                if os.path.exists(output_path):
                    existing_df = pd.read_csv(output_path)
                    combined_df = pd.concat([existing_df, combined_df], ignore_index=True)
                
                combined_df.to_csv(output_path, index=False)
                del combined_df
                gc.collect()
            except Exception as e:
                print(f"保存ID {id}结果失败: {e}")


def validate_result_file_optimized(id, output_dir):
    """优化的结果文件验证"""
    path = os.path.join(output_dir, f"{id}.csv")
    if not os.path.isfile(path):
        return False
    try:
        if os.path.getsize(path) < 100:
            os.remove(path)
            return False
        pd.read_csv(path, nrows=1)
        return True
    except Exception:
        try:
            os.remove(path)
        except Exception:
            pass
        return False


def run_optimized_simulation(scenario_func, output_dir):
    """运行优化的模拟"""
    multiprocessing.freeze_support()
    OptimizedGlobalData.init_global_data()

    is_future = '_future_' in scenario_func
    weather_dirs = FUTURE_WEATHER_MODELS if is_future else {'historical': WEATHER_BASE_DIR}

    for model_name, weather_base_dir in weather_dirs.items():
        sub_output_dir = os.path.join(output_dir, model_name) if is_future else output_dir
        os.makedirs(sub_output_dir, exist_ok=True)
        
        id_range = range(0, 1337)
        valid_ids = [id for id in id_range if not validate_result_file_optimized(id, sub_output_dir)]
        print(f"[{model_name}] 待处理ID数量: {len(valid_ids)}")

        if not valid_ids:
            print(f"[{model_name}] 所有ID已完成，跳过")
            continue

        manager = multiprocessing.Manager()
        result_collector = manager.dict()
        error_log = manager.list()

        for id in valid_ids:
            result_collector[id] = manager.list()
        
        tasks = generate_optimized_tasks(valid_ids, scenario_func, weather_base_dir)
        np.random.shuffle(tasks)
        print(f"[{model_name}] 总任务数量: {len(tasks)}")

        def split_list_optimized(lst, n):
            k, m = divmod(len(lst), n)
            return [lst[i*k + min(i, m):(i+1)*k + min(i+1, m)] for i in range(n)]

        task_batches = split_list_optimized(tasks, TOTAL_POOLS)
        pool_processes = []
        
        for i, batch in enumerate(task_batches):
            p = multiprocessing.Process(
                target=optimized_batch_processor,
                args=(batch, result_collector, error_log, i+1)
            )
            p.start()
            pool_processes.append(p)

        for p in pool_processes:
            p.join()
        
        optimized_save_results(result_collector, error_log, sub_output_dir)
        print(f"[{model_name}] 处理完成 | 成功: {sum(len(v) for v in result_collector.values())} | 失败: {len(error_log)}")


def convent_simulate_optimized():
    print('convent_simulate_optimized')
    run_optimized_simulation(scenario_func='_creat_baseline_irrigate', output_dir='../results/simulation10km/baseline')
    
def deficit_simulate_optimized():
     print('deficit_simulate_optimized')
     run_optimized_simulation(scenario_func='_creat_deficit_irrigate', output_dir='../results/simulation10km/deficit')

def expert_simulate_optimized():
    print('expert_simulate_optimized')
    run_optimized_simulation(scenario_func='_creat_expert_irrigate', output_dir='../results/simulation10km/expert')

def future_simulate_optimized():
    print('future_simulate_optimized')
    run_optimized_simulation(scenario_func='_creat_future_irrigate', output_dir='../results/simulation10km/future')

def main_optimized():
    convent_simulate_optimized()
    deficit_simulate_optimized()
    expert_simulate_optimized()
    future_simulate_optimized()
    
if __name__ == "__main__":
    main_optimized() 