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

from aquacrop import AquaCropModel, Soil, InitialWaterContent
from aquacrop.utils import prepare_weather
from management_scenarios import ManagementScenarios
from crop_varieties import ZoneCropVarieties

# 配置路径与参数
SOIL_CSV_PATH = '../data/grid_10km/aquacrop_inputdata/soil/soil.csv'
WEATHER_BASE_DIR = '../data/grid_10km/aquacrop_inputdata/weather/2000-01-01_2022-12-31'
FUTURE_WEATHER_ROOT = '../data/grid_10km/aquacrop_inputdata/weather/2022-01-01_2081-12-31'

FUTURE_WEATHER_MODELS = {
    name: os.path.join(FUTURE_WEATHER_ROOT, name)
    for name in os.listdir(FUTURE_WEATHER_ROOT)
    if os.path.isdir(os.path.join(FUTURE_WEATHER_ROOT, name)) and name not in ['2000-01-01_2021-12-31']
}
MAX_WORKERS_PER_POOL = 61
TOTAL_POOLS = 5
MEMORY_THRESHOLD = 90

os.environ['DEVELOPMENT'] = 'True'
warnings.simplefilter(action='ignore', category=FutureWarning)


class GlobalData:
    _soil_df = None
    _weather_cache = {}
    _lock = multiprocessing.Lock()

    @classmethod
    def init_global_data(cls):
        with cls._lock:
            if cls._soil_df is None:
                if not os.path.exists(SOIL_CSV_PATH):
                    raise FileNotFoundError(f"土壤数据文件不存在: {SOIL_CSV_PATH}")
                cls._soil_df = pd.read_csv(SOIL_CSV_PATH)
                if cls._soil_df.empty:
                    raise ValueError("土壤数据为空")

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


class ZoneSimulator:
    def __init__(self, id, scenario_func, weather_base_dir):
        self.id = id
        self.soil = self._create_soil()
        self.weather = GlobalData.get_weather(id, weather_base_dir)
        self.varieties = ZoneCropVarieties(id).create_varieties_parameters()
        self.scenarios = getattr(ManagementScenarios(id), scenario_func)()

    def _create_soil(self):
        df = GlobalData.get_soil_df()
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


def generate_tasks(valid_ids, scenario_func, weather_base_dir):
    tasks = []
    for id in valid_ids:
        try:
            simulator = ZoneSimulator(id, scenario_func, weather_base_dir)
            for variety, params in simulator.varieties.items():
                for scenario in simulator.scenarios:
                    tasks.append((id, simulator.weather, simulator.soil, variety, params, scenario, scenario_func))
        except Exception as e:
            print(f"任务生成失败 ID {id}: {str(e)}")
            continue
    return tasks


def worker_process(task):
    id, weather, soil, variety, crop_params, scenario, scenario_func = task
    try:
        if '_future_' in scenario_func:
            start_year, end_year = 2023, 2080
        else:
            start_year, end_year = 2000, 2022

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
        return result.to_dict('records')
    except Exception as e:
        return {'error': f"ID {id} | Variety {variety} | Scenario {scenario[:3]}\n{str(e)}\n{traceback.format_exc()}"}


def memory_guard():
    while True:
        if psutil.virtual_memory().percent > MEMORY_THRESHOLD:
            os._exit(1)
        time.sleep(10)


def batch_processor(task_batch, result_collector, error_log):
    try:
        GlobalData.init_global_data()
    except Exception as e:
        error_log.append(f"进程 {os.getpid()} 初始化失败: {str(e)}")
        return

    with ProcessPoolExecutor(
        max_workers=MAX_WORKERS_PER_POOL,
        initializer=GlobalData.init_global_data
    ) as executor:
        futures = {executor.submit(worker_process, task): task for task in task_batch}

        with tqdm(total=len(futures), desc="批次进度(Pool)", leave=False) as pbar:
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
                            error_log.append(f"提取ID失败: {str(e)}\n数据样例: {df.head().to_dict()}")
                            continue
                        if id_value in result_collector:
                            result_collector[id_value].append(df)
                        else:
                            error_log.append(f"ID {id_value} 未在 result_collector 中预创建")
                except Exception as e:
                    error_log.append(f"Future异常: {str(e)}")
                finally:
                    pbar.update(1)


def save_results(collector, errors, output_dir):
    if errors:
        error_path = os.path.join(output_dir, "error_log.txt")
        with open(error_path, 'a', encoding='utf-8') as f:
            f.write('\n\n'.join(errors) + '\n')
        del errors[:]

    for id, data_list in collector.items():
        if data_list:
            combined_df = pd.concat([pd.DataFrame(df) for df in data_list], ignore_index=True)
            output_path = os.path.join(output_dir, f"{id}.csv")
            if os.path.exists(output_path):
                existing_df = pd.read_csv(output_path)
                combined_df = pd.concat([existing_df, combined_df], ignore_index=True)
            combined_df.to_csv(output_path, index=False)


def validate_result_file(id, output_dir):
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


def run_simulation(scenario_func, output_dir):
    multiprocessing.freeze_support()
    GlobalData.init_global_data()

    is_future = '_future_' in scenario_func
    weather_dirs = FUTURE_WEATHER_MODELS if is_future else {'historical': WEATHER_BASE_DIR}

    for model_name, weather_base_dir in weather_dirs.items():
        sub_output_dir = os.path.join(output_dir, model_name) if is_future else output_dir
        os.makedirs(sub_output_dir, exist_ok=True)

        id_range = range(0, 1337)
        # id_range = list(range(0, 5)) + list(range(1000, 1005))
        valid_ids = [id for id in id_range if not validate_result_file(id, sub_output_dir)]
        print(f"[{model_name}] 待处理ID数量: {len(valid_ids)}")

        manager = multiprocessing.Manager()
        result_collector = manager.dict()
        error_log = manager.list()

        for id in valid_ids:
            result_collector[id] = manager.list()
        tasks = generate_tasks(valid_ids, scenario_func, weather_base_dir)
        np.random.shuffle(tasks)
        # print(f"[{model_name}] 总任务数量: {len(tasks)}")
        mem_process = multiprocessing.Process(target=memory_guard)
        mem_process.daemon = True
        mem_process.start()

        def split_list(lst, n):
            k, m = divmod(len(lst), n)
            return [lst[i*k + min(i, m):(i+1)*k + min(i+1, m)] for i in range(n)]

        task_batches = split_list(tasks, TOTAL_POOLS)
        pool_processes = []
        for batch in task_batches:
            p = multiprocessing.Process(
                target=batch_processor,
                args=(batch, result_collector, error_log)
            )
            p.start()
            pool_processes.append(p)

        for p in pool_processes:
            p.join()
        save_results(result_collector, error_log, sub_output_dir)
        print(f"[{model_name}] 处理完成 | 成功: {sum(len(v) for v in result_collector.values())} | 失败: {len(error_log)}")


if __name__ == "__main__":
    # Baseline 
    # run_simulation(scenario_func='_creat_baseline_irrigate', output_dir='../results/simulation10km/baseline')
    # Deficit 
    # run_simulation(scenario_func='_creat_deficit_irrigate', output_dir='../results/simulation10km/deficit')
    # Expert    
    # run_simulation(scenario_func='_creat_expert_irrigate', output_dir='../results/simulation10km/expert')
    # Future 
    run_simulation(scenario_func='_creat_future_irrigate', output_dir='../results/simulation10km/future')





