import os
import sys
import pandas as pd
import numpy as np
from scipy.optimize import minimize
import hashlib
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import logging
import time
import psutil

# setpath for importing modules
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.extend([project_root, os.path.join(project_root, 'src')])

START_YEAT = 2000

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('optimization.log'),
        logging.StreamHandler()
    ]
)

def log_system_resources():
    """Log current system resource usage"""
    cpu_percent = psutil.cpu_percent(interval=1)
    memory = psutil.virtual_memory()
    logging.info(f"CPU Usage: {cpu_percent}%")
    logging.info(f"Memory Usage: {memory.percent}% ({memory.used / 1024**3:.1f}GB / {memory.total / 1024**3:.1f}GB)")

from src.calibrate_validate.load_data_for_calibration import CalibrationData
from src.calibrate_validate.prepare_aquacrop_input import *
from crop_varieties import ZoneCropVarieties
import tqdm

IRRENTTERVAL = 7  # irrigation interval in days

class HighPerformanceConfig:
    """Configuration for high-performance server optimization"""
    MAX_WORKERS_PER_SITE = 61 # Maximum workers for site-level optimization
    MAX_SITE_WORKERS = 3      # Maximum parallel site processing
    CACHE_SIZE = 10000          # Cache size for 512GB RAM
    MEMORY_THRESHOLD = 90    # Memory usage threshold (%)
    CPU_THRESHOLD = 90        # CPU usage threshold (%)
    
    @staticmethod
    def should_reduce_workers():
        """Check if we should reduce workers due to high resource usage"""
        memory = psutil.virtual_memory()
        cpu_percent = psutil.cpu_percent(interval=1)
        
        if memory.percent > HighPerformanceConfig.MEMORY_THRESHOLD:
            logging.warning(f"High memory usage ({memory.percent}%), consider reducing workers")
            return True
        if cpu_percent > HighPerformanceConfig.CPU_THRESHOLD:
            logging.warning(f"High CPU usage ({cpu_percent}%), consider reducing workers")
            return True
        return False

class SiteDeficIrrigateOptimize():
    def __init__(self, id, load_data, run_baseline=True):  
        self.ID = id
        self.load_data = load_data
        self.baseline_yield = None
        self.baseline_irrigtion = None
        self._cache = {}  # Add cache for model results
        self._max_cache_size = HighPerformanceConfig.CACHE_SIZE  # Use config for cache size
        if run_baseline:  
            self._get_baseline_results()

    def _get_baseline_results(self):

        def run_baseline_model(crop_params, weather, soil):
            # run model with baseline parameters
            model = AquaCropModel(
                sim_start_time=f'{START_YEAT}/{crop_params.planting_date}',
                sim_end_time=f'2022/12/30',
                weather_df=weather,
                soil=soil,
                crop=crop_params,
                initial_water_content=InitialWaterContent(method='Depth', depth_layer=[1], value=['FC']),
                field_management=FieldMngt(mulches=True, mulch_pct=90, f_mulch=0.9),
                irrigation_management=IrrigationManagement(irrigation_method=2, IrrInterval=IRRENTTERVAL, AppEff=75, MaxIrr=50),
            )
            model.run_model(till_termination=True)
            return model.get_simulation_results()
        
        id_varieties = ZoneCropVarieties(self.ID)
        varieties = id_varieties._lookup_site_varieties()
        weather = pre_weather(self.ID, self.load_data.weather_file_paths)
        soil = pre_soil(self.ID, self.load_data.soil_data)
        output = []
        for variety in varieties:
            variety_parameters = id_varieties.create_aquacrop_parameters(variety)
            result = run_baseline_model(variety_parameters, weather, soil)
            output.append(result)
        df_output = pd.concat(output)
        self.baseline_yield = df_output['Dry yield (tonne/ha)'].mean().round(2)
        self.baseline_irrigtion = df_output['Seasonal irrigation (mm)'].mean().round(2)
         
    def _run_defic_model(self, smts):
        """
        funciton to run model and return results for given set of soil moisture targets
        """
        # Create cache key
        cache_key = tuple(smts)
        
        # Check cache first
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        id_varieties = ZoneCropVarieties(self.ID)
        varieties = id_varieties._lookup_site_varieties()
        weather = pre_weather(self.ID, self.load_data.weather_file_paths)
        soil = pre_soil(self.ID, self.load_data.soil_data)
        output = []
        for variety in varieties:
            variety_parameters = id_varieties.create_aquacrop_parameters(variety)
            model = AquaCropModel(
            sim_start_time=f'{START_YEAT}/{variety_parameters.planting_date}',
            sim_end_time=f'2022/12/30',
            weather_df=weather,
            soil=soil,
            crop=variety_parameters,
            initial_water_content=InitialWaterContent(method='Depth', depth_layer=[1], value=['FC']),
            field_management=FieldMngt(mulches=True, mulch_pct=90, f_mulch=0.9),
            irrigation_management=IrrigationManagement(irrigation_method=1, SMT=smts, MaxIrr=50, AppEff=100) # define irrigation management, appeffect=100
        )
            model.run_model(till_termination=True)
            output.append(model.get_simulation_results())
        result = pd.concat(output)
        
        # Cache the result
        self._cache[cache_key] = result
        
        # Limit cache size
        if len(self._cache) > self._max_cache_size:
            # Remove oldest entries (simple FIFO)
            oldest_keys = list(self._cache.keys())[:len(self._cache) - self._max_cache_size]
            for key in oldest_keys:
                del self._cache[key]
        
        return result

    def _evaluate_with_constraint(self, smts):
        """Combined evaluation function that returns both objective and constraint values"""
        smts = np.round(smts).astype(int)
        out = self._run_defic_model(smts)
        yld = out['Dry yield (tonne/ha)'].mean() 
        tirr = out['Seasonal irrigation (mm)'].mean()  
        iwp = (yld * 1000) / (tirr * 10) if tirr > 0 else -np.inf # convert to kg/ha, convert to m3/ha   
        reduction_ratio = (self.baseline_yield - yld) / self.baseline_yield # yield reduction ratio
        constraint = 0.1 - reduction_ratio # constraint to ensure yield reduction does not exceed 10% from baseline yield
        return -round(iwp, 2), constraint  # return negative iwp for minimization

    def _evaluate(self, smts):
        """convert smts to integers and run the model to get yield and irrigation"""
        obj_val, _ = self._evaluate_with_constraint(smts)
        return obj_val

    def _constraint_obs(self, smts):
        """constraint to ensure yield reduction does not exceed 10% from baseline yield"""
        _, constraint_val = self._evaluate_with_constraint(smts)
        return constraint_val

    def _optimize(self, num_searches):
        """optimization function, using parallel processing for multiple initial values"""
        best_smts = None
        best_efficiency = -np.inf
        no_improvement_count = 0
        max_no_improvement = 500 # Early stopping if no improvement for 500 iterations
        
        # create a base seed based on the ID to ensure reproducibility
        base_seed = int(hashlib.sha256(self.ID.encode()).hexdigest()[:8], 16) % (2**32)
        np.random.seed(base_seed)
        x0_list = []
        for i in range(num_searches):
            search_seed = base_seed + i
            np.random.seed(search_seed)
            x0 = np.array([
            np.random.randint(55, 101),
            np.random.randint(60, 101),
            np.random.randint(70, 101),
            np.random.randint(55, 101)
            ])
            x0_list.append(x0)

        # prepare task arguments for parallel execution
        bounds = [(55, 101), (60, 101), (70, 101), (55, 101)]
        task_args = [
            (x0, self.ID, self.load_data.__dict__,  # pass load_data parameters
             self.baseline_yield, self.baseline_irrigtion,
             bounds)
            for x0 in x0_list
        ]
        # Optimize for high-performance server (256 cores, 512GB RAM)
        max_workers = min(61, num_searches, HighPerformanceConfig.MAX_WORKERS_PER_SITE)
        # use ProcessPoolExecutor for parallel optimization
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(parallel_optimization_task, *args)
                for args in task_args
            ]
            # collect results
            for future in as_completed(futures):
                current_smts, current_efficiency = future.result()
                if current_efficiency > best_efficiency:
                    best_efficiency = current_efficiency
                    best_smts = current_smts
                    no_improvement_count = 0
                    logging.info(f"New best efficiency: {best_efficiency:.4f} for SMTs: {best_smts}")
                else:
                    no_improvement_count += 1
                # Early stopping
                if no_improvement_count >= max_no_improvement:
                    logging.info(f"Early stopping after {no_improvement_count} iterations without improvement")
                    break
        return best_smts

def process_id(id):
    try:
        # set random seed for reproducibility
        np.random.seed(os.getpid())  # use process ID as seed
        load_data = CalibrationData(use_initial_crop_parameters=True)
        optimizer = SiteDeficIrrigateOptimize(id, load_data)
        optimized_smts = optimizer._optimize(num_searches=1000)
        
        # Calculate yield and TIRR for the best SMTs if optimization was successful
        best_yield = None
        best_tirr = None
        if optimized_smts is not None:
            out = optimizer._run_defic_model(optimized_smts)
            best_yield = out['Dry yield (tonne/ha)'].mean().round(2)
            best_tirr = out['Seasonal irrigation (mm)'].mean().round(2)
        
        return (
            id,
            optimizer.baseline_yield,
            optimizer.baseline_irrigtion,
            optimized_smts.tolist() if optimized_smts is not None else None,
            best_yield,
            best_tirr,
        )
    except Exception as e:
        print(f"ID {id} failed: {str(e)}")
        return (id, None, None, None, None, None)
    
def parallel_optimization_task(x0, id, load_data_params, baseline_yield, 
                              baseline_irrigation, bounds):
    try:
        # create a dummy class to hold load_data parameters
        class DummyLoadData:
            def __init__(self, **kwargs):
                self.__dict__.update(kwargs)

        # create a dummy load_data instance
        dummy_load_data = DummyLoadData(**load_data_params)
        # create optimizer instance
        optimizer = SiteDeficIrrigateOptimize(id, dummy_load_data, run_baseline=False)
        optimizer.baseline_yield = baseline_yield
        optimizer.baseline_irrigtion = baseline_irrigation

        # perform optimization
        # use SLSQP method for constrained optimization
        res = minimize(
            fun=optimizer._evaluate,
            x0=x0,
            args=(),
            method="SLSQP",
            bounds=bounds,
            constraints={ 
                "type": "ineq",
                "fun": lambda x: optimizer._constraint_obs(x),
            },
            options={"maxiter": 1000}  # revised maxiter for more iterations
        )
        if res.success:
            current_smts = np.clip(res.x.round().astype(int), 50, 100)
            current_efficiency = -res.fun
            return (current_smts, current_efficiency)
        return (None, -np.inf)
    except Exception as e:
        print(f"Optimization task failed: {str(e)}")
        return (None, -np.inf)

def main():
    start_time = time.time()
    logging.info("Starting optimization process...")
    log_system_resources()  # Log initial system state
    
    # load experimental sites
    experiment_sites = pd.read_csv('../data/xinjiang_zones/experimental_sites.csv')
    ids = experiment_sites.ID.unique()
    # ids = ['Weili','Korla']
    logging.info(f"Processing {len(ids)} sites")
    
    results = []
    failed_ids = []
    
    max_site_workers = min(HighPerformanceConfig.MAX_SITE_WORKERS, len(ids))
    with ProcessPoolExecutor(max_workers=max_site_workers) as executor:
        from tqdm import tqdm
        future_to_id = {executor.submit(process_id, id): id for id in ids}
        
        for future in tqdm(as_completed(future_to_id), total=len(ids), desc="optimization progress"):
            id = future_to_id[future]
            try:
                result = future.result(timeout=3600)  # 1 hour timeout per site
                if result[1] is not None:  # Check if optimization was successful
                    results.append(result)
                    logging.info(f"Successfully processed ID {id}")
                else:
                    failed_ids.append(id)
                    logging.warning(f"Failed to optimize ID {id}")
            except Exception as e:
                failed_ids.append(id)
                logging.error(f"Exception processing ID {id}: {str(e)}")
            
            # Log system resources every 10 completed tasks
            if len(results) + len(failed_ids) % 10 == 0:
                log_system_resources()
    
    # Log summary
    logging.info(f"Completed optimization in {time.time() - start_time:.2f} seconds")
    logging.info(f"Successful: {len(results)}, Failed: {len(failed_ids)}")
    if failed_ids:
        logging.warning(f"Failed IDs: {failed_ids}")
    
    # Final system resource log
    log_system_resources()
    
    # convert to DataFrame
    results_df = pd.DataFrame(results, 
        columns=['ID', 'BaselineYield', 'BaselineIrrigation', 'OptimizedSMTs', 'BestYield', 'BestTIRR'])
    # filter out failed optimizations
    smt_cols = ['SMT1', 'SMT2', 'SMT3', 'SMT4']
    results_df[smt_cols] = pd.DataFrame(
        results_df['OptimizedSMTs'].apply(
            lambda x: x if x else [np.nan]*4).tolist(),
        index=results_df.index)
    # set the order of results_df['ID']
    desired_order = ['AKA','CLD','Weili','Tumushuke','Korla','Shihezi146','FKD', 'Shawan', 'Shihezi', 'Huyanghe', 'Urumqi']
    results_df['ID'] = pd.Categorical(results_df['ID'], categories=desired_order, ordered=True)
    results_df = results_df.sort_values('ID').reset_index(drop=True)
    # save results
    output_path = f'../results/simulation10km/optimized_smts_results_{IRRENTTERVAL}.csv'
    print(results_df)
    results_df.drop(columns=['OptimizedSMTs']).to_csv(output_path, index=False)
    print(f"finished optimization, results saved to {output_path}")


def test():
    load_data = CalibrationData(use_initial_crop_parameters=True)
    optimizer = SiteDeficIrrigateOptimize('Korla', load_data)
    print(optimizer.baseline_yield, optimizer.baseline_irrigtion)
    smts = [80, 95, 100, 90]
    out = optimizer._run_defic_model(smts)
    print(out['Dry yield (tonne/ha)'].mean(), out['Seasonal irrigation (mm)'].mean())


if __name__ == '__main__':
    main()
    # test()

