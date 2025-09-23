import os
os.environ['DEVELOPMENT'] = 'True'

import pandas as pd
from sklearn.metrics import r2_score
from scipy.optimize import minimize
from calibrate_validate.summary_models_output import ModelOutputSummary
from calibrate_validate.load_data_for_calibration import CalibrationData
from calibrate_validate.prepare_aquacrop_input import *
from calibrate_validate.single_station_calibration import run_model
from calibrate_validate.calibrated_plots import merging_data
from scipy.optimize import differential_evolution
from scipy.optimize import dual_annealing
from calibrate_validate.calibrated_plots import convert_biomass, convert_canopy_cover, convert_soil_water


import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)



def get_years_for_every_varityes(id, variety, phenology_data):
    '''
    get years for every varityes
    '''
    t_years = phenology_data[(phenology_data.ID == id) & (phenology_data.varieties == variety) & (phenology_data.cal_label == 'cal')].years.unique()
    if len(t_years) <= 1:  # if calibration data only has one year, do not split calibration and validation data
        years = phenology_data[(phenology_data.ID == id) & (phenology_data.varieties == variety)].years.unique()
    else:
        years = t_years
    return years

def process_model_output(id, variety, phenology_data, soil, weather, irrigation_data, var_paramters):
    '''process model output'''
    mos = ModelOutputSummary()
    for year in get_years_for_every_varityes(id, variety, phenology_data):
        sample = phenology_data[(phenology_data.ID == id) & (phenology_data.years == year)].samples.iloc[0]
        sample = str(sample).strip()  # Convert to string and strip any whitespace
        phenology = pre_phenology(id, year, phenology_data)
        planting_date = phenology.planting.iloc[0]
        irrigation = pre_irrigation_schedule(id, sample, irrigation_data)
        crop_paramters = update_phenology_and_create_crop_parameters(var_paramters, phenology)
        model = run_model(year, soil, weather, irrigation, crop_paramters, planting_date)
        mos.get_output_results(id, year, variety, sample, planting_date, model)
    mos.concat_output_results()
    return mos

def r2_reward(sim_results, obs_results, optimized_object):
    """
    Calculate the R2 (coefficient of determination) reward for model optimization.

    Parameters:
    sim_results (ModelOutputSummary): simulate results class.
    obs_results (CalibrationData): calibrate input data calss.
    optimized_object (str): The objective for optimization, which can be one of the following:
        'canopy_cover', 'biomass', 'soil_water', 'yield'

    Returns:
    float: The R2 score between simulated and observed results for the specified objective.
    """
    conbined_data = {'canopy_cover':[sim_results.growth, obs_results.obs_cc, 'canopy_cover', 'cc'],
                'biomass':[sim_results.growth, obs_results.obs_biomass, 'biomass', 'biomass(kg/ha)'],
                'soil_water':[sim_results.water_storage, obs_results.obs_soil_water, 'sim_water_depth_10cm','obs_water_depth_10cm'],
                'yield':[sim_results.final_stats, obs_results.obs_yield,  'Dry yield (tonne/ha)', 'yield_sta(t/ha)']}

    data = conbined_data[optimized_object]
    sim_data = data[0]
    obs_data = data[1]
    sim_col = data[2]
    obs_col = data[3]

    if optimized_object == 'canopy_cover':
        sim_d = convert_canopy_cover(sim_data)

    elif optimized_object =='soil_water':
        sim_d = convert_soil_water(sim_data)

    elif optimized_object == 'yield':
        sim_d = sim_data

    elif optimized_object == 'biomass':
        sim_d = convert_biomass(sim_data )

    else:
        print('objective not found!')
    merged_data = merging_data(sim_d, obs_data, sim_col, obs_col, objective_labes=1)
    r2 = r2_score(merged_data[sim_col], merged_data[obs_col])
    return r2


def update_variety_parameters(params, param_names, variety_parameters):
    for i, param in enumerate(param_names):
        setattr(variety_parameters, param, params[i])

def round_params(params, step_sizes):
    # Round each parameter to the nearest multiple of its step size
    return [round(params[i] / step_sizes[i]) * step_sizes[i] for i in range(len(params))]

def parameter_optimization_de(id, variety, cal_data, soil, weather, variety_parameters, guess_params, optimized_object, maxiternum=50):
    '''
    Optimize canopy cover parameters using Differential Evolution (DE).
    '''
    param_names = guess_params['param_names']
    bounds = guess_params['bound']
    steps = guess_params['step_size']

    def objective_function(params):
        rounded_params = round_params(params, steps)
        update_variety_parameters(rounded_params, param_names, variety_parameters)
        mos = process_model_output(id, variety, cal_data.phenology_data, soil, weather, cal_data.irrigation_data, variety_parameters)
        reward = r2_reward(mos, cal_data, optimized_object)
        return -reward  # Minimize negative R²

    result = differential_evolution(objective_function, bounds, strategy='best1bin', maxiter=maxiternum, disp=True)
    optimized_params = result.x
    optimized_result = -result.fun  # Convert back to positive reward

    rounded_params = round_params(optimized_params, steps)
    update_variety_parameters(rounded_params, param_names, variety_parameters)
    return optimized_result

def sequence_variety_optimize(id, variety, cal_data, soil, weather, variety_parameters):
    '''
    Sequence optimization using a genetic algorithm for a variety.
    '''
    params_dic = {'canopy_cover' : {'param_names':  ['CGC', 'CDC','CCx', 'Zmax'], 
                                'initial_guess': [variety_parameters.CGC, variety_parameters.CDC, variety_parameters.CCx, variety_parameters.Zmax],
                                'bound' : [(0.07, 0.1), (0.05, 0.1), (0.7, 1), (0.6, 1.5)],
                                'step_size':[0.01, 0.01, 0.01, 0.1]},
                  'biomass' : {'param_names':  ['WP', 'WPy','Kcb',], 
                               'bound': [(16, 22), (70, 75), (1, 1.3)],
                                'initial_guess': [variety_parameters.WP, variety_parameters.WPy, variety_parameters.Kcb],
                                'step_size':[1, 1, 0.1]},
                    'yield' : {'param_names':  ['HI0'], 
                                'initial_guess': [variety_parameters.HI0],
                                'bound': [(0.25, 0.45)],
                                'step_size':[0.01]} }  
    all_optimized_results = []
    for optimized_object, guess_params in list(params_dic.items())[2:]:
        print(f'Optimizing object is {optimized_object}')

        optimized_result = parameter_optimization_de(id, variety, cal_data, soil, weather, variety_parameters, guess_params, optimized_object, maxiternum=50)
        all_optimized_results.append([optimized_object, optimized_result])

    all_optimized_results_df = pd.DataFrame(all_optimized_results, columns=['optimized_object', 'optimized_result'])
    all_optimized_results_df['variety'] = variety
    all_optimized_results_df['ID'] = id
    return all_optimized_results_df

    
def main():
    all_id_optimized_params = []
    all_id_optimized_results = []

    for id in cal_data.init_crop_parameters.ID.unique()[1:2]:
        weather = pre_weather(id, cal_data.weather_file_paths)
        soil = pre_soil(id, cal_data.soil_data)

        for variety in cal_data.init_crop_parameters[cal_data.init_crop_parameters.ID == id].varieties.unique():
            print(f'Optimizing {id} {variety}')
            variety_parameters = pre_crop_parameters(id, variety, cal_data.init_crop_parameters)
            optimized_results = sequence_variety_optimize(id, variety, cal_data, soil, weather, variety_parameters)
            print(optimized_results)

            all_id_optimized_results.append(optimized_results)
            all_id_optimized_params.append(variety_parameters.output_crop_parameters())
            # print(optimized_params_df.head())

    all_id_optimized_params_df = pd.concat(all_id_optimized_params, ignore_index=True)
    all_id_optimized_results_df = pd.concat(all_id_optimized_results, ignore_index=True)
    all_id_optimized_results_df.to_csv('../data/calibration/optimizated_results.csv', index=False)
    all_id_optimized_params_df.to_csv('../data/calibration/optimizated_crop_parameters.csv', index=False)
            

if __name__ == '__main__':
    cal_data = CalibrationData(use_initial_crop_parameters=True)
    main()
    


