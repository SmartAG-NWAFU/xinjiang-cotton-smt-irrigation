
from optimization_parameters import *
import time

def optimize_canopy_cover_de(id, variety, cal_data, soil, weather, variety_parameters, maxiternum=20):
    '''
    Optimize canopy cover parameters using Differential Evolution (DE).
    '''

    param_names = ['CGC', 'CDC']
    bounds = [(0.06, 0.1), (0.05, 0.1)]  # CGC and CDC bounds
    steps = [0.01, 0.01]  # Step sizes

    def objective_function(params):
        rounded_params = round_params(params, steps)
        update_variety_parameters(rounded_params, param_names, variety_parameters)
        mos = process_model_output(id, variety, cal_data.phenology_data, soil, weather, cal_data.irrigation_data, variety_parameters)
        reward = r2_reward(mos, cal_data, 'canopy_cover')
        return -reward  # Minimize negative R²

    start_time = time.time()  # Start timing
    result = differential_evolution(objective_function, bounds, strategy='best1bin', maxiter=maxiternum, disp=True)
    end_time = time.time()  # End timing

    optimized_params = result.x
    optimized_result = -result.fun  # Convert back to positive reward

    rounded_params = round_params(optimized_params, steps)
    update_variety_parameters(rounded_params, param_names, variety_parameters)

    print(f"Differential Evolution - Time: {end_time - start_time} seconds")
    print(f"Differential Evolution - Parameters: {rounded_params}")
    return optimized_result

def optimize_canopy_cover_sa(id, variety, cal_data, soil, weather, variety_parameters, maxiternum=20):
    '''
    Optimize canopy cover parameters using Simulated Annealing (SA).
    '''

    param_names = ['CGC', 'CDC']
    bounds = [(0.06, 0.1), (0.05, 0.1)]
    steps = [0.01, 0.01]  # Step sizes

    def objective_function(params):
        rounded_params = round_params(params, steps)
        update_variety_parameters(rounded_params, param_names, variety_parameters)
        mos = process_model_output(id, variety, cal_data.phenology_data, soil, weather, cal_data.irrigation_data, variety_parameters)
        reward = r2_reward(mos, cal_data, 'canopy_cover')
        return -reward  # Minimize negative R²

    start_time = time.time()  # Start timing
    result = dual_annealing(objective_function, bounds, maxiter=maxiternum)
    end_time = time.time()  # End timing

    optimized_params = result.x
    optimized_result = -result.fun  # Convert back to positive reward

    rounded_params = round_params(optimized_params, steps)
    update_variety_parameters(rounded_params, param_names, variety_parameters)

    print(f"Simulated Annealing - Time: {end_time - start_time} seconds")
    print(f"Simulated Annealing - Parameters: {rounded_params}")
    return optimized_result

def optimize_canopy_cover_bfgs(id, variety, cal_data, soil, weather, variety_parameters, maxiternum=500):
    '''
    Optimize canopy cover parameters using BFGS.
    '''

    initial_guess = [variety_parameters.CGC, variety_parameters.CDC, variety_parameters.CCx, variety_parameters.Zmax]  # Initial guesses for CGC and CDC
    param_names = ['CGC', 'CDC','CCx', 'Zmax']
    steps = [0.01, 0.01, 0.01, 0.1]  # Step sizes

    def objective_function(params):
        rounded_params = round_params(params, steps)
        update_variety_parameters(rounded_params, param_names, variety_parameters)
        mos = process_model_output(id, variety, cal_data.phenology_data, soil, weather, cal_data.irrigation_data, variety_parameters)
        reward = r2_reward(mos, cal_data, 'canopy_cover')
        return -reward  # Minimize negative R²

    start_time = time.time()  # Start timing
    result = minimize(objective_function, initial_guess, method='BFGS', options={'maxiter': maxiternum})
    end_time = time.time()  # End timing

    optimized_params = result.x
    optimized_result = -result.fun  # Convert back to positive reward

    rounded_params = round_params(optimized_params, steps)
    update_variety_parameters(rounded_params, param_names, variety_parameters)

    print(f"BFGS - Time: {end_time - start_time} seconds")
    print(f"BFGS - Parameters: {rounded_params}")
    return optimized_result


def optimize_canopy_cover_nelder_mead(id, variety, cal_data, soil, weather, variety_parameters, maxiternum=20):
    '''
    Optimize canopy cover parameters using Nelder-Mead.
    '''

    initial_guess = [variety_parameters.CGC, variety_parameters.CDC, variety_parameters.CCx, variety_parameters.Zmax]  # Initial guesses for CGC and CDC
    param_names = ['CGC', 'CDC','CCx', 'Zmax']
    steps = [0.01, 0.01, 0.01, 0.1]  # Step sizes

    def objective_function(params):
        rounded_params = round_params(params, steps)
        update_variety_parameters(rounded_params, param_names, variety_parameters)
        mos = process_model_output(id, variety, cal_data.phenology_data, soil, weather, cal_data.irrigation_data, variety_parameters)
        reward = r2_reward(mos, cal_data, 'canopy_cover')
        return -reward  # Minimize negative R²

    start_time = time.time()  # Start timing
    result = minimize(objective_function, initial_guess, method='Nelder-Mead', options={'maxiter': maxiternum})
    end_time = time.time()  # End timing

    optimized_params = result.x
    optimized_result = -result.fun  # Convert back to positive reward

    rounded_params = round_params(optimized_params, steps)
    update_variety_parameters(rounded_params, param_names, variety_parameters)

    print(f"Nelder-Mead - Time: {end_time - start_time} seconds")
    print(f"Nelder-Mead - Parameters: {rounded_params}")
    return optimized_result


def parameter_optimization_nelder_mead(id, variety, cal_data, soil, weather, variety_parameters, gusses_parameters, optimized_object, maxiternum=50):
    '''
    Sequence optimization using a genetic algorithm for a variety.
    '''
    param_names = gusses_parameters['param_names']
    initial_guess = gusses_parameters['initial_guess']
    steps = gusses_parameters['step_size']

    def objective_function(params):
        rounded_params = round_params(params, steps)
        update_variety_parameters(rounded_params, param_names, variety_parameters)
        mos = process_model_output(id, variety, cal_data.phenology_data, soil, weather, cal_data.irrigation_data, variety_parameters)
        reward = r2_reward(mos, cal_data, optimized_object)
        return -reward  # Minimize negative R²

    result = minimize(objective_function, initial_guess, method='Nelder-Mead', options={'maxiter': maxiternum})

    optimized_params = result.x
    optimized_result = -result.fun  # Convert back to positive reward

    rounded_params = round_params(optimized_params, steps)
    update_variety_parameters(rounded_params, param_names, variety_parameters)
    return optimized_result


def optimize_canopy_cover_sa(id, variety, cal_data, soil, weather, variety_parameters, guess_params, optimized_object, maxiternum=50):
    '''
    Optimize canopy cover parameters using Simulated Annealing (SA).
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

    result = dual_annealing(objective_function, bounds, maxiter=maxiternum)
    optimized_params = result.x
    optimized_result = -result.fun  # Convert back to positive reward

    rounded_params = round_params(optimized_params, steps)
    update_variety_parameters(rounded_params, param_names, variety_parameters)
    return optimized_result




    
def run_different_algorithms(id, variety, cal_data, soil, weather, variety_parameters, max_iterations=20):

    results = []
    
    # Differential Evolution (DE)
    print("Testing Differential Evolution")
    start_time = time.time()
    result_de = optimize_canopy_cover_de(id, variety, cal_data, soil, weather, variety_parameters, max_iterations)
    end_time = time.time()
    results.append({
        "Algorithm": "Differential Evolution",
        "Result": result_de,
        "Time (s)": end_time - start_time
    })
    
    # Simulated Annealing (SA)
    print("Testing Simulated Annealing")
    start_time = time.time()
    result_sa = optimize_canopy_cover_sa(id, variety, cal_data, soil, weather, variety_parameters, max_iterations)
    end_time = time.time()
    results.append({
        "Algorithm": "Simulated Annealing",
        "Result": result_sa,
        "Time (s)": end_time - start_time
    })
    
    # Nelder-Mead
    print("Testing Nelder-Mead")
    start_time = time.time()
    result_nm = optimize_canopy_cover_nelder_mead(id, variety, cal_data, soil, weather, variety_parameters, max_iterations)
    end_time = time.time()
    results.append({
        "Algorithm": "Nelder-Mead",
        "Result": result_nm,
        "Time (s)": end_time - start_time
    })
    
    # BFGS
    print("Testing BFGS")
    start_time = time.time()
    result_bfgs = optimize_canopy_cover_bfgs(id, variety, cal_data, soil, weather, variety_parameters, max_iterations)
    end_time = time.time()
    results.append({
        "Algorithm": "BFGS",
        "Result": result_bfgs,
        "Time (s)": end_time - start_time
    })

    return results


def test_algorithms():
     for id in cal_data.init_crop_parameters.ID.unique()[-1:]:
        weather = pre_weather(id, cal_data.weather_file_paths)
        soil = pre_soil(id, cal_data.soil_data)

        for variety in cal_data.init_crop_parameters[cal_data.init_crop_parameters.ID == id].varieties.unique()[1:]:
            variety_parameters = pre_crop_parameters(id, variety, cal_data.init_crop_parameters)
            max_iterations = 100  # Set the maximum number of iterations for each algorithm
            results = run_different_algorithms(id, variety, cal_data, soil, weather, variety_parameters, max_iterations)
            # Print results for each algorithm
            print(pd.DataFrame(results))

if __name__ == '__main__':
    test_algorithms()

    

