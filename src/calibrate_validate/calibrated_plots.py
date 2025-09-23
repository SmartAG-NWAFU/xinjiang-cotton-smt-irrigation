import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np

def selecting_soil_depth_for_calibration(sim_soil_water, obs_soil_water, soil_depth=0.5):
    '''
    Calibration for selected soil depth (m).
    Adjusts behavior dynamically based on the soil depth provided.
    
    '''

    def selecting_soil_depth(soil_depth):
        '''
        Selecting soil depth for calibration.
        '''
        # Define depth ranges and corresponding column selections

        depth_mapping = {
            0.1: ([10], ['th1']),
            0.2: ([10, 20], ['th1', 'th2']),
            0.3: ([10, 20, 30], ['th1', 'th2', 'th3']),
            0.4: ([10, 20, 30, 40], ['th1', 'th2', 'th3', 'th4']),
            0.5: ([10, 20, 30, 40, 50], ['th1', 'th2', 'th3', 'th4', 'th5']),
            0.6: ([10, 20, 30, 40, 50, 60], ['th1', 'th2', 'th3', 'th4', 'th5', 'th6']),
            0.7: ([10, 20, 30, 40, 50, 60, 70], ['th1', 'th2', 'th3', 'th4', 'th5', 'th6', 'th7']),
            0.8: ([10, 20, 30, 40, 50, 60, 70, 80], ['th1', 'th2', 'th3', 'th4', 'th5', 'th6', 'th7', 'th8']),
            0.9: ([10, 20, 30, 40, 50, 60, 70, 80, 90], ['th1', 'th2', 'th3', 'th4', 'th5', 'th6', 'th7', 'th8', 'th9']),
            1.0: ([10, 20, 30, 40, 50, 60, 70, 80, 90, 100], ['th1', 'th2', 'th3', 'th4', 'th5', 'th6', 'th7', 'th8', 'th9', 'th10'])
        }

        if soil_depth not in depth_mapping:
            raise ValueError(f"Soil depth {soil_depth} is not supported. Please select from {list(depth_mapping.keys())}.")
        
        return depth_mapping[soil_depth]

    def filter_obs_soil_water(obs_soil_water, selected_depths):
        '''
        Filter the observation soil water data for the selected depths and calculate the mean volumetric water content.
        '''
        # Filter the observation soil water data for the selected depths
        filtered_obs_soil_water = obs_soil_water[obs_soil_water['depth(cm)'].isin(selected_depths)]
        # Group by 'ID' and 'dap', then calculate the mean volumetric water content for each group
        obs_soil_water_result = filtered_obs_soil_water.groupby(['ID', 'years','dap'])['volumetric_water(%)'].mean().reset_index()
        # Convert percentage to point number, then to equivalent depth in mm/m, then multiply by soil depth
        obs_soil_water_result[obs_water_depth_col] = obs_soil_water_result['volumetric_water(%)'] * 0.01 * 1000 * soil_depth
        obs_soil_water_result = obs_soil_water_result[['ID', 'years','dap', obs_water_depth_col]]

        return obs_soil_water_result
    
    def fliter_sim_soil_water(sim_soil_water, selected_columns):
        '''
        Filter the simulation soil water data for the selected depths and calculate the mean volumetric water content.
        '''
        # Calculate the mean of selected columns in sim_soil_water based on the soil depth
        sim_soil_water['temp'] = sim_soil_water[selected_columns].mean(axis=1)
        # Convert simulation soil water storage to equivalent depth in mm/m, then multiply by soil depth
        sim_soil_water[sim_water_depth_col] = sim_soil_water['temp'] * 1000 * soil_depth
        # Filter for rows that are within the growing season
        sim_soil_water = sim_soil_water[sim_soil_water['growing_season'] == 1]
        sim_soil_water_results = sim_soil_water[['dap', sim_water_depth_col, 'ID', 'years', 'samples', 'varieties']]
        return sim_soil_water_results
    

    selected_depths, selected_columns = selecting_soil_depth(soil_depth)
    # Dynamically create column names based on soil_depth
    obs_water_depth_col = f'obs_water_depth_{int(soil_depth * 100)}cm'
    sim_water_depth_col = f'sim_water_depth_{int(soil_depth * 100)}cm'

    # Filter the observation soil water data for the selected depths and calculate the mean volumetric water content
    obs_soil_water_result = filter_obs_soil_water(obs_soil_water, selected_depths)

    # Filter the simulation soil water data for the selected depths and calculate the mean volumetric water content
    sim_soil_water_results = fliter_sim_soil_water(sim_soil_water, selected_columns)

    return [sim_soil_water_results, sim_water_depth_col, obs_soil_water_result, obs_water_depth_col]


def soil_water_depth_calibration_test(id, sim_soil_water, obs_soil_water):
    '''
    testing the most suitable soil water depth for calribration.

    Parameters:
        sim_soil_water (pd.DataFrame): DataFrame containing simulated soil water data.
        obs_soil_water (pd.DataFrame): DataFrame containing observed soil water data.
        soil_depth (float, optional): Soil depth in meters. Default is 0.5.

    Returns:
        None
    '''
    all_depth_metrics = []
    for soil_d in np.arange(0.1, 1.1, 0.1):
        results =  selecting_soil_depth_for_calibration(sim_soil_water, obs_soil_water, soil_depth=round(soil_d,1))
        sim_soil_water_depth = results[0]
        sim_soil_water_colname = results[1]
        obs_soil_water_depth = results[2]
        obs_soil_water_colname = results[3]

        # merge observations with id, years, and samples of simulation
        metrics = calculate_metrics_process_grouped_data(sim_soil_water_depth, obs_soil_water_depth,
                                                          sim_soil_water_colname, obs_soil_water_colname)
        # Calculate the mean of the metrics
        # emove R2 < -100, and then calculate the mean value
        mean_metrics = metrics.loc[metrics['R2'] >= -100, 'R2'].mean(axis=0)
        all_depth_metrics.append(mean_metrics)
        # plot_calibrated_results(id, sim_soil_water_depth, obs_soil_water_depth, metrics, sim_soil_water_colname, obs_soil_water_colname)
    most_suitable_soil_depth = round(np.argmax(all_depth_metrics).astype('double') * 0.1, 1)  
    return most_suitable_soil_depth

def calculate_metrics(sim_values, obs_values):
    '''calculating R², RMSE and d values'''
    if len(sim_values) == 0 or len(obs_values) == 0:
        return 0, 0, 0

    # R2
    r2 = r2_score(obs_values, sim_values)
    # RMSE
    rmse = np.sqrt(mean_squared_error(obs_values, sim_values))
    # d
    d = 1 - (np.sum((obs_values - sim_values) ** 2) / np.sum(
        (np.abs(sim_values - np.mean(obs_values)) + np.abs(obs_values - np.mean(obs_values))) ** 2))
    return r2, rmse, d

def merging_data(sim_data, obs_data, sim_col, obs_col, objective_labes=1):
    """
    Merge simulated and observed data based on specified columns and merge method.

    Parameters:
        sim_data (pd.DataFrame): DataFrame containing simulated data.
        obs_data (pd.DataFrame): DataFrame containing observed data.
        sim_col (str): Column name in sim_data used for merging and comparison.
        obs_col (str): Column name in obs_data used for merging and comparison.
        objective_labes (int, optional): Purpose of the merge. If True, for calculating metrics; if False, for plotting. Default is 1.

    Returns:
        pd.DataFrame: Merged data.

    Notes:
        - If objective_labes is True, the merge method is 'inner', keeping only rows with matching values in both DataFrames.
        - If objective_labes is False, the merge method is 'left', keeping all rows from sim_data and matching rows from obs_data.
        - For soil water ('sim_water_depth_mm'), the merge is on 'ID' and 'date' columns.
        - For canopy cover and biomass, the merge is on 'ID', 'years', 'samples', and 'dap' columns.
    """
    if objective_labes:
        # this merged data is used for calculating metrics
        how_way = 'inner'
    else:
        # this merged data is used for plotting
        how_way = 'left'

    if 'sim_water_depth_' in sim_col:
        # for soil water, use 'date' and 'ID
        merged_data = pd.merge(sim_data[['ID', 'dap', 'years', sim_col]],
                    obs_data[['ID','years','dap', obs_col]],
                    on=['ID', 'years','dap'],
                    how=how_way)
    elif sim_col == 'Dry yield (tonne/ha)':
        # for dry yield, use 'years','samples', and 'ID'
        merged_data = pd.merge(sim_data[['ID', 'years','samples', sim_col]],
                            obs_data[['ID', 'years','samples', obs_col]],
                            on=['ID', 'years','samples'],
                            how=how_way)
    else:
        # for canopy cover and biomass, use 'years','samples', 'ID' and 'dap'
        merged_data = pd.merge(sim_data[['ID', 'years', 'samples', 'dap', sim_col]],
                            obs_data[['ID', 'years', 'samples', 'dap', obs_col]],
                            on=['ID', 'years', 'samples', 'dap'],
                            how=how_way)
    return merged_data

def save_merged_data_for_scatterplot(sim_data, obs_data, sim_col, obs_col, cal_val_labels):
    '''
    Merge simulated and observed data, calculate metrics, and save the merged data to a CSV file.

    This function merges simulated and observed data based on specified columns, calculates metrics (R2, RMSE, and d),
    saves the merged data to a CSV file, and returns the metrics.

    Parameters:
        sim_data (pd.DataFrame): DataFrame containing simulated data.
        obs_data (pd.DataFrame): DataFrame containing observed data.
        sim_col (str): Column name in sim_data used for merging and comparison.
        obs_col (str): Column name in obs_data used for merging and comparison.

    Returns:
        list: A list containing the column name and the calculated metrics (R2, RMSE, and d).

    Notes:
        - The function uses the merging_data function to merge the simulated and observed data.
        - The function calculates metrics using the calculate_metrics function.
        - The function saves the merged data to a CSV file in the '../results/calibration' directory.
        - The function returns a list with the column name and the calculated metrics.
    '''
    # Merge simulated and observed data
    merged_df = merging_data(sim_data, obs_data, sim_col, obs_col, objective_labes=1)
    # Add calibration labels
    added_label_merged_data = pd.merge(merged_df, cal_val_labels[['ID', 'years', 'cal_label']], on=['ID', 'years'], how='left')

    # Calculate metrics
    sim_metrics = []
    # Calculate metrics for each calibration label
    for cal_la in ['cal', 'val']:
        merged_data = added_label_merged_data[added_label_merged_data['cal_label'] == cal_la]
        # print(merged_data)
        r2, rmse, d = calculate_metrics(merged_data[sim_col], merged_data[obs_col])
        sim_metrics.append([cal_la, r2, rmse, d])

    metrics_df = pd.DataFrame(sim_metrics, columns=['cal_label', 'r2', 'rmse', 'd'])
    metrics_df['calibrated_names'] = sim_col

    # Save merged data
    added_label_merged_data.rename(columns={sim_col: 'sim_values', obs_col: 'obs_values'}, inplace=True)
    if sim_col == 'Dry yield (tonne/ha)':
        sim_col = 'yield_t_ha'
    added_label_merged_data.to_csv(f'../../results/calibration/{sim_col}.csv', index=False)

    return metrics_df

    
def calculate_metrics_process_grouped_data(sim_data, obs_data, sim_col, obs_col):
    '''
    Process grouped data and calculate metrics.

    This function merges simulated and observed data, groups the merged data by 'ID' and 'years',
    and calculates metrics (R2, RMSE, and d) for each group.

    Parameters:
        sim_data (pd.DataFrame): DataFrame containing simulated data.
        obs_data (pd.DataFrame): DataFrame containing observed data.
        sim_col (str): Column name in sim_data used for merging and comparison.
        obs_col (str): Column name in obs_data used for merging and comparison.

    Returns:
        pd.DataFrame: A DataFrame containing the calculated metrics for each group.
                      The DataFrame includes columns 'ID', 'years', 'R2', 'RMSE', and 'd'.

    Notes:
        - The function uses the merging_data function to merge the simulated and observed data.
        - The function groups the merged data by 'ID' and 'years'.
        - For each group, the function calculates metrics (R2, RMSE, and d) using the calculate_metrics function.
        - The function returns a DataFrame with the calculated metrics for each group.
    '''
    # Merge simulated and observed data
    merged_data = merging_data(sim_data, obs_data, sim_col, obs_col, objective_labes=1)
    results_list = []  # List to hold individual result dictionaries
    for name, group in merged_data.groupby(['ID', 'years']):
        sim_values = group[sim_col]
        obs_values = group[obs_col]

        # Calculate metrics using the new function
        r2, rmse, d = calculate_metrics(sim_values, obs_values)
        
        # Append the result dictionary to the results list
        results_list.append({
            'ID': name[0],
            'years': name[1],
            'R2': r2,
            'RMSE': rmse,
            'd': d
        })
    # Create a DataFrame from the results list and reset index
    results = pd.DataFrame(results_list)
    return results.reset_index(drop=True)

def ploting_calibrated_results_dap(sim_data, obs_data, sim_col, obs_col):
    '''
    Plotting calibration pictures with dap

    This function merges simulated and observed data, creates a FacetGrid with a line plot for simulated data
    and a point plot for observed data, and adds evaluation precision text.

    Parameters:
        id (str): ID of the plot.
        sim_data (pd.DataFrame): DataFrame containing simulated  data.
        obs_data (pd.DataFrame): DataFrame containing observed data.
        metrics (pd.DataFrame): DataFrame containing calculated metrics.
        sim_col (str): Column name in sim_data used for plotting.
        obs_col (str): Column name in obs_data used for plotting.

    Returns:
        None

    Notes:
        - The function uses the merging_data function to merge the simulated and observed data.
        - The function creates a FacetGrid with a line plot for simulated data and a point plot for observed data.
        - The function adds evaluation precision text to the plot.
        - The function saves the plot to a file in the '../figs/calibration' directory.
    '''
    def plot_simulated_observed(merged_data_id, sim_col, obs_col, grouped_metrics_id, outputfile, id):
        plt.figure(figsize=(12, 8))
        g = sns.FacetGrid(data=merged_data_id, col="years", col_wrap=4, aspect=1.5)
        g.map(sns.lineplot, "dap", sim_col, color="black")
        g.map(sns.pointplot, "dap", obs_col, color="red", markers="d", dodge=True, capsize=0.2, errorbar="sd")
        
        g.set(xticks=[0, 50, 100, 150, 200])
        g.set_axis_labels("Days after planting", y_var=sim_col)

        if sim_col == 'canopy_cover':
            g.set(yticks=[0, 20, 40, 60, 80, 100])

        for year, ax_list in zip(grouped_metrics_id['years'].unique(), g.axes.ravel()):
            subset = grouped_metrics_id[grouped_metrics_id['years'] == year]
            for i, (index, row) in enumerate(subset.iterrows()):
                text = f"R2: {row['R2']:.2f}\nRMSE: {row['RMSE']:.2f}\nd: {row['d']:.2f}"
                ax_list.text(0.2, 0.8, text, transform=ax_list.transAxes, ha='center', va='top')
        
        plt.tight_layout()
        
        if not os.path.exists(outputfile):
            os.makedirs(outputfile)
        
        plt.savefig(os.path.join(outputfile, f'{id}_{sim_col}.png'))
        plt.close()

    #calculate metrics
    grouped_metrics = calculate_metrics_process_grouped_data(sim_data, obs_data, sim_col, obs_col)
    merged_data = merging_data(sim_data, obs_data, sim_col, obs_col, objective_labes=0)
    # ploting for each id
    for id in merged_data['ID'].unique():
        if not ((id == 'Weili' and sim_col == 'biomass') or (id == 'Shihezi' and sim_col == 'canopy_cover') or 
                (id == 'Korla' and sim_col == 'biomass') or (id == 'Tumushuke' and sim_col == 'biomass') or
                (id == 'Huyanghe' and sim_col == 'canopy_cover') or (id == 'Shawan')):
            merged_data_id = merged_data[merged_data['ID'] == id]
            grouped_metrics_id = grouped_metrics[grouped_metrics['ID'] == id]
            plot_simulated_observed(merged_data_id, sim_col, obs_col, grouped_metrics_id, '../../figs/calibration', id)

def convert_canopy_cover(sim_cc):
    '''calculating canopy cover metrics and ploting'''   

    sim_cc['canopy_cover_ns'] = sim_cc['canopy_cover_ns'] * 100
    sim_cc['canopy_cover'] = sim_cc['canopy_cover'] * 100  # convert to percent
    # save_merged_data_for_scatterplot(sim_cc, obs_cc, 'canopy_cover', 'cc')
    return sim_cc
    
def convert_biomass(sim_biomass):
    '''caculating biomass metrics and ploting '''

    sim_biomass['biomass'] = sim_biomass['biomass'] * 10  # convert unit to kg/ha
    sim_biomass['biomass_ns'] = sim_biomass['biomass_ns'] * 10
    return sim_biomass


def convert_soil_water(sim_soil_water, obs_soil_water):
    '''plot calibrated soil water results'''

    # soil_depth = soil_water_depth_calibration_test(id, sim_soil_water, obs_soil_water)
    soil_depth=0.1
    results =  selecting_soil_depth_for_calibration(sim_soil_water, obs_soil_water, soil_depth=soil_depth)
    return results


def model_calibration_results(sim_results, obs_results):
    '''
    This function is used to plot calibrated results.
    '''

    conbined_data = {'canopy_cover':[sim_results.growth, obs_results.obs_cc],
                    'biomass':[sim_results.growth, obs_results.obs_biomass],
                    'yield':[sim_results.final_stats, obs_results.obs_yield],
                    'soil_water':[sim_results.water_storage, obs_results.obs_soil_water],
                    }
    # get calibration validation year label
    cal_val_labels = obs_results.phenology_data[['ID', 'years', 'cal_label']].drop_duplicates()
    all_metrics = []    
    for calibration_type, data in list(conbined_data.items())[:-1]:
        sim_data = data[0]
        obs_data = data[1]

        if calibration_type == 'canopy_cover':
            sim_cc= convert_canopy_cover(sim_data)
            ploting_calibrated_results_dap(sim_cc, obs_data, 'canopy_cover', 'cc')
            metrics = save_merged_data_for_scatterplot(sim_cc, obs_data, 'canopy_cover', 'cc', cal_val_labels)
            all_metrics.append(metrics)

        elif calibration_type == 'biomass':
            sim_biomass= convert_biomass(sim_data)
            ploting_calibrated_results_dap(sim_biomass, obs_data, 'biomass', 'biomass(kg/ha)')
            metrics = save_merged_data_for_scatterplot(sim_biomass, obs_data, 'biomass', 'biomass(kg/ha)',cal_val_labels)
            all_metrics.append(metrics)

        elif calibration_type == 'soil_water':
            results = convert_soil_water(sim_data, obs_data)
            ploting_calibrated_results_dap(results[0], results[2], results[1], results[3])
            metrics = save_merged_data_for_scatterplot(results[0], results[2], results[1], results[3],cal_val_labels)
            all_metrics.append(metrics)

        elif calibration_type == 'yield':
            metrics = save_merged_data_for_scatterplot(sim_data, obs_data, 'Dry yield (tonne/ha)', 'yield_sta(t/ha)',cal_val_labels)
            all_metrics.append(metrics)
        else:
            pass
    # save metrics
    pd.concat(all_metrics).to_csv('../../results/calibration/metrics.csv', index=False)

if __name__ == "__main__":
    pass
