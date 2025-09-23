# -*- coding: utf-8 -*-
# @Time    : 2025/06/14 21:12:00
# @Author  : Bin Chen
# @Description: Calculate average annual temperature, precipitation and ET0 for all locations
# @Reference: 
# @Note: 

import os
import pandas as pd
import numpy as np
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

def calculate_yearly_stats(file_path):
    """
    Calculate yearly statistics for a single location
    
    Args:
        file_path: Path to the weather data file
        
    Returns:
        DataFrame containing yearly statistics (average temperature, total precipitation, total ET0)
    """
    try:
        # Read data, skip header row
        df = pd.read_csv(file_path, skiprows=1, delim_whitespace=True,
                        names=['Day', 'Month', 'Year', 'Tmin', 'Tmax', 'Prcp', 'Et0'])
        
        # Calculate daily average temperature
        df['Tavg'] = (df['Tmax'] + df['Tmin']) / 2
        
        # Calculate yearly statistics
        yearly_stats = df.groupby('Year').agg({
            'Tavg': 'mean',    # Average temperature
            'Prcp': 'sum',     # Total precipitation
            'Et0': 'sum'       # Total ET0
        }).reset_index()
        
        return yearly_stats
    except Exception as e:
        print(f"Error processing file {file_path}: {str(e)}")
        return None

def process_files_in_parallel(file_paths):
    """
    Process a list of files in parallel and return a list of DataFrames.
    """
    results = []
    with ProcessPoolExecutor(max_workers=10) as executor:
        future_to_file = {executor.submit(calculate_yearly_stats, fp): fp for fp in file_paths}
        for future in as_completed(future_to_file):
            res = future.result()
            if res is not None:
                results.append(res)
    return results

def process_historical_data():
    """Process historical weather data (2000-2022)"""
    current_dir = Path(__file__).parent
    data_dir = current_dir.parent.parent / 'data/grid_10km/aquacrop_inputdata/weather/2000-01-01_2022-12-31'
    file_paths = list(data_dir.glob('*.txt'))
    print(f"Processing {len(file_paths)} historical files in parallel...")
    all_stats = process_files_in_parallel(file_paths)
    if not all_stats:
        print("No valid historical data was processed!")
        return None
    all_stats_df = pd.concat(all_stats, ignore_index=True)
    final_stats = all_stats_df.groupby('Year').agg({
        'Tavg': 'mean',
        'Prcp': 'mean',
        'Et0': 'mean'
    }).round(2).reset_index()
    final_stats['SSP'] = 'Historical'
    final_stats = final_stats.rename(columns={'Year': 'Years'})
    return final_stats

def process_future_data():
    """Process future climate scenarios data (2022-2076)"""
    current_dir = Path(__file__).parent
    base_dir = current_dir.parent.parent / 'data/grid_10km/aquacrop_inputdata/weather/2022-01-01_2081-12-31'
    ssp_data = {}
    for scenario_dir in base_dir.glob('ssp*'):
        ssp = scenario_dir.name.split('_')[0]
        print(f"\nProcessing scenario: {ssp}")
        file_paths = list(scenario_dir.glob('*.txt'))
        print(f"Processing {len(file_paths)} files for {ssp} in parallel...")
        scenario_stats = process_files_in_parallel(file_paths)
        if scenario_stats:
            scenario_df = pd.concat(scenario_stats, ignore_index=True)
            scenario_avg = scenario_df.groupby('Year').agg({
                'Tavg': 'mean',
                'Prcp': 'mean',
                'Et0': 'mean'
            }).round(2)
            scenario_avg['SSP'] = ssp
            scenario_avg = scenario_avg.reset_index()
            scenario_avg = scenario_avg.rename(columns={'Year': 'Years'})
            if ssp not in ssp_data:
                ssp_data[ssp] = []
            ssp_data[ssp].append(scenario_avg)
    if not ssp_data:
        print("No valid future data was processed!")
        return None
    all_scenarios = []
    for ssp, data_list in ssp_data.items():
        ssp_combined = pd.concat(data_list)
        ssp_avg = ssp_combined.groupby(['Years', 'SSP']).agg({
            'Tavg': 'mean',
            'Prcp': 'mean',
            'Et0': 'mean'
        }).round(2).reset_index()
        all_scenarios.append(ssp_avg)
    return pd.concat(all_scenarios, ignore_index=True)


def main():
    # Create output directory if it doesn't exist
    current_dir = Path(__file__).parent
    output_dir = current_dir.parent.parent / 'results/xinjiang_weather10km'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process historical data
    print("\nProcessing historical data (2000-2022)...")
    historical_stats = process_historical_data()
    
    # Process future data
    print("\nProcessing future climate scenarios (2022-2081)...")
    future_stats = process_future_data()
    
    # Combine historical and future data
    if historical_stats is not None and future_stats is not None:
        all_stats = pd.concat([historical_stats, future_stats])
        
        # Save results
        output_path = output_dir / 'weather_stats_results_2000-2081.csv'
        all_stats.to_csv(output_path)
        print(f"\nWeather statistics saved to {output_path}")
        print("\nResults preview:")
        print(all_stats)
    else:
        print("Error: Could not process all data successfully")

if __name__ == "__main__":
    main()

