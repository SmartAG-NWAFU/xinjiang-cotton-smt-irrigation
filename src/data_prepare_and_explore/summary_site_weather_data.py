import numpy as np
import pandas as pd
import glob
import os

src_dir = os.path.abspath(os.path.dirname(__file__))

files_dir = glob.glob(os.path.join(src_dir, '../../data/sites/aquacrop_inputdata/weather', '*.txt'))

def summary_site_weather_data():
    results = []
    for file_path in files_dir:
        # Extract site name from file name
        site_name = os.path.splitext(os.path.basename(file_path))[0]
        # Read the file
        df = pd.read_csv(file_path, delim_whitespace=True)
        # Filter years 2000-2022
        df = df[(df['Year'] >= 2000) & (df['Year'] <= 2022)]
        # Calculate average temperature (mean of Tmin and Tmax)
        df['Tavg'] = (df['Tmin(C)'] + df['Tmax(C)']) / 2
        avg_temp = round(df['Tavg'].mean(), 2)
        # Calculate annual precipitation
        annual_precip = df.groupby('Year')['Prcp(mm)'].sum().reset_index()
        annual_precip_mean = round(annual_precip['Prcp(mm)'].mean(),2)
        # Store results
        results.append({
            'site': site_name,
            'avg_temp_2000_2022': avg_temp,
            'annual_precip_2000_2022': annual_precip_mean
        })
    # Concatenate results into a DataFrame
    result_df = pd.DataFrame(results)
    return result_df

if __name__ == '__main__':
    result_df = summary_site_weather_data()
    result_df.to_csv(os.path.join(src_dir,'../../results/xinjiang_weather10km/11_sites_weather_summary.csv'))