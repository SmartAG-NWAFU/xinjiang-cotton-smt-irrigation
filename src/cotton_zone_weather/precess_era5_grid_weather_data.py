import os
import pandas as pd
import numpy as np
from cbgeo import RasterConverter

# Define constants
input_folder = "../../data/grid_10km/aquacrop_inputdata/weather/2000-01-01_2021-12-31"  
output_file_path = "../../results/xinjiang_weather10km"


def calculate_gdd(tmin, tmax, t_base=10):
    """
    Calculates the Growing Degree Days (GDD) for a given day.
    :param tmin: Minimum temperature of the day
    :param tmax: Maximum temperature of the day
    :param t_base: Base temperature for GDD calculation (default is 10°C)
    :return: GDD value for the day
    """
    t_avg = (tmin + tmax) / 2
    return max(t_avg - t_base, 0)  # Use built-in max function


# Define a function to process a single file
def process_file(file_path):
    """
    Processes a single file to calculate the 22-year averages for Tmin, Tmax, Prcp, and Et0.
    :param file_path: Path to the .txt file
    :return: Dictionary containing the means
    """
    df = pd.read_csv(file_path, sep=r"\s+")
    df = df[(df["Month"] >= 4) & (df["Month"] <= 10)]
    df["GDD"] = df.apply(lambda row: calculate_gdd(row["Tmin(C)"], row["Tmax(C)"]), axis=1)

    yearly_data = df.groupby("Year")
    mean_tmin = yearly_data["Tmin(C)"].min().mean().round(2)
    mean_tmax = yearly_data["Tmax(C)"].max().mean().round(2)
    mean_tmean = (mean_tmin + mean_tmax) / 2
    mean_prcp = yearly_data["Prcp(mm)"].sum().mean().round(2)
    mean_et = yearly_data["Et0(mm)"].sum().mean().round(2)
    mean_gdd = yearly_data["GDD"].sum().mean().round(2)

    return {
        "mean_tmean": mean_tmean,
        "mean_tmin": mean_tmin,
        "mean_tmax": mean_tmax,
        "mean_prcp": mean_prcp,
        "mean_et": mean_et,
        "mean_gdd": mean_gdd
    }

# Define a function to process all files in the folder
def process_all_files(input_folder):
    """
    Processes all .txt files in the input folder.
    :param input_folder: Path to the folder containing .txt files
    :return: List of dictionaries with file ID and means
    """
    results = [
        {**process_file(os.path.join(input_folder, file_name)), "ID": os.path.splitext(file_name)[0]}
        for file_name in os.listdir(input_folder) if file_name.endswith(".txt")
    ]

    df = pd.DataFrame(results)
    os.makedirs(output_file_path, exist_ok=True)  # Use exist_ok to avoid checking existence
    output_csv_path = os.path.join(output_file_path, 'grid_units_aquacrop_weather_summary.csv')
    df.to_csv(output_csv_path, index=False)
    print(f"Summary saved to {output_csv_path}")
    return df

# Convert the results to raster
def convert_results_to_raster(results):
    cotton_units = pd.read_csv('../../data/grid_10km/xinjiang_cotton_units.csv')
    results = pd.merge(results, cotton_units, on='ID', how='left')

    value_columns = ['mean_tmean','mean_tmin', 'mean_tmax', 'mean_prcp', 'mean_et', 'mean_gdd']

    converter = RasterConverter(
        df=results,
        value_columns=value_columns,
        res=0.1,  # Assuming 10km resolution is approximately 0.11 degrees
        output_dir=output_file_path,
        lat_column='lat',
        lon_column='lon',
        crs='EPSG:4326'
    )
    converter.csv_to_tif()
  
# Define the main function
def main():
    """
    Main function to process files and save the summary to a CSV.
    """
    # Process all files in the input folder, and save output results
    # df = process_all_files(input_folder)
    df = pd.read_csv('../../results/xinjiang_weather10km/grid_units_aquacrop_weather_summary.csv')
    # Convert the results to a raster
    convert_results_to_raster(df)

# Run the script
if __name__ == "__main__":
    main()