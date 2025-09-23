"""Compute correlations for weather and soil features across periods.

Renamed from `caculate_weather_soil_correlation.py` to fix a typo in the
filename (calculate).
"""

import os
import pandas as pd
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from joblib import Parallel, delayed

try:
    # Prefer shared implementation from model
    from model.metrics import calculate_gdd
except Exception:
    # Fallback: add project root to sys.path if needed (script execution)
    import sys, os as _os
    _root = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), '..', '..'))
    if _root not in sys.path:
        sys.path.append(_root)
    from model.metrics import calculate_gdd

# Input/Output paths
historical_input_folder = "../../data/grid_10km/aquacrop_inputdata/weather/2000-01-01_2022-12-31"
future_input_folder = "../../data/grid_10km/aquacrop_inputdata/weather/2022-01-01_2081-12-31"
output_file_path = "../../results/xinjiang_weather10km"
historical_output_csv_path = os.path.join(output_file_path, 'grid_weather_2000-2022_correlation.csv')
future_output_csv_path = os.path.join(output_file_path, 'grid_weather_2022-2081_correlation.csv')
weather_output_csv_path = os.path.join(output_file_path, 'grid_weather_2000-2081_correlation.csv')
soil_input_folder = "../../data/grid_10km/aquacrop_inputdata/soil/soil.csv"
combined_output_csv_path = os.path.join(output_file_path, 'grid_weather_soil_2000-2081_correlation.csv')


# calculate_gdd now imported from model.metrics

def historical_single_file(file_path):
    try:
        df = pd.read_csv(file_path, sep=r"\s+")
        df = df[(df["Month"] >= 4) & (df["Month"] <= 10)]
        df["GDD"] = df.apply(lambda row: calculate_gdd(row["Tmin(C)"], row["Tmax(C)"]), axis=1)
        df['periods'] = 'history'
        df['ssps'] = 'hitorical'
        df['gcms'] = 'historical'
        yearly_data = df.groupby(["periods", "ssps", "gcms", "Year"])

        mean_tmin = yearly_data["Tmin(C)"].min().mean().round(2)
        mean_tmax = yearly_data["Tmax(C)"].max().mean().round(2)
        mean_tmean = (mean_tmin + mean_tmax) / 2
        mean_prcp = yearly_data["Prcp(mm)"].sum().mean().round(2)
        mean_et = yearly_data["Et0(mm)"].sum().mean().round(2)
        mean_gdd = yearly_data["GDD"].sum().mean().round(2)

        return {
            "ID": os.path.splitext(os.path.basename(file_path))[0],
            "periods": "history",
            "ssps" : "historical",
            "gcms" : "historical",
            "mean_tmean": mean_tmean,
            "mean_tmin": mean_tmin,
            "mean_tmax": mean_tmax,
            "mean_prcp": mean_prcp,
            "mean_et": mean_et,
            "mean_gdd": mean_gdd
        }

    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
        return None

def future_single_file(file_path):
        """Process a single future weather file and return a wide-format row
        per ID that contains average metrics for 2040s/2070s periods.
        """
        try:
            df = pd.read_csv(file_path, sep=r"\s+")

            # Keep April–October only
            df = df[(df["Month"] >= 4) & (df["Month"] <= 10)]

            # Classify into decadal periods
            def classify_decade(year):
                if 2031 <= year <= 2050:
                    return '2040s'
                elif 2061 <= year <= 2080:
                    return '2070s'
                else:
                    return 'other'

            df['periods'] = df['Year'].apply(classify_decade)
            df = df[df['periods'].isin(['2040s', '2070s'])]
            if df.empty:
                return pd.DataFrame()  # Skip files without target periods

            # Compute GDD
            df["GDD"] = df.apply(lambda row: calculate_gdd(row["Tmin(C)"], row["Tmax(C)"]), axis=1)

            # Aggregate by year
            yearly = df.groupby(["periods", "Year"]).agg({
                "Tmin(C)": "min",
                "Tmax(C)": "max",
                "Prcp(mm)": "sum",
                "Et0(mm)": "sum",
                "GDD": "sum"
            }).reset_index()

            # Compute period means
            summary = yearly.groupby("periods").agg({
                "Tmin(C)": "mean",
                "Tmax(C)": "mean",
                "Prcp(mm)": "mean",
                "Et0(mm)": "mean",
                "GDD": "mean"
            }).round(2).reset_index()

            # Add mean_tmean
            summary["mean_tmean"] = ((summary["Tmin(C)"] + summary["Tmax(C)"]) / 2).round(2)

            # Add ID
            summary["ID"] = os.path.splitext(os.path.basename(file_path))[0]

            # Rename to standard column names
            summary = summary.rename(columns={
                "Tmin(C)": "mean_tmin",
                "Tmax(C)": "mean_tmax",
                "Prcp(mm)": "mean_prcp",
                "Et0(mm)": "mean_et",
                "GDD": "mean_gdd"
            })

            # Wide-format pivot
            df_wide = summary.pivot(index="ID", columns="periods")
            df_wide.columns = [f"{period}_{metric}" for metric, period in df_wide.columns]
            df_wide = df_wide.reset_index()

            return df_wide

        except Exception as e:
            print(f"Error processing file {file_path}: {e}")
            return pd.DataFrame()


def process_future_file(future_input_folder):
    """
    Processes a single file to calculate the future weather averages for Tmin, Tmax, Prcp, and Et0.
    :param file_path: Path to the .txt file
    :return: Dictionary containing the means
    """

    all_results = []
    base_dir = Path(future_input_folder)

    for scenario_dir in list(base_dir.iterdir()):
        if not scenario_dir.is_dir():
            continue

        print(f"Processing scenario: {scenario_dir.name}")
        file_paths = list(scenario_dir.glob('*.txt'))

        # Limit number of CPU cores (up to 128 here)
        parallel_results = Parallel(n_jobs=128, backend="loky")(
            delayed(future_single_file)(fp) for fp in file_paths
        )

        valid_results = [df for df in parallel_results if df is not None]
        if valid_results:
            scenario_df = pd.concat(valid_results, ignore_index=True)
            # Split folder name, e.g., "ssp245_BCC-CSM2-MR"
            ssp, gcm = scenario_dir.name.split("_", 1)
            scenario_df["ssps"] = ssp
            scenario_df["gcms"] = gcm
            all_results.append(scenario_df)

    if not all_results:
        print("No valid scenario results.")
        return pd.DataFrame()

    future_df = pd.concat(all_results, ignore_index=True)
    future_df = future_df.groupby(["ID","ssps","gcms"]).mean(numeric_only=True).reset_index()
    # Convert to long format
    df_long = future_df.melt(id_vars=["ID","ssps","gcms"], var_name="metric", value_name="value")
    df_long[["periods", "metric_name"]] = df_long["metric"].str.extract(r"(\d{4}s)_(.+)")
    df_pivoted = df_long.pivot(index=["ID", "ssps","gcms","periods"], columns="metric_name", values="value").reset_index()

    return df_pivoted


def process_historical_file(input_folder):
    file_paths = [os.path.join(input_folder, f) for f in os.listdir(input_folder) if f.endswith(".txt")]

    results = []
    with ProcessPoolExecutor() as executor:
        future_to_file = {executor.submit(historical_single_file, file_path): file_path for file_path in file_paths}
        for future in as_completed(future_to_file):
            result = future.result()
            if result is not None:
                results.append(result)

    df = pd.DataFrame(results)

    return df

def process_soil_file(soil_input_folder):
    df = pd.read_csv(soil_input_folder)
    df = df.groupby('ID')[['sand', 'silt', 'clay','soc','bdod']].mean().reset_index()
    df["ID"] = df["ID"].astype(str)
    return df

def combine_historical_and_future_data(historical_df, future_df_pivoted):
    combined_df = pd.concat([historical_df, future_df_pivoted], axis=0)
    combined_df["ID"] = combined_df["ID"].astype(str)
    combined_df.to_csv(weather_output_csv_path, index=False)
    print(f"Saved to {weather_output_csv_path}")
    return combined_df


def main():
    historical_df = process_historical_file(historical_input_folder)
    future_df_pivoted = process_future_file(future_input_folder)
    # print(future_df_pivoted)
    combined_weather_df = combine_historical_and_future_data(historical_df, future_df_pivoted)
    combined_weather_df = pd.read_csv(weather_output_csv_path)
    soil_df = process_soil_file(soil_input_folder)
    combined_weather_df['ID'] = combined_weather_df['ID'].astype(str)
    soil_df['ID'] = soil_df['ID'].astype(str)
    combined_weather_soil_df = pd.merge(combined_weather_df, soil_df, on="ID", how="left")
    combined_weather_soil_df.to_csv(combined_output_csv_path, index=False)
    print(f"Saved to {combined_output_csv_path}")

if __name__ == "__main__":
    main()
