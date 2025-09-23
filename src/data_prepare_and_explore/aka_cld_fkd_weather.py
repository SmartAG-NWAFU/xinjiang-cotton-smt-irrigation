import pandas as pd
from aquacropeto import convert, fao
import os

def process_nan_values(dataframe):
    """
    Replace values greater than 999900 with NaN and forward fill NaN values.
    """
    dataframe.iloc[:, 1:] = dataframe.iloc[:, 1:].apply(lambda x: x.where(x <= 999900, pd.NA))
    dataframe.fillna(method='ffill', inplace=True)
    return dataframe

def create_date_column(dataframe):
    """
    Create a 'date' column from 'Year', 'Mon', and 'Day' columns.
    """
    dataframe['date'] = pd.to_datetime(dataframe['Year'].astype(str) + '-' + dataframe['Mon'].astype(str) + '-' + dataframe['Day'].astype(str),
                                        format='%Y-%m-%d')
    return dataframe

def add_latitudes_and_elevation(dataframe):
    """
    Add 'Latitudes' and 'Elevation' columns based on 'ID' column.
    """
    latitudes = {'CLD': 37, 'FKD': 45, 'AKA': 40}
    elevation = {'CLD': 1318, 'FKD': 450, 'AKA': 1030}
    dataframe['Elevation'] = dataframe['ID'].map(elevation)
    dataframe['Latitudes'] = dataframe['ID'].map(latitudes)
    return dataframe

def select_useful_data(dataframe):
    """
    Select and rename useful columns.
    """
    order = ["ID", "Latitudes", "Elevation", "date", "MaxTemp", "MinTemp", "Precip", "Humidity", "Windspeed", "Sunhours"]
    dataframe = dataframe[order]
    dataframe.columns = ["ID", "lat", "ele", "date", "tmax", "tmin", "prec", "humidity", "wind", "sunhours"]
    return dataframe

def add_day_of_year(data):
    """
    Add 'doy' (day of year) column based on 'date' column.
    """
    data['date'] = pd.to_datetime(data['date'], format='%Y%m%d')
    data['doy'] = data['date'].dt.day_of_year
    return data

def cover_prec_small_values(data):
    """
    Replace values in 'prec' column that are less than 1 with 0.
    """
    data['prec'] = data['prec'].apply(lambda x: 0 if x < 1 else x)
    return data

def fao_et0(data):
    """
    Calculate FAO ET0 and add it to the dataframe.
    """
    ext_rad = fao.et_rad(convert.deg2rad(data.lat), fao.sol_dec(data.doy),
                        fao.sunset_hour_angle(convert.deg2rad(data.lat),
                        fao.sol_dec(data.doy)),
                        fao.inv_rel_dist_earth_sun(data.doy))

    netrad = fao.sol_rad_from_sun_hours(daylight_hours=fao.daylight_hours(fao.sunset_hour_angle(convert.deg2rad(data.lat),
                            fao.sol_dec(data.doy))),
                            sunshine_hours=data.sunhours, et_rad=ext_rad)
    net_sw = fao.net_in_sol_rad(netrad)
    cl_sky_rad = fao.cs_rad(data.ele, ext_rad)
    net_lw_rad = fao.net_out_lw_rad(data.tmin, data.tmax, netrad, cl_sky_rad, fao.avp_from_tmin(data.tmin))
    net_radiation = fao.net_rad(net_sw, net_lw_rad)

    tmean = data.tmin * 0.5 + data.tmax * 0.5
    ws = fao.wind_speed_2m(data.wind, 10)

    avp = fao.avp_from_rhmean(svp_tmax=fao.svp_from_t(data.tmax), svp_tmin=fao.svp_from_t(data.tmin),
                          rh_mean=data.humidity)  # humidity unit is percent
    
    svp = fao.mean_svp(data.tmin, data.tmax)
    delta = fao.delta_svp(tmean)

    psy = fao.psy_const_of_psychrometer(1, fao.atm_pressure(data.ele))

    faopm = fao.fao56_penman_monteith(net_radiation, convert.celsius2kelvin(tmean), ws, svp, avp, delta, psy)

    data["et0"] = faopm
    data.et0 = data.et0.clip(0.1)

    # Change time format
    data['Year'] = data['date'].dt.year
    data['Month'] = data['date'].dt.month
    data['Day'] = data['date'].dt.day
    data = data[["ID", "Day", "Month", "Year", "tmin", "tmax", "prec", "et0"]]
    # Rename columns
    data.columns = ["ID", "Day", "Month", "Year", "Tmin(C)", "Tmax(C)", "Prcp(mm)", "Et0(mm)"]
    return data

def main():
    try:
        df = pd.read_csv('../data/station_weather/fkd_cld_aks_met_data_2000-2020.csv')
    except FileNotFoundError:
        print("Error: The specified file was not found.")
        return

    # Process the dataframe
    df = process_nan_values(df)
    df = create_date_column(df)
    df = add_latitudes_and_elevation(df)
    df = select_useful_data(df)
    df = add_day_of_year(df)

    # Calculate et0
    df = cover_prec_small_values(df)
    df = fao_et0(df)

    # Split the dataframe by 'ID' column and save each subset to a separate CSV file
    unique_ids = df['ID'].unique()
    output_dir = '../data/station_weather/'

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    for unique_id in unique_ids:
        subset_df = df[df['ID'] == unique_id]
        subset_df = subset_df.drop(columns=['ID'])
        output_file = os.path.join(output_dir, f'station_{unique_id}.csv')
        subset_df.to_csv(output_file.replace('.csv', '.txt'), index=False, sep=' ')

if __name__ == "__main__":
    main()