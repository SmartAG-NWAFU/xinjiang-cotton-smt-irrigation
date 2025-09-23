"""
AquaCrop meteorological data preprocessing module
Convert raw meteorological data to the format required by AquaCrop
"""

import logging
import glob
import os
from pathlib import Path
from typing import List, Tuple
import pandas as pd
from aquacropeto import convert, fao
from multiprocessing import Pool
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime


class AquaCropWeatherPreparer:
    """Meteorological data preprocessing class to convert raw data to the format required by AquaCrop"""
    
    def __init__(
        self, 
        input_dir: str,
        elevation_file: str, 
        output_dir: str,
        num_workers: int = 4
    ) -> None:
        """
        Initialize the preprocessing object
        
        :param input_dir: Path to raw data files (e.g. '../../data/sites/2000-01-01_2021-12-31/*.csv')
        :param elevation_file: Path to elevation data file
        :param output_dir: Path to output directory
        :param num_workers: Number of parallel worker processes
        """
        self.input_files = self.create_input_fiels(input_dir)
        self.elevation_df = pd.read_csv(elevation_file)
        self.output_dir = Path(output_dir)
        self.num_workers = num_workers
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    def create_input_fiels(self, input_dir):
        """Get all CSV files from the input directory using glob"""
        csv_pattern = os.path.join(input_dir, "*.csv")
        output_list = glob.glob(csv_pattern)
        return sorted(output_list)  # Sort to ensure consistent order

    def process_all_files(self) -> None:
        """Process all input files"""
        args = [
            (file_path, self.elevation_df, self._extract_site_id(file_path), self.output_dir)
            for file_path in self.input_files
        ]
        
        with Pool(processes=self.num_workers) as pool:
            pool.map(self._process_single_file, args)

    @staticmethod
    def _extract_site_id(file_path: str) -> str:
        """Extract site ID from file path (file name without extension)"""
        return Path(file_path).stem

    @classmethod
    def _transform_data(
        cls, 
        df: pd.DataFrame, 
        elevation_data: pd.DataFrame,
        site_id: str
    ) -> pd.DataFrame:
        """Perform the complete data transformation process"""
        # Basic transformations
        df = cls._basic_transformations(df)
        # print(df.head())
        
        # Add geographic information
        df = cls._add_geo_info(df, elevation_data, site_id)
        
        # Calculate ET0
        df = cls._calculate_et0(df)
        
        return df[["Day", "Month", "Year", "Tmin(C)", "Tmax(C)", "Prcp(mm)", "Et0(mm)"]]

    @staticmethod
    def _basic_transformations(df: pd.DataFrame) -> pd.DataFrame:
        """Perform basic data transformations"""
        # Rename columns
        df.columns = ['prec', 'tmin', 'tmax', 'netrad', 'uwind', 'vwind', 'date']
        
        # Calculate wind speed
        df['wind'] = (df['uwind']**2 + df['vwind']**2)**0.5
        
        # Unit conversions
        df['netrad'] *= 1e-6  # J/m² -> MJ/m²
        df['prec'] *= 1000    # m -> mm
        
        # Date processing
        df['date'] = pd.to_datetime(df['date'], format='%Y%m%d')
        df['doy'] = df['date'].dt.dayofyear
        
        # Handle trace precipitation
        df['prec'] = df['prec'].where(df['prec'] >= 1, 0)
        
        # Temperature unit conversion
        df['tmin'] = convert.kelvin2celsius(df.tmin)
        df['tmax'] = convert.kelvin2celsius(df.tmax)
        
        return df

    @staticmethod
    def _add_geo_info(
        df: pd.DataFrame,
        elevation_data: pd.DataFrame,
        site_id: str
    ) -> pd.DataFrame:
        """Add geographic information"""
        # site_id = int(site_id)
        # elevation_data['ID'] = elevation_data['ID'].astype(int)
        try:
            geo_info = elevation_data[elevation_data.ID == int(site_id)].iloc[0]
            df['lat'] = geo_info['lat']
            df['ele'] = geo_info['elevation']
            return df
        except IndexError:
            raise ValueError(f"Cannot find elevation data for site {site_id}")

    @staticmethod
    def _calculate_et0(df: pd.DataFrame) -> pd.DataFrame:
        """Calculate FAO-56 Penman-Monteith reference evapotranspiration"""
        # Solar radiation calculation
        net_sw = fao.net_in_sol_rad(df.netrad)
        ext_rad = fao.et_rad(
            convert.deg2rad(df.lat),
            fao.sol_dec(df.doy),
            fao.sunset_hour_angle(convert.deg2rad(df.lat), fao.sol_dec(df.doy)),
            fao.inv_rel_dist_earth_sun(df.doy)
        )
        
        # Longwave radiation calculation
        cl_sky_rad = fao.cs_rad(df.ele, ext_rad)
        net_lw_rad = fao.net_out_lw_rad(
            df.tmin, df.tmax, df.netrad, cl_sky_rad, 
            fao.avp_from_tmin(df.tmin)
        )
        
        # Net radiation calculation
        net_radiation = fao.net_rad(net_sw, net_lw_rad)
        
        # Meteorological parameter calculation
        tmean = df.tmin*0.5 + df.tmax*0.5
        ws = fao.wind_speed_2m(df.wind, 10)
        avp = fao.avp_from_tmin(df.tmin)
        svp = fao.mean_svp(df.tmin, df.tmax)
        delta = fao.delta_svp(tmean)
        psy = fao.psy_const_of_psychrometer(1, fao.atm_pressure(df.ele))
        
        # FAO Penman-Monteith calculation
        et0 = fao.fao56_penman_monteith(
            net_radiation, 
            convert.celsius2kelvin(tmean),
            ws, svp, avp, delta, psy
        )
        
        # Process results
        df["et0"] = et0.clip(0.1)
        df['Year'] = df['date'].dt.year
        df['Month'] = df['date'].dt.month
        df['Day'] = df['date'].dt.day
        
        return df.rename(columns={
            "tmin": "Tmin(C)",
            "tmax": "Tmax(C)",
            "prec": "Prcp(mm)",
            "et0": "Et0(mm)"
        })
    
    @staticmethod
    def _process_single_file(args: Tuple[str, pd.DataFrame, str, Path]) -> None:
        """Process a single file (called by multiple processes)"""
        file_path, elevation_data, site_id, output_dir = args
        
        try:
            df = pd.read_csv(file_path)
            # Use site ID as output file name
            output_path = output_dir / f"{site_id}.txt"
            if output_path.exists():
                logging.info(f"Skipping existing file for site {site_id}: {output_path}")
                return
            df = AquaCropWeatherPreparer._transform_data(df, elevation_data, site_id)
            df.to_csv(output_path, index=False, sep=" ")
            logging.info(f"Successfully processed file for site {site_id}: {file_path}")
            
        except Exception as e:
            logging.error(f"Failed to process file for site {site_id}: {str(e)}")
            raise

class AquaCropWeatherPreparerFuture(AquaCropWeatherPreparer):
    """Meteorological data preprocessing class for future weather data"""
    def __init__(
        self, 
        input_dir: str,
        elevation_file: str, 
        output_dir: str,
        num_workers: int = 1
    ) -> None:
        """
        Initialize the preprocessing object for future weather data
        
        :param input_dir: Path to raw data files (e.g. '../../data/sites/2000-01-01_2021-12-31/*.csv')
        :param elevation_file: Path to elevation data file
        :param output_dir: Path to output directory
        :param num_workers: Number of parallel worker processes
        """
        super().__init__(input_dir, elevation_file, output_dir, num_workers)

    @staticmethod
    def _basic_transformations(df: pd.DataFrame) -> pd.DataFrame:
        """Perform basic data transformations"""
        # Rename columns
        df.columns = ['prec', 'rlds', 'rsds', 'wind', 'tas', 'tmax', 'tmin', 'system_index']
        
        df['rlds'] *= 0.0864   # W/m² → MJ/m²/day
        df['rsds'] *= 0.0864   # W/m² → MJ/m²/day
        df['prec'] *= 86400    # kg/m^2/s -> mm
    
        # Handle trace precipitation
        df['prec'] = df['prec'].where(df['prec'] >= 1, 0)
        
        # Temperature unit conversion
        df['tmin'] = convert.kelvin2celsius(df.tmin)
        df['tmax'] = convert.kelvin2celsius(df.tmax)
        
        return df

    @staticmethod
    def _add_geo_info(
        df: pd.DataFrame,
        elevation_data: pd.DataFrame,
        site_id: str
    ) -> pd.DataFrame:
        """Add geographic information"""
        try:
            geo_info = elevation_data[elevation_data.ID == int(site_id)].iloc[0]
            df['lat'] = geo_info['lat']
            df['ele'] = geo_info['elevation']
            return df
        except IndexError:
            raise ValueError(f"Cannot find elevation data for site {site_id}")

    @staticmethod
    def _calculate_et0(df: pd.DataFrame) -> pd.DataFrame:
        """Calculate FAO-56 Penman-Monteith reference evapotranspiration"""
        # Solar radiation calculation
        net_sw = fao.net_in_sol_rad(df.rsds)

        # 长波辐射计算（优化版本）
        tmax_kelvin = df.tmax + 273.15
        tmin_kelvin = df.tmin + 273.15
        surface_emit = fao.STEFAN_BOLTZMANN_CONSTANT * ((tmax_kelvin**4 + tmin_kelvin**4)/2)
        net_lw = surface_emit - df.rlds  # 地面发射 - 下行长波
        # Net radiation calculation
        net_radiation = fao.net_rad(net_sw, net_lw)
        
        # Meteorological parameter calculation
        tmean = df.tmin*0.5 + df.tmax*0.5
        ws = fao.wind_speed_2m(df.wind, 10)
        avp = fao.avp_from_tmin(df.tmin)
        svp = fao.mean_svp(df.tmin, df.tmax)
        delta = fao.delta_svp(tmean)
        psy = fao.psy_const_of_psychrometer(1, fao.atm_pressure(df.ele))
        
        # FAO Penman-Monteith calculation
        et0 = fao.fao56_penman_monteith(
            net_radiation, 
            convert.celsius2kelvin(tmean),
            ws, svp, avp, delta, psy
        )
        
        # Process results
        df["et0"] = et0.clip(0.1)
        return df.rename(columns={
            "tmin": "Tmin(C)",
            "tmax": "Tmax(C)",
            "prec": "Prcp(mm)",
            "et0": "Et0(mm)"
        })
    
    @staticmethod
    def _split_system_index(df):
        # 拆分 system_index 成两列
        split_cols = df['system_index'].str.rsplit('_', n=1, expand=True)
        df['ssp_model'] = split_cols[0]
        df['date'] = split_cols[1]
        df['date'] = pd.to_datetime(df['date'], format='%Y%m%d')
        df['Year'] = df['date'].dt.year
        df['Month'] = df['date'].dt.month
        df['Day'] = df['date'].dt.day
        return df

    @classmethod
    def _transform_data(
        cls, 
        df: pd.DataFrame, 
        elevation_data: pd.DataFrame,
        site_id: str
    ) -> pd.DataFrame:
        """Perform the complete data transformation process"""
        # Basic transformations
        df = cls._basic_transformations(df)
        # Add geographic information
        df = cls._add_geo_info(df, elevation_data, site_id)
        
        # Calculate ET0
        df = cls._calculate_et0(df)

        # Split index 
        df = cls._split_system_index(df)

        return df[["Day", "Month", "Year", "Tmin(C)", "Tmax(C)", "Prcp(mm)", "Et0(mm)","ssp_model"]]
    
    @staticmethod
    def _process_single_file(args: Tuple[str, pd.DataFrame, str, Path]) -> None:
        """Process a single file (called by multiple processes)"""
        file_path, elevation_data, site_id, output_dir = args
        
        try:
            df = pd.read_csv(file_path)
            df = AquaCropWeatherPreparerFuture._transform_data(df, elevation_data, site_id)
            # Split by ssp_model
            for ssp_model, group_df in df.groupby('ssp_model'):
                # create output directory for each ssp_model
                model_dir = output_dir / ssp_model
                model_dir.mkdir(parents=True, exist_ok=True)
                group_df = group_df.drop(columns=['ssp_model'])
                # save to file
                output_path = model_dir / f"{site_id}.txt"
                group_df.to_csv(output_path, index=False, sep=" ")
                print(f"Saved: {output_path}")

        except Exception as e:
            logging.error(f"Failed to process file for site {site_id}: {str(e)}")
            raise

    def process_all_files(self) -> None:
        """Process all input files"""
        args = [
            (file_path, self.elevation_df, self._extract_site_id(file_path), self.output_dir)
            for file_path in self.input_files
        ]
        
        with Pool(processes=min(self.num_workers, len(self.input_files))) as pool:
            pool.map(self._process_single_file, args)


def sites_weather_data_preparation():
    """Prepare weather data for all sites"""
    processor = AquaCropWeatherPreparer(
        input_dir='../../data/sites/2000-01-01_2022-12-31/',
        elevation_file='../../data/sites/elevation.csv',
        output_dir='../../data/sites/aquacrop_inputdata/weather',
        num_workers=4
    )
    processor.process_all_files()

def grids_history_weather_data_preparation():
    """Prepare weather data for all grids"""
    processor = AquaCropWeatherPreparer(
        input_dir='../../data/grid_10km/weather/2000-01-01_2022-12-31/',
        elevation_file='../../data/grid_10km/elevation.csv',
        output_dir='../../data/grid_10km/aquacrop_inputdata/weather/2000-01-01_2022-12-31',
        num_workers=61
    )
    processor.process_all_files()

def grids_future_weather_data_preparation():
    """Prepare weather data for all grids"""
    processor = AquaCropWeatherPreparerFuture(
        input_dir='../../data/grid_10km/weather/2022-01-01_2081-12-31_summary',
        elevation_file='../../data/grid_10km/elevation.csv',
        output_dir='../../data/grid_10km/aquacrop_inputdata/weather/2022-01-01_2081-12-31',
        num_workers=61
    )
    processor.process_all_files()

def test_results(file_path, year):
    # 读取数据
    df = pd.read_csv(file_path, delim_whitespace=True)

    # 构造完整日期列
    df['Date'] = pd.to_datetime(df[['Year', 'Month', 'Day']])
    df = df[df['Year'] == year]

    # 设置日期为索引
    df.set_index('Date', inplace=True)

    # 创建图形和两个y轴
    fig, ax1 = plt.subplots(figsize=(16, 8))

    # 画 Tmin 和 Tmax 折线图（左 y 轴）
    ax1.plot(df.index, df['Tmin(C)'], color='blue', label='Tmin (°C)', linewidth=1)
    ax1.plot(df.index, df['Tmax(C)'], color='red', label='Tmax (°C)', linewidth=1)
    ax1.set_ylabel('Temperature (°C)', fontsize=12)
    ax1.set_xlabel('Date', fontsize=12)
    ax1.tick_params(axis='y', labelcolor='black')

    # 设置主 x 轴为月份显示
    ax1.xaxis.set_major_locator(mdates.MonthLocator())
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b'))

    # 创建第二个 y 轴
    ax2 = ax1.twinx()

    # 画 Et0 折线图
    ax2.plot(df.index, df['Et0(mm)'], color='green', label='Et0 (mm)', linestyle='--', linewidth=1.2)

    # 画 Prcp 柱状图
    ax2.bar(df.index, df['Prcp(mm)'], color='skyblue', label='Prcp (mm)', alpha=0.4, width=1)

    ax2.set_ylabel('Et0 / Precipitation (mm)', fontsize=12)
    ax2.tick_params(axis='y', labelcolor='black')

    # 合并图例
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=10)

    # 设置标题
    title_label = f'Weather Data Visualization - {year}'
    plt.title(title_label, fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.5)
    # 自动排版
    fig.autofmt_xdate()
    plt.tight_layout()
    site_id = os.path.splitext(os.path.basename(file_path))[0]
    scenario_dir = os.path.basename(os.path.dirname(file_path)) 
    # 构造文件名
    save_name = f'{scenario_dir}_{title_label}_site{site_id}.png'
    # 保存图片
    plt.savefig(f'../../figs/test/{save_name}')

def test():
    # test_results('../../data/grid_10km/aquacrop_inputdata/weather/2000-01-01_2021-12-31/200.txt',2000)
    test_results('../../data/grid_10km/aquacrop_inputdata/weather/2022-01-01_2076-12-31/ssp245_BCC-CSM2-MR/200.txt', 2070)
    test_results('../../data/grids/aquacrop_inputdata/weather/2022-01-01_2076-12-31/ssp585_BCC-CSM2-MR/200.txt',2070)

def main() -> None:
    """Main function"""
    # sites_weather_data_preparation()
    grids_history_weather_data_preparation()
    # grids_future_weather_data_preparation()


if __name__ == "__main__":
    main()
    # test()