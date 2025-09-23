import pandas as pd
import numpy as np
import math
import glob
import matplotlib.pyplot as plt
from typing import Tuple
import sys
import re
import os
from sklearn.linear_model import LinearRegression
import statsmodels.formula.api as smf
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.decomposition import PCA
import statsmodels.formula.api as smf
import warnings
warnings.filterwarnings("ignore")


# get current script absolute path
current_dir = os.path.dirname(os.path.abspath(__file__))
# get project root path (assume model and utils are in the same directory)
project_root = os.path.dirname(current_dir)
# add project root to sys.path
sys.path.append(project_root)
from utils.csv_convert_raster import RasterConverter

class ResultsAnalysis:
    def __init__(self, result_dir, output_dir):
        self.results_dir = result_dir
        self.output_dir = output_dir

    def _load_data(self, filepath):
        return pd.read_csv(filepath)
    
    def _calculate_mean_values(self, df):

        grouped_df = df.groupby('scenario_params')[['Dry yield (tonne/ha)', 'Seasonal irrigation (mm)']].mean().round(2)
        return grouped_df.reset_index()
    
    def _caculate_iwp(self, df):
        df['iwp(kg/m3)'] = (df['Dry yield (tonne/ha)'] * 1000) / (df['Seasonal irrigation (mm)'] * 10) # kg/m3
        return df
    
    def _save_output_results(self, output_df):
        output_df.to_csv(f'{self.output_dir}/statistic_results.csv', index=False)

    # Convert the results to raster
    @staticmethod
    def _convert_results_to_raster(results, columns, output_dir):
        cotton_units = pd.read_csv(os.path.join(project_root, 'data/grid_10km/xinjiang_cotton_units.csv'))
        results = pd.merge(results, cotton_units, on='ID', how='left')
        # convert to wide table
        df_wide = results.pivot(
            index=['ID','lon','lat'],
            columns='scenario_params',
            values=columns
        )
        # merge multi-level column indices
        df_wide.columns = [f'{metric}_{scenario}' for metric, scenario in df_wide.columns]
        # reset index and sort columns
        df_wide = df_wide.reset_index()
        scenarios = list(results['scenario_params'].unique())
        ordered_columns = ['ID','lon','lat'] + [f'{metric}_{scen}' for scen in scenarios 
                                for metric in columns]
        df_wide = df_wide[ordered_columns]
        # format output (keep two decimal places)
        pd.options.display.float_format = '{:.2f}'.format
        value_columns = df_wide.columns[3:]
        converter = RasterConverter(
            df=df_wide,
            value_columns=value_columns,
            res=0.1,
            output_dir= output_dir,
            lat_column='lat',
            lon_column='lon',
            crs='EPSG:4326'
        )
        converter.csv_to_tif()

    def _analyze_data(self):
        filepaths = sorted(glob.glob(f'{self.results_dir}/[0-9]*.csv'))
        output = [] 
        for filepath in filepaths:
            df = self._load_data(filepath)
            mean_values = self._calculate_mean_values(df)
            mean_values.insert(0, 'ID', int(os.path.basename(filepath).split('.')[0]))
            output.append(mean_values)
        
        output_df = pd.concat(output, ignore_index=True)
        output_df = self._caculate_iwp(output_df)
        convert_columns = ['Dry yield (tonne/ha)', 'Seasonal irrigation (mm)', 'iwp(kg/m3)']
        ResultsAnalysis._convert_results_to_raster(output_df, convert_columns, self.output_dir)
        self._save_output_results(output_df)

    def plot_yield_scenarios(self, df, id):
        """create yield management scenario comparison plot for a single site
        
        Args:
            data (pd.DataFrame): site data
            station_id (str): site ID, used to save image
        """
        scenarios = np.unique(df['man_scen'])
        num_scenarios = len(scenarios)
        
        # calculate the number of rows and columns of the subplot (maximum 3 columns)
        cols = min(3, num_scenarios)  # maximum 3 columns
        rows = math.ceil(num_scenarios / cols)  # calculate the number of rows needed
        
        # create subplot layout
        fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows), squeeze=False)
        
        # iterate over each management scenario and plot the subplot
        for idx, scenario in enumerate(scenarios):
            row, col = divmod(idx, cols)
            ax = axes[row, col]
            
            # filter data
            scenario_data = df[df['man_scen'] == scenario]
            scenario_data['Year'] = pd.to_datetime(scenario_data['Harvest Date (YYYY/MM/DD)']).dt.year
            
            # plot on the corresponding subplot
            for variety in scenario_data['varieties'].unique():
                variety_data = scenario_data[scenario_data['varieties'] == variety]
                ax.plot(variety_data['Year'], 
                        variety_data['Dry yield (tonne/ha)'], 
                        # variety_data['Yield potential (tonne/ha)'],
                        marker='o', 
                        label=variety)
            
            # set subplot attributes
            ax.set_xlabel('Year')
            ax.set_ylabel('Dry yield (tonne/ha)')
            ax.set_title(f'Management Scenario: {scenario}')
            ax.legend()
            ax.grid(True)
        
        # remove blank subplot
        for i in range(num_scenarios, rows * cols):
            fig.delaxes(axes[i // cols, i % cols])
        
        # set total title
        plt.suptitle(f'Cotton Dry Yield Over Years - Station {id}', fontsize=16)
        
        # adjust subplot spacing
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        
        # save image
        save_dir = '../figs/simulation'
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(f'{save_dir}/{id}_yield_management_scenarios.png', 
                    dpi=300, bbox_inches='tight')
        plt.close()

class FutureResultsAnalysis(ResultsAnalysis):
    '''
    init FutureResultsAnalysis
    '''
    def __init__(self, result_dir, output_dir, fd_name):
        super().__init__(result_dir, output_dir)
        self.times = ['2040s', '2070s']
        self.gcm = None
        self.ssp = None
        self._spilt_gcm_ssp(fd_name)

    def _spilt_gcm_ssp(self, fd_name):
        self.ssp, self.gcm = fd_name.split('_', 1)

    def _calculate_mean_values(self, df):
        df['year'] = pd.to_datetime(df['Harvest Date (YYYY/MM/DD)']).dt.year

        def classify_decade(year):
            if 2031 <= year <= 2050:
                return '2040s'
            elif 2061 <= year <= 2080:
                return '2070s'
            else:
                return 'other'
        df['times'] = df['year'].apply(classify_decade)
        df = df[df['times'].isin(self.times)].reset_index(drop=True)

        grouped_df = df.groupby(['times', 'scenario_params'])[['Dry yield (tonne/ha)', 'Seasonal irrigation (mm)']].mean().round(2)
        return grouped_df.reset_index()

    def _analyze_data(self):
        filepaths = sorted(glob.glob(f'{self.results_dir}/[0-9]*.csv'))
        output = [] 
        for filepath in filepaths:
            df = self._load_data(filepath)
            mean_values = self._calculate_mean_values(df)
            mean_values.insert(0, 'ID', int(os.path.basename(filepath).split('.')[0]))
            mean_values.insert(1, 'gcms', self.gcm)
            mean_values.insert(2, 'ssps', self.ssp)
            output.append(mean_values)
        
        output_df = pd.concat(output, ignore_index=True)
        # convert_columns = ['Dry yield (tonne/ha)', 'Seasonal irrigation (mm)']
        # self._convert_results_to_raster(output_df, convert_columns)
        self._save_output_results(output_df)
        return output_df
    
    @staticmethod
    def _convert_results_to_raster(results, columns, output_dir):

        cotton_units = pd.read_csv(os.path.join(project_root, 'data/grid_10km/xinjiang_cotton_units.csv'))
        df = pd.merge(results, cotton_units, on='ID', how='left')
        
        for period, group in df.groupby('periods'):
            period_output_dir = os.path.join(project_root, output_dir, period)
            os.makedirs(period_output_dir, exist_ok=True)
            converter = RasterConverter(
                df=group,
                value_columns=columns,
                res=0.1,
                output_dir=period_output_dir,
                lat_column='lat',
                lon_column='lon',
                crs='EPSG:4326'
            )
            converter.csv_to_tif()



class PotentialReturn(ResultsAnalysis):
    def __init__(self, result_dir: Tuple[str, str], output_dir: str):
        """
        initialize PotentialReturn class.

        :param result_dir: directory path tuple containing baseline (baseline) and deficit (deficit) results
        :param output_dir: result output directory
        """
        super().__init__(result_dir, output_dir)
        self.cotton_price = 7.20  # yuan / kg
        self.irrigation_price = 0.407  # yuan / m³

        # read data
        self.baseline_results = self._load_data(result_dir[0])
        self.deficit_results = self._load_data(result_dir[1])
        
        # merge data
        self.merged_df = self._merge_data()
        self._calculate_changes()

    def _merge_data(self) -> pd.DataFrame:
        """
            merge baseline and deficit result datasets.
        
        :return: merged DataFrame
        """
        merged_df = pd.merge(
            self.baseline_results, 
            self.deficit_results[['ID', 'Dry yield (tonne/ha)', 'Seasonal irrigation (mm)']], 
            on='ID', 
            how='left', 
            suffixes=('_baseline', '_deficit'))
        merged_df = merged_df[merged_df['scenario_params'] == '90-7-0']
        return merged_df

    @staticmethod
    def _calculate_irrigation_change(df: pd.DataFrame) -> pd.DataFrame:
        """calculate irrigation change (mm)"""
        df['Seasonal irrigation_diff'] = df['Seasonal irrigation (mm)_deficit'] - df['Seasonal irrigation (mm)_baseline'] 
        return df

    @staticmethod
    def _calculate_yield_change(df: pd.DataFrame) -> pd.DataFrame:
        """calculate yield change (tonne/ha)"""
        df['Dry yield_diff'] =  df['Dry yield (tonne/ha)_deficit'] - df['Dry yield (tonne/ha)_baseline']
        return df

    @staticmethod
    def _calculate_iwp_change(df: pd.DataFrame) -> pd.DataFrame:
        """calculate irrigation water productivity (IWP) change"""
        iwpb = (df['Dry yield (tonne/ha)_baseline'] * 1000) / (df['Seasonal irrigation (mm)_baseline'] * 10) # kg/m3
        iwpd = (df['Dry yield (tonne/ha)_deficit'] * 1000) / (df['Seasonal irrigation (mm)_deficit'] * 10)  # kg/m3
        df['iwp_diff'] = iwpd - iwpb
        return df

    def _calculate_potential_profit(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        calculate potential profit change:
        - income from yield change
        - cost change from water saving

        :param df: DataFrame to calculate
        :return: calculated DataFrame
        """
        yield_change = df['Dry yield_diff'] * 1000 # 1t = 1000 kg
        irrigation_change = df['Seasonal irrigation_diff'] * 10   # m3/ha , in the case of 1 ha
        
        df['potential_profit'] = (yield_change * self.cotton_price) - (irrigation_change * self.irrigation_price)
        return df
    
    def _calculate_changes(self) -> None:
        """
        calculate all changes, apply calculation logic after grouping by 'ID' and 'scenario_params'.
        """
        def process_group(group: pd.DataFrame) -> pd.DataFrame:
            group = self._calculate_irrigation_change(group)
            group = self._calculate_yield_change(group)
            group = self._calculate_iwp_change(group)
            group = self._calculate_potential_profit(group)
            return group
        
        self.merged_df = self.merged_df.groupby(['ID', 'scenario_params'], group_keys=False).apply(process_group).reset_index(drop=True)

    def _save_output_results(self):
       output_df = self.merged_df[['ID', 'scenario_params', 'Seasonal irrigation_diff', 'Dry yield_diff', 'iwp_diff', 'potential_profit']]
       output_df.to_csv(os.path.join(self.output_dir, 'potential_return_results.csv'))
       convert_columns = ['Seasonal irrigation_diff', 'Dry yield_diff', 'iwp_diff', 'potential_profit']
       ResultsAnalysis._convert_results_to_raster(output_df, convert_columns,self.output_dir)

def baseline():
    result_dir = os.path.join(project_root,'results/simulation10km/baseline')
    output_dir = os.path.join(project_root,'results/analysis10km/history/baseline')
    os.makedirs(output_dir, exist_ok=True)
    analysis = ResultsAnalysis(result_dir, output_dir)
    analysis._analyze_data()

def deficit():
    result_dir = os.path.join(project_root,'results/simulation10km/deficit')
    output_dir = os.path.join(project_root,'results/analysis10km/history/deficit')
    os.makedirs(output_dir, exist_ok=True)
    analysis = ResultsAnalysis(result_dir, output_dir)
    analysis._analyze_data()

def expert():
    result_dir = os.path.join(project_root,'results/simulation10km/expert')
    output_dir = os.path.join(project_root,'results/analysis10km/history/expert')
    os.makedirs(output_dir, exist_ok=True)
    analysis = ResultsAnalysis(result_dir, output_dir)
    analysis._analyze_data()

def optimize_return_9000_9070():
    baseline_result = os.path.join(project_root, 'results/analysis10km/history/baseline/statistic_results.csv')
    deficit_result = os.path.join(project_root, 'results/analysis10km/history/deficit/statistic_results.csv')
    result_dirs = [baseline_result, deficit_result]
    output_dir = os.path.join(project_root,'results/analysis10km/history/optimize_return9000-9070')
    os.makedirs(output_dir, exist_ok=True)
    optimize_return = PotentialReturn(result_dirs, output_dir)
    optimize_return._save_output_results()

def optimize_return_9071_9070():
    baseline_result = os.path.join(project_root, 'results/analysis10km/history/baseline/statistic_results.csv')
    deficit_result = os.path.join(project_root, 'results/analysis10km/history/expert/statistic_results.csv')
    result_dirs = [baseline_result, deficit_result]
    output_dir = os.path.join(project_root,'results/analysis10km/history/optimize_return9071-9070')
    os.makedirs(output_dir, exist_ok=True)
    optimize_return = PotentialReturn(result_dirs, output_dir)
    optimize_return._save_output_results()

def history():
    baseline()
    deficit()
    # expert()
    optimize_return_9000_9070()
    # optimize_return_9071_9070()

def future():
    def load_and_process_csv(file_path, scenario_filter, periods, gcms):
        df = pd.read_csv(file_path)
        # df = df.drop('iwp(kg/m3)', axis=1)
        df = df[df['scenario_params'] == scenario_filter]
        df['periods'] = periods
        df['gcms'] = gcms
        # grouped = df.groupby(['periods', 'gcms', 'scenario_params'], as_index=False)[
        #     ['Dry yield (tonne/ha)', 'Seasonal irrigation (mm)']
        # ].mean()
        # grouped['iwp'] = (grouped['Dry yield (tonne/ha)'] * 1000) / (grouped['Seasonal irrigation (mm)'] * 10)
        return df
    
    def run_future_simulation_analysis(project_root):
        base_dir = os.path.join(project_root, 'results/simulation10km/future')
        all_results = []
        for fd_name in os.listdir(base_dir):
            input_path = os.path.join(base_dir, fd_name)
            output_path = os.path.join(project_root, 'results/analysis10km/future', fd_name)
            os.makedirs(output_path, exist_ok=True)

            analysis = FutureResultsAnalysis(input_path, output_path, fd_name)
            output_df = analysis._analyze_data()
            all_results.append(output_df)

        all_results_df = pd.concat(all_results, ignore_index=True)
        all_results_df['periods'] = all_results_df['times'] + '_' + all_results_df['ssps']
        all_results_df = all_results_df.drop(['times','ssps'], axis=1)
        output_file = os.path.join(project_root, 'results/analysis10km/future/summary_future_simulation_results.csv')
        all_results_df.to_csv(output_file, index=False)
        return all_results_df  
    
    def merged_history_periods(grouped_future):
        # 处理 baseline 和 deficit 情况
        grouped_baseline = load_and_process_csv(
            os.path.join(project_root, 'results/analysis10km/history/baseline/statistic_results.csv'),
            scenario_filter='90-7-0',
            periods='history',
            gcms='historical'
        )
        grouped_deficit = load_and_process_csv(
            os.path.join(project_root, 'results/analysis10km/history/deficit/statistic_results.csv'),
            scenario_filter='90-0-0',
            periods='history',
            gcms='historical'
        )
        # grouped_expert = load_and_process_csv(
        #     os.path.join(project_root, 'results/analysis10km/history/expert/statistic_results.csv'),
        #     scenario_filter='90-7-1',
        #     periods='history',
        #     gcms='historical'
        # )

        # 合并结果并保存
        grouped_all = pd.concat([grouped_future, grouped_baseline, grouped_deficit])
        # grouped_all.to_csv(
        #     os.path.join(project_root, 'results/analysis10km/future/different_periods_results.csv'),
        #     index=False
        # )
        return grouped_all
   
    def static_future_return(df, output_dir):
        df = df.copy()
        df['iwp(kg/m3)'] = (df['Dry yield (tonne/ha)'] * 1000) / (df['Seasonal irrigation (mm)'] * 10)
        df_all = merged_history_periods(df)
        df_all = df_all.groupby(['periods', 'gcms', 'scenario_params'], as_index=False)[
        ['Dry yield (tonne/ha)', 'Seasonal irrigation (mm)', 'iwp(kg/m3)']].mean()
        # save future simulate results
        df_all.to_csv(os.path.join(output_dir, 'different_periods_results.csv'), index=False)

        # 将数据 pivot 成以 scenario_params 为列的格式，便于计算差值
        pivot_df = df_all.pivot(index=["periods", "gcms"], columns="scenario_params")
        # 计算差值：90-0-0 - 90-7-0，和 90-7-1 - 90-7-0
        diff_9000 = pivot_df["Dry yield (tonne/ha)"]["90-0-0"] - pivot_df["Dry yield (tonne/ha)"]["90-7-0"]
        # diff_9071 = pivot_df["Dry yield (tonne/ha)"]["90-7-1"] - pivot_df["Dry yield (tonne/ha)"]["90-7-0"]

        irr_9000 = pivot_df["Seasonal irrigation (mm)"]["90-0-0"] - pivot_df["Seasonal irrigation (mm)"]["90-7-0"]
        # irr_9071 = pivot_df["Seasonal irrigation (mm)"]["90-7-1"] - pivot_df["Seasonal irrigation (mm)"]["90-7-0"]

        iwp_9000 = pivot_df["iwp(kg/m3)"]["90-0-0"] - pivot_df["iwp(kg/m3)"]["90-7-0"]
        # iwp_9071 = pivot_df["iwp(kg/m3)"]["90-7-1"] - pivot_df["iwp(kg/m3)"]["90-7-0"]

        results = pd.DataFrame({
            "Dry_yield(9000-9070)": diff_9000 * 1000,  # 转为 kg/ha
            # "Dry_yield(9071-9070)": diff_9071 * 1000,
            "Seasonal_irrigation(9000-9070)": irr_9000 * 10,  # mm -> m³/ha
            # "Seasonal_irrigation(9071-9070)": irr_9071 * 10,
            "iwp(9000-9070)": iwp_9000,  # kg/m³
            # "iwp(9071-9070)": iwp_9071,
        })
        results = results.reset_index()  
        results["profit(9000-9070)"] = results["Dry_yield(9000-9070)"] * 7.20 - results["Seasonal_irrigation(9000-9070)"] * 0.407
        # results["profit(9071-9070)"] = results["Dry_yield(9071-9070)"] * 7.20 - results["Seasonal_irrigation(9071-9070)"] * 0.407

        value_vars = [col for col in results.columns if "(" in col]
        df_melted = pd.melt(
            results,
            id_vars=["periods", "gcms"],
            value_vars=value_vars,
            var_name="variable",
            value_name="value"
        )
        df_melted["metric"] = df_melted["variable"].str.extract(r"^(.*?)\(")
        df_melted["return_label"] = df_melted["variable"].str.extract(r"\((.*?)\)")
        df_melted = df_melted.drop(columns=["variable"])
        df_long = df_melted.pivot_table(
            index=["periods", "gcms", "return_label"],
            columns="metric",
            values="value"
        ).reset_index()
        output_file = os.path.join(output_dir, 'static_future_return.csv')
        df_long.to_csv(output_file, index=False)

    def grids_future_results(df):
        df_raster = df.copy()
        df_raster = merged_history_periods(df_raster)
        df_raster = df_raster.groupby(['ID', 'periods', 'scenario_params'], as_index=False)[
            ['Dry yield (tonne/ha)', 'Seasonal irrigation (mm)']].mean()
        df_raster['iwp(kg/m3)'] = (df_raster['Dry yield (tonne/ha)'] * 1000) / (df_raster['Seasonal irrigation (mm)'] * 10)
        for scenaio, group in df_raster.groupby('scenario_params'):
            convert_columns = ['Dry yield (tonne/ha)', 'Seasonal irrigation (mm)', 'iwp(kg/m3)']
            output_dir = os.path.join(project_root, 'results/analysis10km/future/', scenaio)
            FutureResultsAnalysis._convert_results_to_raster(group, convert_columns, output_dir)

        # save results
        dfd = df.copy()
        dfd = merged_history_periods(dfd)
        dfd = dfd.groupby(['ID', 'periods','gcms', 'scenario_params'], as_index=False)[
            ['Dry yield (tonne/ha)', 'Seasonal irrigation (mm)']].mean()
        dfd['iwp(kg/m3)'] = (dfd['Dry yield (tonne/ha)'] * 1000) / (dfd['Seasonal irrigation (mm)'] * 10)
        df_wide = dfd.pivot(index=['ID','periods','gcms'], 
                    columns='scenario_params', 
                    values=['Dry yield (tonne/ha)', 'Seasonal irrigation (mm)', 'iwp(kg/m3)'])
        df_wide.columns = [f"{sc}_{metric}" for metric, sc in df_wide.columns]
        df_wide = df_wide.reset_index()
        df_wide['periods'] = df_wide['periods'].replace('history', 'history_historical')
        df_wide = df_wide.groupby(['ID', 'periods','gcms']).mean(numeric_only=True).reset_index()
        df_wide.columns = [['ID','periods','gcms','Yield_SDI','Yield_CI', 'Irr_SDI','Irr_CI','IWP_SDI','IWP_CI']]
        df_wide.to_csv(os.path.join(project_root, 'results/analysis10km/future/grids_future_results.csv'), index=False)
    
    def grids_future_return(df, output_dir):    
        df = df.copy()
        # df = df[df['scenario_params'] != '90-7-1']
        df = df.groupby(['ID', 'periods', 'gcms','scenario_params'], as_index=False)[
            ['Dry yield (tonne/ha)', 'Seasonal irrigation (mm)']].mean()
        df['iwp(kg/m3)'] = (df['Dry yield (tonne/ha)'] * 1000) / (df['Seasonal irrigation (mm)'] * 10)
        df_all = merged_history_periods(df)
        df_wide = df_all.pivot(index=['ID','periods','gcms'], 
                    columns='scenario_params', 
                    values=['Dry yield (tonne/ha)', 'Seasonal irrigation (mm)', 'iwp(kg/m3)'])
        df_wide.columns = [f"{sc}_{metric}" for metric, sc in df_wide.columns]
        # reset
        df_wide = df_wide.reset_index()
        # calculate difference columns
        df_wide['Dry yield_diff'] = df_wide['90-0-0_Dry yield (tonne/ha)'] - df_wide['90-7-0_Dry yield (tonne/ha)']
        df_wide['Seasonal irrigation_diff'] = df_wide['90-0-0_Seasonal irrigation (mm)'] - df_wide['90-7-0_Seasonal irrigation (mm)']
        df_wide['iwp_diff'] =  df_wide['90-0-0_iwp(kg/m3)'] - df_wide['90-7-0_iwp(kg/m3)']

        yield_change = df_wide['Dry yield_diff'] * 1000 # 1t = 1000 kg
        irrigation_change = df_wide['Seasonal irrigation_diff'] * 10   # m3/ha , in the case of 1 ha
        df_wide['potential_profit'] = (yield_change * 7.20) - (irrigation_change * 0.407)
        df_profit = df_wide[['ID','periods', 'gcms','Dry yield_diff','Seasonal irrigation_diff','iwp_diff','potential_profit']]
        # save results
        df_profit['periods'] = df_profit['periods'].replace('history', 'history_historical')
        df_profit.to_csv(os.path.join(output_dir, 'grids_future_return.csv'), index=False)
        # save raster
        df_raster = df_profit.groupby(['ID', 'periods'], as_index=False)[
                    ['Dry yield_diff','Seasonal irrigation_diff','iwp_diff','potential_profit']].mean()
        df_raster = df_raster[['ID','periods', 'Dry yield_diff','Seasonal irrigation_diff','iwp_diff','potential_profit']]
        convert_columns = ['Dry yield_diff','Seasonal irrigation_diff','iwp_diff','potential_profit']
        FutureResultsAnalysis._convert_results_to_raster(df_raster, convert_columns, output_dir)


    # df_future = run_future_simulation_analysis(project_root)
    df_future = pd.read_csv(os.path.join(project_root, 'results/analysis10km/future/summary_future_simulation_results.csv'))
    # grids future results
    grids_future_results(df_future)
    # grids future return
    output_dir = os.path.join(project_root, 'results/analysis10km/future/return')
    os.makedirs(output_dir, exist_ok=True)
    grids_future_return(df_future, output_dir)
    # # caculate future optimize retuen
    static_future_return(df_future, output_dir)


def calculate_vif(df, columns):
    """
    计算指定列的VIF值
    
    参数:
        df (pd.DataFrame): 数据框
        columns (list): 要计算VIF的列名列表
    
    返回:
        pd.DataFrame: 包含每列VIF值的数据框
    """
    # 选择需要计算的列
    X = df[columns].copy()
    
    # 如果有缺失值，需要先处理
    X = X.dropna()
    
    # VIF计算
    vif_data = pd.DataFrame()
    vif_data['feature'] = X.columns
    vif_data['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    
    return vif_data


def correlation_analysis_pca(df):
    # 目标列重命名
    target_rename = {
        'potential_profit': 'Gain',
        'Seasonal irrigation_diff': 'ΔIrrigation',
        'Dry yield_diff': 'ΔYield',
        'iwp_diff': 'ΔIWP'
    }
    df = df.rename(columns=target_rename)
    df['Gain'] = df['Gain'] / 1000

    # 变量分组
    temp_vars = ["mean_tmean", "mean_et", "mean_gdd"]
    rain_vars = ["mean_prcp"]
    soil_vars = ["silt", "clay", "sand", "soc", "bdod"]

    # mask 设定
    mask_data = {
        'ΔYield': {'vmin': -0.5, 'vmax': 1},
        'ΔIrrigation': {'vmin': -300, 'vmax': 0},
        'ΔIWP': {'vmin': 0.5, 'vmax': 1},
        'Gain': {'vmin': -0.5, 'vmax': 7.5}
    }

    results = []
    df["periods"] = df["periods"].astype("category")
    base_period = df["periods"].cat.categories[0]

    targets = ['Gain','ΔIrrigation','ΔYield','ΔIWP']

    for target in targets:
        # 子集，去缺失
        model_df = df[[target, "periods"] + temp_vars + rain_vars + soil_vars].dropna()

        # 掩膜处理
        if target in mask_data:
            vmin = mask_data[target]['vmin']
            vmax = mask_data[target]['vmax']
            model_df = model_df[(model_df[target] >= vmin) & (model_df[target] <= vmax)]

        # 标准化
        scaler_temp = StandardScaler()
        scaler_rain = StandardScaler()
        scaler_soil = StandardScaler()

        temp_scaled = scaler_temp.fit_transform(model_df[temp_vars])
        rain_scaled = scaler_rain.fit_transform(model_df[rain_vars])
        soil_scaled = scaler_soil.fit_transform(model_df[soil_vars])

        # PCA
        pca_temp = PCA(n_components=1)
        model_df["PC_temp"] = pca_temp.fit_transform(temp_scaled)

        pca_rain = PCA(n_components=1)
        model_df["PC_rain"] = pca_rain.fit_transform(rain_scaled)

        pca_soil = PCA(n_components=2)
        soil_pcs = pca_soil.fit_transform(soil_scaled)
        model_df["PC_soil1"] = soil_pcs[:,0]
        model_df["PC_soil2"] = soil_pcs[:,1]

        # vif_data = calculate_vif(model_df, ['PC_temp','PC_rain','PC_soil1','PC_soil2'])
        # print(vif_data)
        # 构建带 period 交互的公式
        formula = f"{target} ~ PC_temp*C(periods) + PC_rain*C(periods) + PC_soil1*C(periods) + PC_soil2*C(periods)"

        # 回归
        model = smf.ols(formula=formula, data=model_df).fit()
        coef = model.params

        # 提取 base period 的主效应
        pcs = ["PC_temp", "PC_rain", "PC_soil1", "PC_soil2"]
        for pc in pcs:
            base_coef = coef.get(pc, 0.0)
            results.append({
                "target": target,
                "period": base_period,
                "feature": pc,
                "values": base_coef
            })
            # 提取交互项
            for p in df["periods"].cat.categories[1:]:
                interaction = f"{pc}:C(periods)[T.{p}]"
                period_coef = base_coef + coef.get(interaction, 0.0)
                results.append({
                    "target": target,
                    "period": p,
                    "feature": pc,
                    "values": period_coef
                })

    # 汇总平均
    result_df = pd.DataFrame(results)
    avg_df = result_df.groupby(["target", "feature"], as_index=False)["values"].mean()
    avg_df["period"] = "all"

    all_coef_df = pd.concat([result_df, avg_df], ignore_index=True)
    return all_coef_df

def correlation_analysis(df):
    target_rename = {
        'potential_profit': 'EB',
        'Seasonal irrigation_diff': 'ΔIrrigation',
        'Dry yield_diff': 'ΔYield',
        'iwp_diff': 'ΔIWP'
    }
    df = df.rename(columns=target_rename)
    df['EB'] = df['EB'] / 1000
    all_variablities = ["mean_tmean", "mean_prcp", "mean_et", "clay", "silt", "soc","bdod"]
    vif_data = calculate_vif(df, all_variablities)
    # print(vif_data)

    targets = ['EB','ΔIrrigation','ΔYield','ΔIWP']
    features = ["mean_tmean", "mean_prcp", "mean_et", "mean_gdd","silt", "clay", "soc", "bdod"]
    mask_data = {
        'ΔYield': {'vmin': -0.5, 'vmax': 1},
        'ΔIrrigation': {'vmin': -300, 'vmax': 0},
        'ΔIWP': {'vmin': 0.5, 'vmax': 1},
        'EB': {'vmin': -0.5, 'vmax': 7.5}
    }
    results = []
    df["periods"] = df["periods"].astype("category")
    base_period = df["periods"].cat.categories[0]

    for target in targets:
        # 子集，去缺失
        model_df = df[features + [target, "periods"]].dropna()
        # 掩膜处理
        if target in mask_data:
            vmin = mask_data[target]['vmin']
            vmax = mask_data[target]['vmax']
            model_df = model_df[(model_df[target] >= vmin) & (model_df[target] <= vmax)]
        # 对 X 和 y 同时做标准化
        scaler_X = StandardScaler()
        scaler_y = StandardScaler()

        model_df_std = model_df.copy()
        model_df_std[features] = scaler_X.fit_transform(model_df[features])
        model_df_std[target] = scaler_y.fit_transform(model_df[[target]])

        # 构造带交互的公式
        formula_parts = [f"{f}*C(periods)" for f in features]
        formula = f"{target} ~ " + " + ".join(formula_parts)

        # 回归
        model = smf.ols(formula=formula, data=model_df_std).fit()
        coef = model.params

        # 提取 base period 的主效应
        for feat in features:
            base_coef = coef.get(feat, 0.0)
            results.append({
                "target": target,
                "period": base_period,
                "feature": feat,
                "values": base_coef
            })

            # 提取交互项，组合得到其余 period 的系数
            for p in df["periods"].cat.categories[1:]:
                interaction = f"{feat}:C(periods)[T.{p}]"
                period_coef = base_coef + coef.get(interaction, 0.0)
                results.append({
                    "target": target,
                    "period": p,
                    "feature": feat,
                    "values": period_coef
                })

    result_df = pd.DataFrame(results)
    avg_df = result_df.groupby(["target", "feature"], as_index=False)["values"].mean()
    avg_df["period"] = "all"
    # merged 
    all_coef_df = pd.concat([result_df, avg_df], ignore_index=True)
    return all_coef_df

def correlation_analysis_pca_with_loadings(df):
    # 目标列重命名
    target_rename = {
        'potential_profit': 'Gain',
        'Seasonal irrigation_diff': 'ΔIrrigation',
        'Dry yield_diff': 'ΔYield',
        'iwp_diff': 'ΔIWP'
    }
    df = df.rename(columns=target_rename)
    df['Gain'] = df['Gain'] / 1000

    # 变量分组
    temp_vars = ["mean_tmean", "mean_et", "mean_gdd"]
    rain_vars = ["mean_prcp"]
    soil_vars = ["silt", "clay", "sand", "soc", "bdod"]

    mask_data = {
        'ΔYield': {'vmin': -0.5, 'vmax': 1},
        'ΔIrrigation': {'vmin': -300, 'vmax': 0},
        'ΔIWP': {'vmin': 0.5, 'vmax': 1},
        'Gain': {'vmin': -0.5, 'vmax': 7.5}
    }

    results = []
    explained_contributions = []
    df["periods"] = df["periods"].astype("category")
    base_period = df["periods"].cat.categories[0]

    targets = ['Gain','ΔIrrigation','ΔYield','ΔIWP']

    for target in targets:
        # 子集，去缺失
        model_df = df[[target, "periods"] + temp_vars + rain_vars + soil_vars].dropna()

        # 掩膜处理
        if target in mask_data:
            vmin = mask_data[target]['vmin']
            vmax = mask_data[target]['vmax']
            model_df = model_df[(model_df[target] >= vmin) & (model_df[target] <= vmax)]

        # 标准化
        scaler_temp = StandardScaler()
        scaler_rain = StandardScaler()
        scaler_soil = StandardScaler()

        temp_scaled = scaler_temp.fit_transform(model_df[temp_vars])
        rain_scaled = scaler_rain.fit_transform(model_df[rain_vars])
        soil_scaled = scaler_soil.fit_transform(model_df[soil_vars])

        # PCA
        pca_temp = PCA(n_components=1)
        model_df["PC_temp"] = pca_temp.fit_transform(temp_scaled)

        pca_rain = PCA(n_components=1)
        model_df["PC_rain"] = pca_rain.fit_transform(rain_scaled)

        pca_soil = PCA(n_components=2)
        soil_pcs = pca_soil.fit_transform(soil_scaled)
        model_df["PC_soil1"] = soil_pcs[:,0]
        model_df["PC_soil2"] = soil_pcs[:,1]

        # 构建带 period 交互的公式
        formula = f"{target} ~ PC_temp*C(periods) + PC_rain*C(periods) + PC_soil1*C(periods) + PC_soil2*C(periods)"
        model = smf.ols(formula=formula, data=model_df).fit()
        coef = model.params

        pcs = ["PC_temp", "PC_rain", "PC_soil1", "PC_soil2"]

        # 回归系数映射回原始变量
        pca_loadings = {
            "PC_temp": pd.Series(pca_temp.components_[0], index=temp_vars),
            "PC_rain": pd.Series(pca_rain.components_[0], index=rain_vars),
            "PC_soil1": pd.Series(pca_soil.components_[0], index=soil_vars),
            "PC_soil2": pd.Series(pca_soil.components_[1], index=soil_vars)
        }

        # 1️⃣ 提取每个 period 的主效应
        for pc in pcs:
            base_coef = coef.get(pc, 0.0)
            results.append({
                "target": target,
                "period": base_period,
                "feature": pc,
                "values": base_coef
            })
            # 映射回原始变量
            for var, loading in pca_loadings[pc].items():
                explained_contributions.append({
                    "target": target,
                    "period": base_period,
                    "PC": pc,
                    "feature": var,
                    "contribution": base_coef * loading
                })

            # 2️⃣ 交互项系数
            for p in df["periods"].cat.categories[1:]:
                interaction = f"{pc}:C(periods)[T.{p}]"
                period_coef = base_coef + coef.get(interaction, 0.0)
                results.append({
                    "target": target,
                    "period": p,
                    "feature": pc,
                    "values": period_coef
                })
                for var, loading in pca_loadings[pc].items():
                    explained_contributions.append({
                        "target": target,
                        "period": p,
                        "PC": pc,
                        "feature": var,
                        "contribution": period_coef * loading
                    })

    # 汇总平均
    result_df = pd.DataFrame(results)
    avg_df = result_df.groupby(["target", "feature"], as_index=False)["values"].mean()
    avg_df["period"] = "all"
    all_coef_df = pd.concat([result_df, avg_df], ignore_index=True)

    contribution_df = pd.DataFrame(explained_contributions)
    avg_contribution_df = contribution_df.groupby(["target","feature"], as_index=False)["contribution"].mean()
    avg_contribution_df["period"] = "all"
    all_contribution_df = pd.concat([contribution_df, avg_contribution_df], ignore_index=True)

    return all_coef_df, all_contribution_df


import matplotlib.pyplot as plt
import seaborn as sns

def visualize_pca_regression(all_coef_df, all_contribution_df, targets=None):
    """
    可视化 PCA+回归结果
    
    参数:
        all_coef_df: DataFrame, PCA主成分回归系数
        all_contribution_df: DataFrame, 原始变量贡献
        targets: list, 需要展示的目标变量，如果为 None 则展示全部
    """
    if targets is None:
        targets = all_coef_df['target'].unique()
    
    sns.set(style="whitegrid", font_scale=1.1)

    for target in targets:
        # -------------------------------
        # 1️⃣ PCA 主成分回归系数柱状图
        # -------------------------------
        df_coef = all_coef_df[all_coef_df['target'] == target]
        plt.figure(figsize=(8,5))
        sns.barplot(data=df_coef, x='feature', y='values', hue='period', palette='Set2')
        plt.title(f'Regression Coefficients (PCA) for {target}')
        plt.ylabel('Standardized Coefficient')
        plt.xlabel('Principal Component')
        plt.legend(title='Period', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.show()

        # -------------------------------
        # 2️⃣ 原始变量贡献热力图
        # -------------------------------
        df_contrib = all_contribution_df[all_contribution_df['target'] == target]
        # pivot，行：原始变量，列：period
        pivot_df = df_contrib.pivot_table(index='feature', columns='period', values='contribution', fill_value=0)

        plt.figure(figsize=(8,6))
        sns.heatmap(pivot_df, annot=True, fmt=".2f", cmap='coolwarm', center=0, linewidths=.5)
        plt.title(f'Original Variable Contributions to {target}')
        plt.ylabel('Original Variable')
        plt.xlabel('Period')
        plt.tight_layout()
        plt.show()

def weather_soil_potential_return_correlation():
    def process_grid_df(df):
        # df = df[df['return_label'] == '9000-9070'] # noly snalysis deficit irrigation return
        # df = df.drop('return_label')
        # df['periods'] = df['periods'].replace('history', 'history_historical')
        df[['periods', 'ssps']] = df['periods'].str.extract(r'([a-zA-Z0-9]+)_([a-z0-9]+)')
        # df = df.groupby(['ID', 'periods']).mean(numeric_only=True).reset_index()
        return df
    # read data
    weather_soil_df = pd.read_csv(os.path.join(project_root, 'results/xinjiang_weather10km/grid_weather_soil_2000-2081_correlation.csv'))
    grid_return_df = pd.read_csv(os.path.join(project_root, 'results/analysis10km/future/return/grids_future_return.csv'))
    grid_future_results_df = pd.read_csv(os.path.join(project_root, 'results/analysis10km/future/grids_future_results.csv'))
    grid_return_df = process_grid_df(grid_return_df)
    grid_future_results_df = process_grid_df(grid_future_results_df)
    # merge two df
    merged_df = pd.merge(weather_soil_df, grid_return_df, on=['ID', 'ssps', 'gcms', 'periods'], how='left')
    merged_df = pd.merge(merged_df, grid_future_results_df, on=['ID', 'ssps', 'gcms', 'periods'], how='left')
    output_file = os.path.join(project_root, 'results/analysis10km/correlation/weather_soil_potential_return_correlation.csv')
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    merged_df.to_csv(output_file, index=False)

    merged_df = merged_df.groupby(['ID', 'periods'], as_index=False).mean(numeric_only=True)
    # print(merged_df)
    # all_coef_df, all_contribution_df = correlation_analysis_pca_with_loadings(merged_df)
    all_coef_df = correlation_analysis(merged_df)
    # print(all_contribution_df)
    output_file = os.path.join(project_root, 'results/analysis10km/correlation/standardized_coefficients_by_period.csv')
    # all_contribution_df = all_contribution_df.rename(columns={'contribution': 'values'})
    # all_contribution_df = all_contribution_df.drop(columns=['PC'])
    all_coef_df.to_csv(output_file, index=False)
    # print(all_coef_df)
    # # 可视化所有目标变量
    # visualize_pca_regression(all_coef_df, all_contribution_df)


def main():
    # history()
    # future()
    weather_soil_potential_return_correlation()
if __name__ == "__main__":
    main()

    

