import glob
import pandas as pd
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
# print(project_root)
class CalibrationData:
    def __init__(self, use_initial_crop_parameters=True):
        self.using_initial_crop_parameters = use_initial_crop_parameters
        self.experiment_sites = pd.read_csv(os.path.join(project_root, 'data/xinjiang_zones/experimental_sites.csv'))
        self.weather_file_paths = glob.glob(os.path.join(project_root, 'data/sites/aquacrop_inputdata/weather/*.txt'))
        self.soil_data = pd.read_csv(os.path.join(project_root, 'data/sites/aquacrop_inputdata/soil/soil.csv'))
        self.irrigation_data = pd.read_csv(os.path.join(project_root, 'data/sites/aquacrop_inputdata/irrigation/irrigations.csv'))
        self.phenology_data = pd.read_csv(os.path.join(project_root, 'data/sites/aquacrop_inputdata/crop/sites_phenology.csv'))
        self.init_crop_parameters = pd.read_csv(os.path.join(project_root, 'data/sites/aquacrop_inputdata/crop/sites_crop_parameters_calval.csv'))
        self.optimized_crop_parameters = self.checking_optimized_crop_parameters()
        self.obs_cc_biomass_path = os.path.join(project_root, 'data/sites/aquacrop_inputdata/observation/obs_lai_biomass_dap_results.csv')
        self.obs_soil_water_storage_path = os.path.join(project_root, 'data/calibration/soil_water.csv')
        self.obs_yield = pd.read_csv(os.path.join(project_root, 'data/sites/aquacrop_inputdata/observation/obs_finally_yield.csv'))
        self.obs_cc, self.obs_biomass = self.prepare_observed_cc_biomass(self.obs_cc_biomass_path)
        self.obs_soil_water = self.prepare_observed_soil_water(self.obs_soil_water_storage_path)

    def checking_optimized_crop_parameters(self):
        if self.using_initial_crop_parameters:
            # print('using initial crop parameters!')
            # Consider raising an exception instead of just printing
            # raise FileNotFoundError('Optimized crop parameters file not found.')
            optimized_crop_parameters = self.init_crop_parameters
        else:
            # print('using optimized crop parameters!')
            optimized_crop_parameters = pd.read_csv(os.path.join(project_root,'data/calibration/optimizated_crop_parameters.csv'))
        return optimized_crop_parameters
    
    def prepare_observed_cc_biomass(self, obs_cc_biomass_path):
        obs_cc_biomass = pd.read_csv(obs_cc_biomass_path)
        obs_cc = obs_cc_biomass[['ID', 'years','samples','dap','cc']]
        obs_cc = obs_cc.groupby(['ID', 'years', 'samples', 'dap']).mean().reset_index()
        obs_cc['cc'] = obs_cc['cc'] * 100  # convert to percentage
        obs_cc = obs_cc.dropna() # drop rows with NaN values
        obs_biomass = obs_cc_biomass[['ID', 'years','samples','dap','biomass(kg/ha)']]
        obs_biomass = obs_biomass.groupby(['ID', 'years', 'samples', 'dap']).mean().reset_index()
        obs_biomass = obs_biomass.dropna() # drop rows with NaN values
        return obs_cc, obs_biomass
    
    def prepare_observed_soil_water(self, obs_soil_water_path):
        obs_soil_water = pd.read_csv(obs_soil_water_path)
        obs_soil_water['date'] = pd.to_datetime(obs_soil_water['date'])

        if any(obs_soil_water['ID'] == 'FKD'):
            # data = obs_soil_water[obs_soil_water['date'].dt.year.isin([2009, 2010, 2011, 2012, 2013])]
            data = obs_soil_water
        else:
            data = obs_soil_water

        def add_dap_for_soil_water_observation(data):
            '''
            add dap column to data
            '''
            data['years'] = data['date'].dt.year
            df = pd.merge(data, self.phenology_data[['ID', 'years','planting']], on=['ID', 'years'], how='left')
            df['planting_date'] = pd.to_datetime(df['years'].astype(str) + '/' + df['planting'])
            df['dap'] = (df['date'] - df['planting_date']).dt.days
            return df.dropna()
        return add_dap_for_soil_water_observation(data)

if __name__ == '__main__':
    cal_data = CalibrationData(use_initial_crop_parameters=True)
    print(cal_data.soil_data)
    # print(cal_data.optimized_crop_parameters)
    # print(cal_data.obs_soil_water)
