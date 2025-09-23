import os
os.environ['DEVELOPMENT'] = 'True'

import pandas as pd
from aquacrop import Crop
import numpy as np
from planting_date import PlantingDate


class ZoneCropVarieties():
    def __init__(self, id):
        self.ID = id
        self.units_table = pd.read_csv('../data/grid_10km/aquacrop_inputdata/units_labels.csv')
        self.experimental_sites_varities = pd.read_csv('../data/xinjiang_zones/experimental_sites.csv')
        self.sites_varities_parameters = pd.read_csv('../data/sites/aquacrop_inputdata/crop/sites_crop_parameters.csv')
        self.sites_varities_phenology = pd.read_csv('../data/sites/aquacrop_inputdata/crop/sites_phenology.csv')
        self.varieties_parameters_table = self._pre_varieties_parameters()
        self.varieties_parameters_dic = None
        # First calculate the parameters dictionary
        # self.varieties_parameters_dic = self.calculate_variety_parameters()

    def _lookup_id_varieties(self):
        zone_label = self.units_table[self.units_table['ID'] == self.ID]['labels'].values[0]
        # Get the variety for the given zone
        # variety_series = self.experimental_sites_varities[self.experimental_sites_varities['labels'] == zone_label]['varieties']
        variety_series = self.experimental_sites_varities['varieties']
        variety_list = [item for sublist in variety_series.str.split(',') for item in sublist]
        variety_list = list(set(variety_list))
        return variety_list
    
    def _lookup_site_varieties(self):
        variety_series = self.experimental_sites_varities[self.experimental_sites_varities['ID'] == self.ID]['varieties']
        # variety_series = self.experimental_sites_varities['varieties']
        variety_list = [item for sublist in variety_series.str.split(',') for item in sublist]
        variety_list = list(set(variety_list))
        return variety_list


    def _phenology_paramters(self):
        # Calculate the mean of numeric phenology parameters by 'varieties'
        numeric_columns = ['EmergenceCD', 'YldFormCD', 'SenescenceCD', 'MaturityCD']
        means_phenology = self.sites_varities_phenology.groupby('varieties')[numeric_columns].mean().astype(int).reset_index()
        return means_phenology
    
    def _pre_crop_planting_date(self):
        self.sites_varities_phenology['planting'] = pd.to_datetime(self.sites_varities_phenology['planting'], format='%m/%d')
        # Calculate the average planting date for each variety
        planting_date = (self.sites_varities_phenology.groupby('varieties')['planting']
                            .mean()
                            .dt.strftime('%m/%d')
                            .reset_index())
        # planting_date = PlantingDate(self.ID)
        return planting_date

    def _pre_varieties_parameters(self):
        # Merge initial parameters with averaged phenology parameters and planting dates
        merged_data = pd.merge(self.sites_varities_parameters, self._phenology_paramters(), on='varieties', how='left')
        merged_data = pd.merge(merged_data, self._pre_crop_planting_date(), on='varieties', how='left')
        return merged_data

    def create_aquacrop_parameters(self, variety):
        variety_parameters_row = self.varieties_parameters_table[self.varieties_parameters_table['varieties']==variety].iloc[0]
        variety_parameters = Crop(
            'Cotton', 
            CalendarType=1, 
            SwitchGDD=0,
            # Phenology parameters
            planting_date=variety_parameters_row.planting,
            # planting_date='04/20',
            # harvest_date = '12/31',
            # Field management parameters
            EmergenceCD=variety_parameters_row.EmergenceCD,
            SenescenceCD=variety_parameters_row.SenescenceCD,
            MaturityCD=variety_parameters_row.MaturityCD,
            # MaturityCD=250,
            YldFormCD=variety_parameters_row.YldFormCD,
            # Crop parameters
            CGC_CD=variety_parameters_row.CGC, 
            CDC_CD=variety_parameters_row.CDC,
            CCx=variety_parameters_row.CCx,
            WP=variety_parameters_row.WP,
            WPy=variety_parameters_row.WPy,
            Zmax=variety_parameters_row.Zmax,
            Zmin=variety_parameters_row.Zmin,
            Kcb=variety_parameters_row.Kcb,
            HI0=variety_parameters_row.HI0,
            p_up1=variety_parameters_row.PUP1,
            p_lo1=variety_parameters_row.PLOW1,
            p_up2=variety_parameters_row.PUP2,
            p_up3=variety_parameters_row.PUP3,
            PlantPop=variety_parameters_row.PlantPop
        )
        return variety_parameters  # Return the dictionary of variety parameters

    def create_varieties_parameters(self):
        # Create a dictionary of variety parameters
        variety_parameters_dic = {}
        for variety in self._lookup_id_varieties():
            variety_parameters_dic[variety] = self.create_aquacrop_parameters(variety)
        return variety_parameters_dic

if __name__ == '__main__':
    crop_varieties = ZoneCropVarieties(id='AKA')
    # print(crop_varieties.varieties_parameters_table)
    # print(crop_varieties._lookup_id_varieties())
    print(crop_varieties._lookup_site_varieties())
    # print(crop_varieties.create_varieties_parameters())
