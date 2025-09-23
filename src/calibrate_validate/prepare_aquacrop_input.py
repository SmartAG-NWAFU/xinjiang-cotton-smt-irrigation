import os
os.environ['DEVELOPMENT'] = 'True'

import pandas as pd
import glob
from aquacrop import AquaCropModel, Soil, Crop, InitialWaterContent, FieldMngt, GroundWater, IrrigationManagement
from aquacrop.utils import prepare_weather
from calibrate_validate.init_varieties_parameters import InitVarietyParameters


def pre_weather(id, weather_file_paths):
    weather_file_path = next((path for path in weather_file_paths if id.upper() in path.upper()), None)

    if weather_file_path:
        weather_data = prepare_weather(weather_file_path)
        return weather_data
    else:
        raise FileNotFoundError(f"No weather file found for id: {id}")

def pre_soil(id, soil_d):
    # soil prepare
    tempt_soil = soil_d[soil_d.ID == id]
    # Sort soil data by depth
    depth_order = ['0-5cm', '5-15cm', '15-30cm', '30-60cm', '60-100cm', '100-200cm']
    # Ensure you are modifying the DataFrame in place using .loc
    tempt_soil.loc[:, 'depth'] = pd.Categorical(tempt_soil['depth'], categories=depth_order, ordered=True)
    tempt_soil = tempt_soil.sort_values('depth').reset_index(drop=True)
    soil_layer = [0.05, 0.1, 0.15, 0.3, 0.4, 1]   # the thickness for every layer of soil
    soil = Soil('custom', dz=[0.1] * 10)
    # add soil layers
    for i in range(len(soil_layer)):
        soil.add_layer_from_texture(thickness=soil_layer[i],
                                    Sand=tempt_soil["sand"].iloc[i], Clay=tempt_soil["clay"].iloc[i],
                                    OrgMat=tempt_soil["soc"].iloc[i],
                                    penetrability=100)
    return soil

def pre_irrigation_schedule(id, sample, irrigation_data):
    maxirrigation = {'FKD':85, 'CLD':500, 'AKA':500, 'Weili':500
                    , 'Shihezi':500, 'Tumushuke':500, 'Korla':500
                    , 'Huyanghe':500, 'Urumqi':500, 'Shawan':500,'Shihezi146':500}
    tempt = irrigation_data[(irrigation_data.ID == id) & (irrigation_data.samples == sample)].iloc[:,-3:-1]
    tempt = tempt.drop_duplicates()  # if duplicates,drop it
    tempt.columns = ['Date', 'Depth']  # name columns
    irr_mngt = IrrigationManagement(irrigation_method=3, Schedule=tempt, AppEff=80,
                                    MaxIrr=maxirrigation[id])  # specify irrigation management. for KFD, it is 85, for CLD, it is 85
    return irr_mngt

def pre_phenology(id, year, phenology_data):
    pheno = phenology_data[(phenology_data.ID == id) & (phenology_data.years == year)]
    return pheno

def create_variety_parameters(init_var_paramters):
    return InitVarietyParameters(
        id=init_var_paramters['ID'].iloc[0],
        variety=init_var_paramters['varieties'].iloc[0],
        EmergenceCD=0,
        YldFormCD=0,
        SenescenceCD=0,
        MaturityCD=0,
        PlantPop=init_var_paramters['PlantPop'].iloc[0],
        CGC=init_var_paramters['CGC'].iloc[0],
        CDC=init_var_paramters['CDC'].iloc[0],
        CCx=init_var_paramters['CCx'].iloc[0],
        Zmin=init_var_paramters['Zmin'].iloc[0],
        Zmax=init_var_paramters['Zmax'].iloc[0],
        Kcb=init_var_paramters['Kcb'].iloc[0],
        WP=init_var_paramters['WP'].iloc[0],
        WPy=init_var_paramters['WPy'].iloc[0],
        HI0=init_var_paramters['HI0'].iloc[0],
        PUP1=init_var_paramters['PUP1'].iloc[0],
        PLOW1=init_var_paramters['PLOW1'].iloc[0],
        PUP2=init_var_paramters['PUP2'].iloc[0],
        PUP3=init_var_paramters['PUP3'].iloc[0]
    )

def create_crop(var_param, pheno):
    return Crop('Cotton', CalendarType=1, SwitchGDD=1,
                ## phenology parameters
                planting_date=pheno.planting.iloc[0],
                harvest_date=pheno.harvesting.iloc[0],
                ## field mangement paramters
                EmergenceCD=var_param.EmergenceCD,
                SenescenceCD=var_param.SenescenceCD,
                MaturityCD=var_param.MaturityCD,
                YldFormCD=var_param.YldFormCD,
                ## crop parameters
                CGC_CD=var_param.CGC, 
                CDC_CD=var_param.CDC,
                CCx=var_param.CCx,
                WP=var_param.WP,
                WPy=var_param.WPy,
                Zmax=var_param.Zmax,
                Zmin=var_param.Zmin,
                Kcb=var_param.Kcb,
                HI0=var_param.HI0,
                p_up1=var_param.PUP1,
                p_lo1=var_param.PLOW1,
                p_up2=var_param.PUP2,
                p_up3=var_param.PUP3,
                PlantPop=var_param.PlantPop
                )

def get_init_variety_parameters(id, variety, init_crop_paramters):
    init_var_paramters = init_crop_paramters[(init_crop_paramters.ID == id) & (init_crop_paramters.varieties == variety)]
    return init_var_paramters

def pre_crop_parameters(id, variety, init_crop_paramters):
    '''output variety parameters calss for each variety

    Args:
        id (str): id
        variety (str): variety
        init_crop_paramters (DataFrame): crop parameters

    Returns:
        var_param (InitVarietyParameters): _description_
    '''

    init_var_paramters = get_init_variety_parameters(id, variety, init_crop_paramters)
    var_param = create_variety_parameters(init_var_paramters)
    return var_param

def update_phenology_and_create_crop_parameters(vartye_crop_parametes, phenology):
    '''update crop parameters througth every years'phenology and create crop parameters

    Args:
        vartye_crop_parametes (InitVarietyParameters): crop parameters
        phenology (DataFrame): _description_

    Returns:
        cotton (Crop): crop class

    '''

    vartye_crop_parametes.EmergenceCD = phenology.EmergenceCD.iloc[0]
    vartye_crop_parametes.SenescenceCD = phenology.SenescenceCD.iloc[0]
    vartye_crop_parametes.MaturityCD = phenology.MaturityCD.iloc[0]
    vartye_crop_parametes.YldFormCD = phenology.YldFormCD.iloc[0]
    cotton = create_crop(vartye_crop_parametes, phenology)
    return cotton


def prepare_observed_cc_biomass(obs_cc_biomass_path):
    obs_cc_biomass = pd.read_csv(obs_cc_biomass_path)
    obs_cc_biomass = obs_cc_biomass.groupby(['ID', 'years','samples','dap']).mean().reset_index()
    obs_cc = obs_cc_biomass[['ID', 'years','samples','dap','cc']]
    obs_biomass = obs_cc_biomass[['ID', 'years','samples','dap','biomass_kg_ha']]
    return obs_cc, obs_biomass

def prepare_observed_yield(obs_yield_path):
    obs_yield = pd.read_csv(obs_yield_path)
    return obs_yield    
