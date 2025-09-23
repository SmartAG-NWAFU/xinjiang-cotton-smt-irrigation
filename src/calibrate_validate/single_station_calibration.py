import os
os.environ['DEVELOPMENT'] = 'True'

from aquacrop import AquaCropModel, Soil, Crop, InitialWaterContent, FieldMngt, GroundWater, IrrigationManagement
from summary_models_output import ModelOutputSummary
from load_data_for_calibration import CalibrationData
from prepare_aquacrop_input import *
from calibrated_plots import model_calibration_results
from init_varieties_parameters import InitVarietyParameters
# import plot_all_figures
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

script_directory = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_directory)

def run_model(year, soil, weather, irrigation, crop_paramters, planting_date):
    '''fit and run model'''

    # initial water content
    initWC = InitialWaterContent(method='Depth', depth_layer=[1], value=['FC'])

    # filed_management
    filed_management = FieldMngt(mulches=True, mulch_pct=80, f_mulch=0.9)

    model = AquaCropModel(
        sim_start_time=f'{int(year)}/{planting_date}',
        # sim_end_time=f'{int(year)}/{harvesting_date}',
        sim_end_time=f'{int(year)}/12/30',
        weather_df=weather,
        soil=soil,
        crop=crop_paramters,
        initial_water_content=initWC,
        field_management=filed_management,
        irrigation_management=irrigation
        # groundwater=GroundWater(water_table='Y', method="Constant", dates=[f'{int(year)}/4/01'], values=[1.5])
        )
    model.run_model(till_termination=True)
    return model

def process_annual_model_output(id, mos, variety, phenology_data, soil, weather, irrigation_data, vartye_crop_parametes):
    '''Iterates over years in phenology data, prepares inputs, runs model, and collects output.'''
    
    for year in phenology_data[(phenology_data.ID == id) & (phenology_data.varieties == variety)].years.unique():
        sample = phenology_data[(phenology_data.ID == id) & (phenology_data.years == year)].samples.iloc[0]
        sample = str(sample).strip()  # Convert to string and strip any whitespace
        phenology = pre_phenology(id, year, phenology_data)
        planting_date = phenology.planting.iloc[0]
        irrigation = pre_irrigation_schedule(id, sample, irrigation_data)
        crop_paramters = update_phenology_and_create_crop_parameters(vartye_crop_parametes, phenology)
        model = run_model(year, soil, weather, irrigation, crop_paramters, planting_date)
        mos.get_output_results(id, year, variety, sample, planting_date, model)
    return mos

def single_station_calibration():

    cal_data = CalibrationData(use_initial_crop_parameters=True)

    sites_id = cal_data.experiment_sites.ID.unique()
    # sites_id = ['Shihezi146']
    mos = ModelOutputSummary()
    for id in sites_id:
        weather = pre_weather(id, cal_data.weather_file_paths)
        soil = pre_soil(id, cal_data.soil_data)
        varieties = cal_data.optimized_crop_parameters[(cal_data.optimized_crop_parameters.ID == id)].varieties.unique()
        for variety in varieties:
            print(id, variety)
            variety_crop_parametes = pre_crop_parameters(id, variety, cal_data.optimized_crop_parameters)
            mos = process_annual_model_output(id, mos, variety, cal_data.phenology_data, soil, weather,
                                        cal_data.irrigation_data, variety_crop_parametes)
    mos.concat_output_results()
    mos.save_output_results('../../results/calibration/model_output/')
    model_calibration_results(mos, cal_data)
    # plot_all_figures.fig1_calibration()

if __name__ == '__main__': 
    single_station_calibration()

