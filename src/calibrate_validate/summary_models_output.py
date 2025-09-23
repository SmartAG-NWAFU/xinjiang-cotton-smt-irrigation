import pandas as pd

class ModelOutputSummary:
    def __init__(self):
        self.crop_growth_list = []
        self.water_flux_list = []
        self.water_storage_list = []
        self.final_list = []   
        self.growth = pd.DataFrame()
        self.water_flux = pd.DataFrame()
        self.water_storage = pd.DataFrame()
        self.final_stats = pd.DataFrame()   

    def summary_growth_results(self, id, year, var, sample, model):
        temp = model.get_crop_growth()
        temp['ID'] = id
        temp['years'] = year
        temp['samples'] = sample
        temp['varieties'] = var
        self.crop_growth_list.append(temp)

    def summary_water_flux_results(self, id, year, var, sample,  model):
        temp = model.get_water_flux()
        temp['ID'] = id
        temp['years'] = year
        temp['samples'] = sample
        temp['varieties'] = var
        self.water_flux_list.append(temp)

    def summary_water_storage_results(self, id, year, var, sample, planting_date, model):
        temp = model.get_water_storage()
        temp['ID'] = id
        temp['years'] = year
        temp['samples'] = sample
        temp['varieties'] = var
        temp['planting_date'] = planting_date # adding planting date can help calibration of soil water  
        self.water_storage_list.append(temp)

    def summary_final_stats_results(self, id, year, var, sample, model):
        temp = model.get_simulation_results()
        temp['ID'] = id
        temp['years'] = year
        temp['samples'] = sample
        temp['varieties'] = var
        self.final_list.append(temp)

    def get_output_results(self, id, year, var, sample, planting_date, model):
        self.summary_growth_results(id, year, var, sample, model)
        self.summary_final_stats_results(id, year, var, sample, model)
        self.summary_water_flux_results(id, year, var, sample,  model)
        self.summary_water_storage_results(id, year, var, sample, planting_date, model)

    def concat_output_results(self):
        self.growth = pd.concat(self.crop_growth_list, axis=0)
        self.water_flux = pd.concat(self.water_flux_list, axis=0)
        self.water_storage = pd.concat(self.water_storage_list, axis=0)
        self.final_stats = pd.concat(self.final_list, axis=0)


    def save_output_results(self, output_path):
        self.growth.to_csv(f'{output_path}/growth.csv', index=False)
        self.water_flux.to_csv(f'{output_path}/water_flux.csv', index=False)
        self.water_storage.to_csv(f'{output_path}/water_storage.csv', index=False)
        self.final_stats.to_csv(f'{output_path}/final_stats.csv', index=False)



