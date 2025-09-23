import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter, MonthLocator
from matplotlib.font_manager import FontProperties
import seaborn as sns
import os

class ModelInputDataProcess():

    def __init__(self):
        self.development_stages_path = '../../data/cnera_data/development_stages.csv'
        self.irrigation_schedule_path = '../../data/cnera_data/irrigation_schedules.csv'
        self.soil_mechanical_data_path = '../../data/cnera_data/土壤机械组成.csv'
        self.soil_orangic_data_path = '../../data/cnera_data/土壤养分.csv'
        self.experience_parameters_file_path = '../../data/calibration/init_experiences_parameters.csv'
        self.calibration_year_label_path = '../../data/calibration/calibration_valibration_year_label.csv'
        self.soil_water_path = '../../data/cnera_data/土壤含水量.csv'


        self.pheno = pd.DataFrame()
        self.soil = pd.DataFrame()
        self.irrigation_schedule = pd.DataFrame()
        self.init_crop_parameters = pd.DataFrame()

    def add_caliration_lable(self):
        cal_label = pd.read_csv(self.calibration_year_label_path)
        self.pheno = pd.merge(self.pheno, cal_label, on=['ID', 'years'], how='left')
        return self.pheno
    
    def phenology_process(self):
        def calculate_dap_columns(develoment_stages, cal_dapcolms):
            for col in cal_dapcolms:
                new_col = col + "_dap"  
                develoment_stages[new_col] = (develoment_stages[col] - develoment_stages['播种期(月/日/年)']).dt.days
            return develoment_stages
        
        def map_chinese_to_english(crop):
            chinese_to_english_mapping = {
            '中棉49': 'Zhongmian_49',
            '拓农1号': 'Tuonong_1',
            '塔河36': 'Tahe_36',
            '塔棉2号': 'TaMian_2',
            '新陆中73号': 'XinluZhong_73',
            '新陆中66号': 'XinluZhong_66',
            '新陆早5号': 'XinluEarly_5',
            '策科1号': 'Ceko_1',
            '中棉35': 'Zhongmian_35',
            '新陆中28号': 'XinluZhong_28',
            '豫棉15': 'YuMian_15',
            '新陆中54号': 'XinluZhong_54',
            '中科棉1号': 'ZhongkeMian_1',
            'K7号': 'K7',
            '新陆21号': 'Xinlu_21',
            '新陆早10号': 'XinluEarly_10',
            '新陆早15号': 'XinluEarly_15',
            '新陆早13号': 'XinluEarly_13',
            '新石K4': 'Xinshi_K4',
            'Jan-83': 'Jan-83',
            '822': '822',
            '中30': 'Zhong_30'}
            crop['作物品种'] = crop['作物品种'].str.strip()
            crop['作物品种'] = crop['作物品种'].replace(chinese_to_english_mapping)
            return crop
        
        def filter_and_combine_crops(crop_pheno):
            fkd_crop = crop_pheno[(crop_pheno['ID'] == 'FKD') & (crop_pheno['samples'] == 'FKDZH01ABC_01') & (crop_pheno['varieties'] != '822')]
            cld_crop = crop_pheno[
                (crop_pheno['ID'] == 'CLD') & 
                (crop_pheno['samples'] == 'CLDZH01ABC_01') & 
                (crop_pheno['varieties'] != 'K7') & 
                (crop_pheno['varieties'] != 'Zhongmian_49') & 
                (crop_pheno['varieties'] != 'Zhongmian_35') &
                (crop_pheno['varieties'] != 'Xinlu_21') &
                (crop_pheno['varieties'] != 'XinluZhong_54') 
            ]
            aka_crop = crop_pheno[(crop_pheno['ID'] == 'AKA') &
                            (crop_pheno['samples'] == 'AKAZH01ABC_01') &
                            (crop_pheno['varieties'] != '822') &
                            (crop_pheno['varieties'] != 'Tahe_36') &
                            (crop_pheno['varieties'] != 'XinluZhong_73') &
                            (crop_pheno['varieties'] != 'XinluEarly_5') 
                            ]
            return pd.concat([fkd_crop, cld_crop, aka_crop])
             
        development_stages = pd.read_csv(self.development_stages_path)
        # Convert the date columns to datetime objects

        for col in development_stages.columns[-8:-1]:
            development_stages[col] = pd.to_datetime(development_stages[col])

        cal_dapcolms = ['出苗期(月/日/年)', '现蕾期(月/日/年)', '开花期(月/日/年)', '吐絮期(月/日/年)', '最终收获期(月/日/年)']
        dev_stages = calculate_dap_columns(development_stages, cal_dapcolms)
        crop_pheno = dev_stages.drop(['样地名称', '样地类别', '出苗期(月/日/年)', '现蕾期(月/日/年)', '开花期(月/日/年)', '打顶期(月/日/年)', '吐絮期(月/日/年)', '备注'], axis=1)

        crop_pheno = map_chinese_to_english(crop_pheno)
        crop_pheno['播种期(月/日/年)'] = crop_pheno['播种期(月/日/年)'].dt.strftime('%m/%d')
        crop_pheno['最终收获期(月/日/年)'] = crop_pheno['最终收获期(月/日/年)'].dt.strftime('%m/%d')
        crop_pheno.columns = ['ID', 'years', 'samples', 'varieties', 'planting', 'harvesting', 'emergence_dap', 'squaring_dap', 'flowering_dap', 'opening_dap', 'harvesting_dap']
        
        self.pheno = filter_and_combine_crops(crop_pheno)
        displace_names = {'emergence_dap':'EmergenceCD','flowering_dap':'YldFormCD','opening_dap':'SenescenceCD','harvesting_dap':'MaturityCD'}
        self.pheno.rename(columns=displace_names, inplace=True)
        self.pheno = self.add_caliration_lable()
        self.pheno.to_csv('../../data/calibration/cotton_phenology.csv', index=False)


    def irrigation_process(self):

        def process_irrigation_data(irrigation_data):
            irrigation_schedule = irrigation_data[['生态站代码','样地代码','年','灌溉时间(月/日/年)','作物生育时期','灌溉量(mm)']]
            irrigation_schedule['灌溉时间(月/日/年)'] = pd.to_datetime(irrigation_schedule['灌溉时间(月/日/年)'].str.strip(), format='%m/%d/%Y')
            irrigation_schedule.columns = ['ID', 'samples','years','date','period','depth']
            
            irrigation_schedule['period'] = irrigation_schedule['period'].replace({
                '春灌': 'spring_irrigation',
                '蕾期': 'bud_stage',
                '铃期': 'boll_stage',
                '冬灌': 'winter_irrigation',
                '开花期': 'flowering_stage',
                '现蕾期': 'squaring_stage',
                '吐絮期': 'boll_opening_stage',
                '播种前期': 'pre_sowing_stage',
                '打顶期': 'topping_stage',
                '收获期': 'harvest_stage',
                '花铃期': 'flower_boll_stage',
                '苗期': 'seedling_stage',
                '播前期': 'pre_sowing',
                '雷期': 'bud_stage',
                '结铃期': 'boll_setting_stage',
                '花期': 'flowering_stage',
                '播种前': 'pre_sowing',
                '播种后': 'post_sowing'
            })
            irrigation_schedule.samples = irrigation_schedule.samples.replace('fkdzh01', 'FKDZH01')
            return irrigation_schedule
        def calculate_accumulated_depth(irrigation_schedule):
            irrigation_schedule.sort_values(by=['ID','samples','years','date'], inplace=True)
            irrigation_schedule['accum_depth'] = irrigation_schedule.groupby(['ID','samples','years'])['depth'].cumsum()
            return irrigation_schedule
        
        def plot_annual_irrigations(id_data, id_label):
            fig, axes = plt.subplots(4, 5, figsize=(14, 10), sharex=False, sharey=True)
            
            for i, (year, group) in enumerate(id_data.groupby('years')):
                row, col = divmod(i, 5)
                ax = axes[row, col]
                
                sns.lineplot(data=group, x='date', y='accum_depth', hue='samples', drawstyle='steps-post', ax=ax)
                
                ax.set_title(f'Year {year}')
                ax.xaxis.set_major_locator(MonthLocator())
                ax.xaxis.set_major_formatter(DateFormatter('%b'))
                
                if i == 0:
                    ax.legend(title='Sample')
                else:
                    ax.get_legend().remove()
                    
            plt.tight_layout()
            
            output_directory = '../../figs/irrigation_schedule/'
            os.makedirs(output_directory, exist_ok=True)
            plt.savefig(os.path.join(output_directory, id_label + '_irrigation.png'))


        irrigation_schedule = pd.read_csv(self.irrigation_schedule_path)
        irr_schedule = process_irrigation_data(irrigation_schedule)
        self.irrigation_schedule = irr_schedule[['ID', 'samples','date','depth','period']]
        self.irrigation_schedule.to_csv('../../data/calibration/irrigations.csv', index=False)
        # calculate accumulated depth

        irr_mangement_plots = calculate_accumulated_depth(irr_schedule)
        # plot irrigations

        for name, group in irr_mangement_plots.groupby('ID'):
            plot_annual_irrigations(group, name)


    def soil_process(self):
        def process_soil_mechanical(file_path):
            soil_mechanical = pd.read_csv(file_path)
            soil_mechanical_mean = soil_mechanical.groupby(['生态站代码','样地代码','采样深度(cm)'])[['2-0.05mm砂粒百分率',
                                            '0.05-0.002mm粉粒百分率','小于0.002mm粘粒百分率']].mean().reset_index()
            soil_mechanical_mean[soil_mechanical_mean.columns[4:]] = soil_mechanical_mean[soil_mechanical_mean.columns[4:]].round(2)
            return soil_mechanical_mean

        def process_soil_organic_matter(file_path):
            soil_organic_matter = pd.read_csv(file_path)
            soil_organic_matter_mean = soil_organic_matter.groupby(['生态站代码','样地代码', '采样深度(cm)'])['土壤有机质(g/kg)'].mean().reset_index()
            soil_organic_matter_mean['SOC(%)'] = soil_organic_matter_mean['土壤有机质(g/kg)'] * 0.1 # convert unit to percentage
            del soil_organic_matter_mean['土壤有机质(g/kg)']
            return soil_organic_matter_mean

        def merge_soil_data(mechanical_data, organic_matter_data):
            soil = pd.merge(mechanical_data, organic_matter_data, on=['生态站代码','样地代码', '采样深度(cm)'], how='inner')
            soil.columns = ['ID', 'samples','depth', 'sand', 'silt', 'clay', 'soc']
            soil['depth'] = soil['depth'].str.strip().astype(str) + 'cm'
            return soil

        # process soil  data
        soil_mechanical_mean = process_soil_mechanical(self.soil_mechanical_data_path)
        soil_organic_matter_mean = process_soil_organic_matter(self.soil_orangic_data_path)
        self.soil = merge_soil_data(soil_mechanical_mean, soil_organic_matter_mean)
        self.soil.to_csv('../../data/calibration/soil.csv', index=False)

    def init_crop_parameters_process(self):
        if self.pheno.empty:
            print('Please run phenology_process first')
        else:
            self.init_crop_parameters = pd.read_csv(self.experience_parameters_file_path)
            # print(self.pheno.head())

            # pheno_mean = self.pheno.groupby(['ID','varieties'])[['emergence_dap','squaring_dap','flowering_dap',
            #                                                 'opening_dap','harvesting_dap']].mean().reset_index()
            # pheno_mean[['emergence_dap','squaring_dap',
            #             'flowering_dap','opening_dap',
            #             'harvesting_dap']] = pheno_mean[['emergence_dap', 'squaring_dap','flowering_dap',
            #                                             'opening_dap','harvesting_dap']].astype(int)
            # self.init_crop_parameters = pd.merge(pheno_mean, experience_parameters, on=['ID','varieties'], how='inner')
            # self.init_crop_parameters = pd.merge(self.pheno, experience_parameters, on=['ID','varieties'], how='inner')
            
            self.init_crop_parameters.to_csv('../../data/calibration/init_crop_parameters.csv', index=False)


    def soil_water_process(self):
        def load_soil_water_data(filepath):
            soil_water = pd.read_csv(filepath)
            soil_water['date'] = pd.to_datetime(soil_water['年'].astype(str) + '-' + soil_water['月'].astype(str) + '-' + soil_water['日'].astype(str), format='%Y-%m-%d')
            soil_water = soil_water[['台站代码','date','样地代码','土地利用类型','观测层次（cm）','体积含水量（%）']]
            soil_water.columns = ['ID','date','samples','landuse','depth(cm)','volumetric_water(%)']
            return soil_water

        def calculate_soil_water_mean(soil_water):
            soil_water_mean = soil_water.groupby(['ID','date','depth(cm)'])['volumetric_water(%)'].mean().reset_index()
            return soil_water_mean

        def save_soil_water_data(soil_water_mean, output_filepath):
            soil_water_mean.to_csv(output_filepath, index=False)

        def plot_annual_soil_water(id, data):
            data['years'] = data['date'].dt.year

            fig, axes = plt.subplots(4, 3, figsize=(16, 12), sharex=False, sharey=True)
            
            for i, (year, group) in enumerate(data.groupby('years')):
                row, col = divmod(i, 3)
                ax = axes[row, col]
                
                sns.lineplot(data=group, x='date', y='volumetric_water(%)', hue='depth(cm)', drawstyle='steps-post', ax=ax)
                
                ax.set_title(f'Year {year}')
                ax.xaxis.set_major_locator(MonthLocator())
                ax.xaxis.set_major_formatter(DateFormatter('%b'))
                
                if i == 0:
                    ax.legend(title='depth(cm)')
                else:
                    ax.get_legend().remove()
                    
            plt.tight_layout()
            outputfile = '../../figs/observation_data_test/'
            os.makedirs(outputfile, exist_ok=True)
            plt.savefig(os.path.join(outputfile, id + '_' + 'volumetric_water.png'))
            
        def plot_soil_water(data):
            # Plot observation soil water for each unique ID
            for unique_id, group in data.groupby('ID'):
                plot_annual_soil_water(unique_id, group)


        # Load soil water data
        soil_water = load_soil_water_data(self.soil_water_path)
        # Calculate mean soil water content
        soil_water_mean = calculate_soil_water_mean(soil_water)
        # soil_water_data = selecting_same_years_for_different_id(soil_water_mean)

        # Save the results
        save_soil_water_data(soil_water_mean, '../../data/calibration/soil_water.csv')
        # plot the results
        plot_soil_water(soil_water_mean)

    def run_all_process(self):
        # self.phenology_process()
        # self.irrigation_process()
        # self.soil_process()
        # self.init_crop_parameters_process()
        self.soil_water_process()



if __name__ == '__main__':
    process = ModelInputDataProcess()
    process.run_all_process()




