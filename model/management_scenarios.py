import os
os.environ['DEVELOPMENT'] = 'True'
from aquacrop import FieldMngt, IrrigationManagement
import random
import numpy as np
import pandas as pd


class ManagementScenarios():
    def __init__(self, id):
        self.ID = id
        self.id_label = self._get_id_label()
        self.optimized_deficit_irrigation = self._get_zone_deficit_irrigation()
        self.baseline_scenarios = self._basline_irrigate()
        self.difict_irrigation = self._get_dificit_threshold()
        self.exp_irr_interval = None
        self.exp_irr_amount_day = None
        self.exp_irr_amount = None
        self.exp_irr_schedule = self._get_expert_recommendation()


    def _get_id_label(self):
        units_table = pd.read_csv('../data/grid_10km/aquacrop_inputdata/units_labels.csv')
        id_label = units_table[units_table['ID'] == self.ID]['labels'].values[0]
        return id_label
        
    def _get_zone_deficit_irrigation(self):
        # site_label = pd.read_csv('../data/xinjiang_zones/experimental_sites.csv')
        optimized_deficit_irrigation_df = pd.read_csv('../results/simulation10km/SMT_summary.csv')
        # optimized_deficit_irrigation_merged = pd.merge(optimized_deficit_irrigation_df, site_label[['ID','labels']], on='ID',how='left')
        optimized_deficit_irrigation_df = optimized_deficit_irrigation_df[optimized_deficit_irrigation_df['ID'] == self.ID]
        return optimized_deficit_irrigation_df

    def _get_dificit_threshold(self):
        '''创建不同的灌溉阈值
        '''
        # Get the irrigation for the given zone
        smts_df = self.optimized_deficit_irrigation[['SMT1', 'SMT2', 'SMT3', 'SMT4']]
        # smts_df = smts_df.mean(axis=0).astype(int)
        # smts = [smts_df.values.tolist()]
        smts = smts_df.values.tolist()
        return smts

    def _basline_irrigate(self):
        """
        Generate a list of management scenarios.
        Returns a list of tuples, each containing a FieldMngt and IrrigationManagement object.
        """
        mulch_pcts = [90]
        irr_intervals = [7]
        depths = [0]
        management_scenarios = []
        for mulch_pct in mulch_pcts:  # If you only want the first mulch_pct, MULCH_PCT[:1] is fine
            for irr_interval in irr_intervals:
                for depth_day in depths:
                    # Create FieldMngt and IrrigationManagement objects
                    management_scenarios.append((mulch_pct, irr_interval, depth_day))  # Append each combination here
        return management_scenarios  # Return after the loop completes
    
    def _get_expert_recommendation(self):
        if self.id_label == 1: # south xinjiang
            self.exp_irr_interval = 5
            self.exp_irr_amount_day = 45
            self.exp_irr_amount = 450
        else:
            self.exp_irr_interval = 7 # north xinjiang
            self.exp_irr_amount_day = 33
            self.exp_irr_amount = 360

        # 存放所有年份的灌溉日期
        all_dates = []
        for year in range(2000, 2081):  # 2000 到 2080
            # 每年4月1日到12月31日，按间隔生成
            yearly_dates = pd.date_range(
                start=f'{year}-04-10',
                end=f'{year}-12-31',
                freq=f'{self.exp_irr_interval}D'
            )
            all_dates.extend(yearly_dates)

        # 转换为字符串格式
        all_dates = pd.to_datetime(all_dates).strftime('%Y-%m-%d')
        depths = [self.exp_irr_amount_day] * len(all_dates)

        schedule = pd.DataFrame({'Date': all_dates, 'Depth': depths})
        return schedule
        
    def _creat_baseline_irrigate(self):

        management_scenarios_list = []
        for scenario in self.baseline_scenarios:
            mulch_pct, irr_interval, depth_day = scenario
            field_management = FieldMngt(mulches=True, mulch_pct=90, f_mulch=0.9)
            # irrigation_management = IrrigationManagement(irrigation_method=3,ss
            #   schedule=create_irrigation_calendar(depth_day, irr_interval), AppEff=80, MaxIrr=500) # define irrigation management
            irrigation_management = IrrigationManagement(irrigation_method=2, IrrInterval=irr_interval, AppEff=75, MaxIrr=50) 
            # app_eff = 55%, max_irr = 35
            # define irrigation management
            # Append the tuple (field_management, irrigation_management) to the list
            management_scenarios_list.append([mulch_pct, irr_interval, depth_day, field_management, irrigation_management])

        return management_scenarios_list

    def _creat_deficit_irrigate(self):
        deficit_irrigation_list = []
        field_management = FieldMngt(mulches=True, mulch_pct=90, f_mulch=0.9)
        for smts in self.difict_irrigation:
            irrmngt = IrrigationManagement(irrigation_method=1, SMT=smts, MaxIrr=4, AppEff=100) # define irrigation management
            deficit_irrigation_list.append([90, 0, 0, field_management, irrmngt])
        return deficit_irrigation_list
    
    def _creat_expert_irrigate(self):
        field_management = FieldMngt(mulches=True, mulch_pct=90, f_mulch=0.9)
        expert_recommendation = IrrigationManagement(irrigation_method=3, Schedule=self.exp_irr_schedule, AppEff=100,
                                                     MaxIrrSeason=self.exp_irr_amount)
        return [[90, 7, 1, field_management, expert_recommendation]]

    def _creat_future_irrigate(self):
        future_irrigation_list = []
        field_management = FieldMngt(mulches=True, mulch_pct=90, f_mulch=0.9)
        baseline_irrigation_management = IrrigationManagement(irrigation_method=2,
                                                              IrrInterval=7, AppEff=75, MaxIrr=50) # define irrigation management
        future_irrigation_list.append([90, 7, 0, field_management, baseline_irrigation_management])
        
        # expert_recommendation = IrrigationManagement(irrigation_method=3, Schedule=self.exp_irr_schedule, 
        #                                              AppEff=85, MaxIrrSeason=self.exp_irr_amount)
        # future_irrigation_list.append([90, 7, 1, field_management, expert_recommendation])

        for smts in self.difict_irrigation:
            smt_irrmngt = IrrigationManagement(irrigation_method=1, SMT=smts, MaxIrr=4, AppEff=100) # define irrigation management
            future_irrigation_list.append([90, 0, 0, field_management, smt_irrmngt])
        return future_irrigation_list
        

if __name__ == "__main__":
    scenarios = ManagementScenarios(1004)
    # print(scenarios.baseline_scenarios)
    # print(scenarios.optimized_deficit_irrigation)
    # print(scenarios.difict_irrigation)
    print(scenarios.exp_irr_schedule)
    # print(scenarios._creat_baseline_irrigate())
    # print(scenarios._creat_deficit_irrigate())
    # print(scenarios._creat_expert_irrigate())
    # print(scenarios._creat_future_irrigate())

