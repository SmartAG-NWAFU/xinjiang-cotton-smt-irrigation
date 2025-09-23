import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def filter_sample(df):
    filter_conditions = [
        (df['ID'] == 'AKA') & (df['samples'] == 'AKAZH01ABC_01'),
        (df['ID'] == 'CLD') & (df['samples'] == 'CLDZH01ABC_01'),
        (df['ID'] == 'FKD') & (df['samples'] == 'FKDZH01ABC_01'),
        ~df['ID'].isin(['AKA', 'CLD', 'FKD'])
    ]
    # apply fliter 
    df_filtered = df[np.logical_or.reduce(filter_conditions)]
    return df_filtered

def summary_yield(file_path):
    df = pd.read_csv(file_path)
    df_filtered = filter_sample(df)
    df_filtered = df_filtered[['ID','years','yield_sta(t/ha)']]
    return df_filtered


def summary_irrigation_depth(file_path):
    df = pd.read_csv(file_path)
    df = filter_sample(df)
    # 提取年份
    df = df.copy()
    df['years'] = pd.to_datetime(df['date']).dt.year
    # 按ID和years分组，计算depth的和
    df_grouped = df.groupby(['ID', 'years'], as_index=False)['depth'].sum()
    df_grouped = df_grouped.rename(columns={'depth': 'sum_depth(mm)'})
    return df_grouped[['ID', 'years', 'sum_depth(mm)']]

def caculated_iwp(df):
    df['iwp(kg/m3)'] = (df['yield_sta(t/ha)'] * 1000) / (df['sum_depth(mm)'] * 10)
    return df

if __name__ == "__main__":
    yield_file_path = '../../data/sites/aquacrop_inputdata/observation/obs_finally_yield.csv'
    irrigation_file_path = '../../data/sites/aquacrop_inputdata/irrigation/irrigations.csv'
    df_yield = summary_yield(yield_file_path)
    df_irr = summary_irrigation_depth(irrigation_file_path)
    df_merged = pd.merge(df_yield, df_irr, on=['ID','years'])
    # print(df_merged.ID.unique())
    df_results = caculated_iwp(df_merged)
    df_results.to_csv('../../results/sites_yield_irrigation_statisic.csv', index=False)


