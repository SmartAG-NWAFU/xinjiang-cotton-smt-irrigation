import pandas as pd
import os
from functools import reduce
from typing import List



class AquacropSoilPreparer:
    """Soil data processor"""
    
    SOIL_CONSTITUENTS = ['sand', 'silt', 'clay', 'soc', 'bdod']
    MERGE_COLUMNS = ['ID', 'lat', 'lon', 'depth']
    OUTPUT_ORDER = ['ID', 'depth', 'sand', 'clay', 'silt', 'soc', 'bdod']

    def __init__(self, input_dir: str):
        """
        :param input_dir: Directory path containing *_mean.csv files
        """
        self.input_dir = input_dir

    def _preprocess_columns(self, df: pd.DataFrame, var: str) -> pd.DataFrame:
        """Column name preprocessing (key fixing step)"""
        # Keep ID and coordinate columns
        base_cols = [c for c in df.columns if c in ['ID', 'lat', 'lon']]
        
        # Process feature columns: bdod_0-5cm_mean -> bdod_0-5cm
        feature_cols = [
            f"{var}_{'_'.join(col.split('_')[1:-1])}"  # Extract depth part, remove _mean
            for col in df.columns
            if col.startswith(var) and '_mean' in col
        ]
        
        # Reconstruct DataFrame
        processed = df[base_cols + [c for c in df.columns if c.startswith(var)]]
        processed.columns = base_cols + feature_cols
        return processed

    def _reshape_to_long(self, df: pd.DataFrame, var: str) -> pd.DataFrame:
        """Wide to long table conversion (fixing data loss issues)"""
        return pd.wide_to_long(
            df,
            stubnames=var,
            i=['ID', 'lat', 'lon'],
            j='depth',
            sep='_',
            suffix=r'.+'
        ).reset_index()

    def _convert_units(self, df: pd.DataFrame, var: str) -> pd.DataFrame:
        """Unit system conversion"""
        conversions = {
            'soc': 0.01 * 0.1,   # dg/kg -> % (0.01 * 0.1=0.001)
            'bdod': 0.01 * 0.1,  # cg/cm³ -> g/cm³ -> %
            'default': 0.1       # g/kg -> %
        }
        
        scale = conversions.get(var, conversions['default'])
        df[var] = df[var] * scale
        return df

    def _process_single(self, var: str) -> pd.DataFrame:
        """Process a single soil constituent"""
        # Load data
        raw_df = pd.read_csv(os.path.join(self.input_dir, f"{var}_mean.csv"))
        
        # Processing steps
        df = self._preprocess_columns(raw_df, var)
        long_df = self._reshape_to_long(df, var)
        return self._convert_units(long_df, var)

    def process(self, output_path: str) -> None:
        """Execute the complete processing workflow"""
        # Process all constituents
        processed_dfs = [self._process_single(var) for var in self.SOIL_CONSTITUENTS]
        # Merge data (validate data structure before merging)
        merged = reduce(
            lambda left, right: pd.merge(left, right, on=self.MERGE_COLUMNS, how='inner'),
            processed_dfs
        )
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        return merged[self.OUTPUT_ORDER].to_csv(output_path, index=False)


class AquacropGridsSoilPreparer(AquacropSoilPreparer):
    """Grid soil data processor"""

    def __init__(self, input_dir: str):
        """
        :param input_dir: Directory path containing *_mean.csv files
        """
        super().__init__(input_dir)
        self.units = pd.read_csv('../../data/grid_10km/xinjiang_cotton_units.csv')
    def _interpolate_missing_ids(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add missing IDs to the DataFrame"""
        df_c = df.drop(columns=['lon', 'lat']).copy()
        df_c = pd.merge(self.units[['ID','lon','lat']], df_c, on='ID', how='left')
        return df_c.fillna(method='ffill')  # 前向填充

    
    def _process_single(self, var: str) -> pd.DataFrame:
        """Process a single soil constituent"""
        # Load data
        raw_df = pd.read_csv(os.path.join(self.input_dir, f"{var}_mean.csv"))
        df = self._interpolate_missing_ids(raw_df)
        # Processing steps
        df_col = self._preprocess_columns(df, var)
        long_df = self._reshape_to_long(df_col, var)
        long_df_converted = self._convert_units(long_df, var)
        return long_df_converted


def sites_soil_preparation() -> None:
    sites = AquacropSoilPreparer(input_dir='../../data/sites/soil')
    sites.process('../../data/sites/aquacrop_inputdata/soil/soil.csv')


def grids_soil_preparation() -> None:
    grids = AquacropGridsSoilPreparer(input_dir='../../data/grid_10km/soil')
    grids.process('../../data/grid_10km/aquacrop_inputdata/soil/soil.csv')
    # grids._process_single('sand')


if __name__ == '__main__':
    # sites_soil_preparation()
    grids_soil_preparation()
