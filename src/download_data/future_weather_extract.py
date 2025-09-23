import os
import pandas as pd
import concurrent.futures
import ee
from gee_extractor import GeeExtractor

class Cmip6Extractor(GeeExtractor):
    def __init__(self, project, data_id, time_range, gcm_models, extract_vars, point_data_path, output_path, num_workers):
        super().__init__(project, data_id, None, point_data_path, output_path)
        self.time_range = time_range
        self.gcm_models = gcm_models
        self.extract_vars = extract_vars
        self.num_workers = num_workers
        self.scale = 27830  # CMIP6 daily data Pixel Size 27830 meters

    @staticmethod
    def get_gee_data(gee_data_id, time_range, gcm_model, extract_vars):
        """Retrieve GEE data for the defined time range."""
        gee_data = ee.ImageCollection(gee_data_id) \
            .filterDate(time_range[0], time_range[1]) \
            .filter(ee.Filter.inList('model', gcm_model)) \
            .filter(ee.Filter.inList('variable', extract_vars))
        print(gee_data.first().getInfo())

    def convert_dataframe_to_feature_collection(self, point):
        """Convert the pandas dataframe into a GEE Feature Collection."""
        features = [ee.Feature(ee.Geometry.Point([row['lon'], row['lat']]), dict(row)) for _, row in point.iterrows()]
        return ee.FeatureCollection(features)

    def extract_point_data(self, cmip6_data, ee_fc):
        """Extract data for each point location using the CMIP6 dataset."""
        def add_index(image):
            return image.set('system_index', image.get('system:index'))

        def raster_extraction(image):
            sampled_features = image.sampleRegions(collection=ee_fc, scale=self.scale)
            return sampled_features.map(lambda f: f.set('system_index', image.get('system:index')))

        results = cmip6_data.filterBounds(ee_fc).select(self.extract_vars) \
                            .map(add_index).map(raster_extraction).flatten()
        return results

    def convert_to_dataframe_and_save(self, results_list, point):
        """Convert the combined results from GEE to a pandas DataFrame."""
        data_frames = []
        sample_result = results_list[0].first().getInfo()
        column_df = list(sample_result['properties'].keys())

        for results in results_list:
            data = results.reduceColumns(ee.Reducer.toList(len(column_df)), column_df).values().get(0).getInfo()
            df = pd.DataFrame(data, columns=column_df)
            df = df.filter(self.extract_vars + ['system_index'])
            data_frames.append(df)

        combined_df = pd.concat(data_frames, ignore_index=True)
        file_name = int(point['ID'].iloc[0])
        if isinstance(file_name, (int, float)):
            file_name = int(file_name)
        output_file = f'{self.output_path}/{file_name}.csv'
        combined_df.to_csv(output_file, index=False)
        print(f"Data saved for point {file_name}.")

    def checking_point_data_weather_exists(self, point):
        """Check if the output file for a point already exists."""
        file_name = int(point['ID'].iloc[0])
        if isinstance(file_name, (int, float)):
            file_name = int(file_name)  # Convert to integer if it's a number
        output_file = f'{self.output_path}/{file_name}.csv'
        return os.path.exists(output_file)

    def one_point_extract(self, row):
        """Extract data for one point and combine results."""
        ee.Initialize()
        point = pd.DataFrame([row])
        ee_fc = self.convert_dataframe_to_feature_collection(point)

        if self.checking_point_data_weather_exists(point):
            print(f"Combined data for point {int(point['ID'].iloc[0])} already exists. Skipping.")
            return

        results_list = []
        for model_name in self.gcm_models:
            cmip6_data = ee.ImageCollection(self.gee_data_id) \
                            .filterDate(self.time_range[0], self.time_range[1]) \
                            .filter(ee.Filter.eq('model', model_name))
            results = self.extract_point_data(cmip6_data, ee_fc)
            results_list.append(results)

        self.convert_to_dataframe_and_save(results_list, point)

    def multi_point_extract(self):
        """Extract data for multiple points in parallel."""
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            executor.map(self.one_point_extract, [row for _, row in self.point_df.iterrows()])

    def run_extracting_sequence(self):
        self.read_point_data()
        for _, row in self.point_df.iterrows():
            print(row)
            self.one_point_extract(row)

    def run_extracting_sequence_parallel(self):
        self.read_point_data()
        self.multi_point_extract()

def check_output_path(output_path):
    """Check if the output path exists, and if not, create it."""
    if not os.path.exists(output_path):
        os.makedirs(output_path)
        print(f"Output path '{output_path}' created.")
    return output_path

def process_time_range(time_range, config):
    """Process data extraction for a specific time range."""
    print(f"Processing time range: {time_range}")
    output_path = check_output_path(os.path.join("../../data/grid_10km/weather/", f"{time_range[0]}_{time_range[1]}"))
    extractor = Cmip6Extractor(
        project=config["project"],
        data_id=config["data_id"],
        time_range=time_range,
        gcm_models=config["models"],
        extract_vars=config["extract_vars"],
        point_data_path=config["data_path"],
        output_path=output_path,
        num_workers=config["num_workers"]
    )
    extractor.run_extracting_sequence_parallel()
    # extractor.run_extracting_sequence()

def test():
    """Test function to run the extractor with a single time range."""
    config = {
        "project": "ee-xinjiangcotton",
        "data_id": "NASA/GDDP-CMIP6",
        "time_ranges": [("2022-01-01", "2032-12-31")],
        "extract_vars": ['pr', 'rlds', 'rsds', 'sfcWind', 'tas', 'tasmax', 'tasmin'],
        "models": ["BCC-CSM2-MR"],
        "data_path": "../../data/grid_10km/xinjiang_cotton_units.csv",
        "num_workers": 1
    }
    process_time_range(config["time_ranges"][0], config)

def main():
    config = {
        "project": "ee-xinjiangcotton",
        "data_id": "NASA/GDDP-CMIP6",
        "time_ranges": [
            ("2022-01-01", "2032-12-31"),
            ("2033-01-01", "2043-12-31"),
            ("2044-01-01", "2054-12-31"),
            ("2055-01-01", "2065-12-31"),
            ("2066-01-01", "2076-12-31"),
            ("2077-01-01", "2081-12-31")
        ],
        "extract_vars": ['pr', 'rlds', 'rsds', 'sfcWind', 'tas', 'tasmax', 'tasmin'],
        # "models": [
        #     "ACCESS-CM2", "ACCESS-ESM1-5", "BCC-CSM2-MR", "CMCC-CM2-SR5", "CMCC-ESM2",
        #     "CNRM-CM6-1", "CNRM-ESM2-1", "CanESM5", "EC-Earth3", "EC-Earth3-Veg-LR",
        #     "FGOALS-g3", "GFDL-CM4", "GFDL-ESM4", "GISS-E2-1-G", "HadGEM3-GC31-LL",
        #     "HadGEM3-GC31-MM", "INM-CM4-8", "INM-CM5-0", "IPSL-CM6A-LR", "KACE-1-0-G",
        #     "KIOST-ESM", "MIROC-ES2L", "MIROC6", "MPI-ESM1-2-HR", "MPI-ESM1-2-LR",
        #     "MRI-ESM2-0", "NESM3", "NorESM2-MM", "UKESM1-0-LL"
        # ],
        "models":["BCC-CSM2-MR", "FGOALS-g3", "GFDL-ESM4","HadGEM3-GC31-LL", "IPSL-CM6A-LR", "MIROC6"],
        "data_path": "../../data/grid_10km/xinjiang_cotton_units.csv",
        "num_workers": 10
    }

    with concurrent.futures.ThreadPoolExecutor(max_workers=config["num_workers"]) as executor:
        futures = {executor.submit(process_time_range, time_range, config): time_range for time_range in config["time_ranges"]}
        for future in concurrent.futures.as_completed(futures):
            time_range = futures[future]
            try:
                future.result()
            except Exception as e:
                print(f"Time range {time_range} generated an exception: {e}")

if __name__ == '__main__':
    main()
    # test()
