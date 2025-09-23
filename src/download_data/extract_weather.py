import pandas as pd
import multiprocessing
import ee
from gee_extractor import GeeExtractor
import os
from pathlib import Path


class Era5Extractor(GeeExtractor):
    def __init__(self, project, data_id, time_range, extract_vars, point_data_path, output_path):
        super().__init__(project, data_id, None, point_data_path, output_path)
        self.time_range = time_range
        self.extract_vars = extract_vars
        self.era5_data = None
        self.sacle = 11132 # era5 daily data Pixel Size 11132 meters

    def get_era5_data(self):
        """Retrieve ERA5 data for the defined time range."""
        self.era5_data = ee.ImageCollection(self.gee_data_id) \
            .filterDate(self.time_range[0], self.time_range[1])
        print("ERA5 data retrieved.")

    def convert_dataframe_to_feature_collection(self, point):
        """Convert the pandas dataframe into a GEE Feature Collection."""
        features = []
        for _, row in point.iterrows():
            poi_geometry = ee.Geometry.Point([row['lon'], row['lat']])
            poi_properties = dict(row)
            poi_feature = ee.Feature(poi_geometry, poi_properties)
            features.append(poi_feature)

        ee_fc = ee.FeatureCollection(features)
        return ee_fc
    
    def extract_point_data(self, era5_data, ee_fc):
        """Extract data for each point location using the ERA5 dataset."""
        def addDate(image):
            img_date = ee.Date(image.date())
            img_date = ee.Number.parse(img_date.format('YYYYMMdd'))
            return image.addBands(ee.Image(img_date).rename('date').toInt())

        def rasterExtraction(image):
            feature = image.sampleRegions(
                collection=ee_fc,
                scale=self.sacle,  # Scale in meters  
            )
            return feature

        results = era5_data.filterBounds(ee_fc).select(self.extract_vars)\
                .map(addDate).map(rasterExtraction).flatten()
        return results

    def convert_to_dataframe(self, results, point):
        """Convert the results from GEE to a pandas DataFrame."""
        sample_result = results.first().getInfo()
        column_df = list(sample_result['properties'].keys())

        nested_list = results.reduceColumns(ee.Reducer.toList(len(column_df)), column_df).values().get(0)
        data = nested_list.getInfo()

        df = pd.DataFrame(data, columns=column_df)
        df = df.filter(self.extract_vars + ['date'])

        file_name = point['ID'].iloc[0]
        if isinstance(file_name, (int, float)):
            file_name = int(file_name)  # Convert to integer if it's a number
        output_file = f'{self.output_path}/{file_name}.csv'
        df.to_csv(output_file, index=False)
        print(f"Data saved for point {file_name}.")


    def checking_point_data_weather_exits(self, point):
        """Check if the output file for a point already exists."""
        file_name = point['ID'].iloc[0]
        if isinstance(file_name, float):
            output_file = f'{self.output_path}/{int(file_name)}.csv'
        else:
            output_file = f'{self.output_path}/{file_name}.csv'
        return os.path.exists(output_file)

    def one_point_extract(self, row):
        """Extract one point data."""
        # Initializes GEE within each child process
        ee.Authenticate()
        ee.Initialize(project=self.project)
        # Retrieve ERA5 data
        era5_data = ee.ImageCollection(self.gee_data_id).filterDate(self.time_range[0], self.time_range[1])
        point = pd.DataFrame([row])
        if self.checking_point_data_weather_exits(point):
            print(f"Data for point {int(point['ID'].iloc[0])} already exists. Skipping.")
            return
        ee_fc = self.convert_dataframe_to_feature_collection(point)
        results = self.extract_point_data(era5_data, ee_fc)
        self.convert_to_dataframe(results, point)

    def multi_point_extract(self, num_workers=4):
        """Loop through multiple points and extract data for each in parallel."""
        with multiprocessing.Pool(processes=num_workers) as pool:
            pool.map(self.one_point_extract, [row for _, row in self.point_df.iterrows()])

    def run_extracting_sequence(self):
        # Retrieve data, and load points
        self.get_era5_data()
        self.read_point_data()
        # Extract data for one point and save as CSV
        for _, row in self.point_df.iterrows():
            self.one_point_extract(row)    

    def run_extracting_sequence_parallel(self):
        # Retrieve data, and load points
        # self.get_era5_data()
        self.read_point_data()
        # Extract data for multiple points and save as CSV using multiprocessing
        self.multi_point_extract(num_workers=10)  # Set the number of parallel workers here


def weather_extract(time_ranges, data_path, output_path):
    """Extract weather data for multiple time ranges."""

    project = "ee-xinjiangcotton"
    data_id = "ECMWF/ERA5_LAND/DAILY_AGGR"
    extract_vars = ['total_precipitation_sum', 'temperature_2m_min','temperature_2m_max',
                    'surface_net_solar_radiation_sum','u_component_of_wind_10m','v_component_of_wind_10m']
    
    def check_output_path(output_path):
        """Check if the output path exists, and if not, create it."""
        if not os.path.exists(output_path):
            os.makedirs(output_path)
            print(f"Output path '{output_path}' created.")
        return output_path

    for time_range in time_ranges:
        print(f"Processing time range: {time_range}")
        output_path_update = check_output_path(os.path.join(output_path, f"{time_range[0]}_{time_range[1]}"))

        # Create an instance of the ERA5Extractor class
        extractor = Era5Extractor(project=project, data_id=data_id,
                                time_range=time_range, extract_vars=extract_vars, 
                                point_data_path=data_path, output_path=output_path_update)

        # Run the extracting sequence
        extractor.run_extracting_sequence_parallel()
        # Run the extracting sequence by one point
        # extractor.run_extracting_sequence()


def unit_weather_extract():
    """Extract weather data for each grid unit."""
    data_path = "../../data/grid_10km/xinjiang_cotton_units.csv"
    output_path = Path("../../data/grid_10km/weather")
    output_path.mkdir(parents=True, exist_ok=True)
    time_ranges = [("2000-01-01", "2022-12-31")]
    weather_extract(time_ranges, data_path, output_path)


def site_weather_extract():
    """Extract weather data for each site."""
    data_path = "../../data/xinjiang_zones/experimental_sites.csv"
    output_path = "../../data/sites/"
    time_ranges = [("2000-01-01", "2022-12-31")]
    weather_extract(time_ranges, data_path, output_path)


if __name__ == '__main__':
    # site_weather_extract()
    unit_weather_extract()












