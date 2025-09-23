import pandas as pd
import ee

class GeeExtractor():
    def __init__(self, project, data_id, object_variable, point_data_path, output_path):
        self.project = project
        self.gee_data_id = data_id
        self.obj_var = object_variable
        self.point_data_path = point_data_path
        self.output_path = output_path
        self.point_df = None
        self.ee_fc = None
        self.results = None
        self.authenticate_and_initialize_ee()  # Initialize GEE in the base class

    def authenticate_and_initialize_ee(self):
        """Authenticate and initialize Google Earth Engine (GEE)."""
        ee.Authenticate()
        ee.Initialize(project=self.project)
        print("Authenticated and initialized Earth Engine.")

    def read_point_data(self):
        """Read point data from a CSV file."""
        point_data = pd.read_csv(self.point_data_path)
        self.point_df = point_data.filter(items=['ID', 'lat', 'lon'])
        print("Point data loaded.")

    def convert_dataframe_to_feature_collection(self):
        """Convert the pandas dataframe into a GEE Feature Collection."""
        features = []
        for _, row in self.point_df.iterrows():
            poi_geometry = ee.Geometry.Point([row['lon'], row['lat']])
            poi_properties = dict(row)
            poi_feature = ee.Feature(poi_geometry, poi_properties)
            features.append(poi_feature)

        self.ee_fc = ee.FeatureCollection(features)
        print("Feature collection created.")
    
    def get_object_data(self):
        image = ee.Image(self.gee_data_id)
        self.obj_data = image
        print("Object data retrieved.")


    def extract_data(self, scale=11132):
        """Extract data for each point location using the DEM dataset."""

        self.reults = self.obj_data.sampleRegions(collection=self.ee_fc, scale=scale)

    def convert_to_dataframe(self):
        """Convert the GEE results to a pandas DataFrame and save as CSV."""
        # Sample a result to extract the structure of the data
        sample_result =  self.reults.first().getInfo()

        # Extract column names from the first result
        column_df = list(sample_result['properties'].keys())

        # Get nested data from GEE result
        nested_list =  self.reults.reduceColumns(ee.Reducer.toList(len(column_df)), column_df).values().get(0)
        data = nested_list.getInfo()

        # Create a DataFrame and save it as a CSV file
        df = pd.DataFrame(data, columns=column_df)
        output_file = f'{self.output_path}/{self.obj_var}.csv'
        df.to_csv(output_file, index=False)
        print(f"Data for {self.obj_var} saved to {output_file}.")

    def stastic_variables_extract(self):
        """Main method to run the DEM extraction process."""
        self.read_point_data()
        self.get_object_data()
        self.convert_dataframe_to_feature_collection()
        self.extract_data()
        self.convert_to_dataframe()


