import pandas as pd
import numpy as np
import rasterio
from rasterio.transform import from_origin
from rasterio.enums import Resampling

class RasterConverter:
    """
    A class to convert a CSV (or pandas DataFrame) containing latitude, longitude, and attribute values 
    into raster (TIFF) files. The class allows for customization of grid resolution, coordinate reference 
    system (CRS), and other parameters. Multiple raster files can be generated for each value column provided.

    Attributes:
    ----------
    df : pandas.DataFrame
        The input DataFrame containing latitude, longitude, and attribute columns.
    value_columns : list of str
        The list of column names containing the attribute values to be used for rasterization.
    res : float
        The grid resolution, in degrees. Default is 0.1.
    output_dir : str
        The directory where the TIFF files will be saved. The file names are automatically generated.
    lat_column : str
        The name of the column containing latitude values. Default is 'lat'.
    lon_column : str
        The name of the column containing longitude values. Default is 'lon'.
    crs : str
        The coordinate reference system (CRS) for the output rasters. Default is 'EPSG:4326'.
    
    Methods:
    --------
    csv_to_tif():
        Converts the input DataFrame into multiple raster (TIFF) files based on the provided value columns.
    """
    
    def __init__(self, df, value_columns, res=0.1, output_dir='./', lat_column='lat', lon_column='lon', crs='EPSG:4326'):
        """
        Initializes the RasterConverter instance with the required parameters.

        Parameters:
        ----------
        df : pandas.DataFrame
            The input DataFrame containing the latitude, longitude, and attribute values.
        value_columns : list of str
            A list of column names containing the attribute values to be used for rasterization.
        res : float, optional
            The grid resolution in degrees (default is 0.1).
        output_dir : str, optional
            The directory where the TIFF files will be saved (default is './').
        lat_column : str, optional
            The name of the column containing latitude values (default is 'lat').
        lon_column : str, optional
            The name of the column containing longitude values (default is 'lon').
        crs : str, optional
            The coordinate reference system (CRS) for the output rasters (default is 'EPSG:4326').
        """
        
        self.df = df
        self.value_columns = value_columns
        self.res = res
        self.output_dir = output_dir
        self.lat_column = lat_column
        self.lon_column = lon_column
        self.crs = crs

    def csv_to_tif(self):
        """
        Converts the input DataFrame into multiple raster (TIFF) files based on the provided value columns. 
        Each value column will be converted to a separate raster file. The raster grids are created based 
        on the latitude and longitude values, with an optional grid resolution. The output TIFF files are 
        saved in the specified directory with automatically generated file names.

        The method assumes that the DataFrame contains columns for latitude, longitude, and one or more 
        attribute values to populate the raster. Missing or out-of-bound values are treated as NaN.

        Returns:
        --------
        None
            The method does not return any values but creates multiple raster (TIFF) files in the specified directory.
        """
        
        # Step 1: Extract latitude, longitude, and other value columns
        lons = self.df[self.lon_column].values
        lats = self.df[self.lat_column].values

        # Step 2: Calculate the raster grid dimensions (width and height)
        min_lon, max_lon = min(lons), max(lons)
        min_lat, max_lat = min(lats), max(lats)

        width = int((max_lon - min_lon) / self.res) + 1
        height = int((max_lat - min_lat) / self.res) + 1

        # Step 3: Iterate over each value column to generate a raster file
        for value_column in self.value_columns:
            # Extract the attribute values for the current column
            values = self.df[value_column].values

            # Step 4: Create an empty raster array filled with NaN (to handle missing data)
            raster_data = np.full((height, width), np.nan, dtype=np.float32)

            # Step 5: Convert latitude and longitude values to raster grid indices
            x_indices = ((lons - min_lon) / self.res).astype(int)
            y_indices = ((max_lat - lats) / self.res).astype(int)

            # Step 6: Populate the raster data array with attribute values
            for i in range(len(lons)):
                raster_data[y_indices[i], x_indices[i]] = values[i]

            # Step 7: Define the raster's upper-left corner coordinates
            left_top_corner = (min_lon, max_lat)

            # Step 8: Define the affine transform for the raster (georeferencing)
            transform = from_origin(left_top_corner[0], left_top_corner[1], self.res, self.res)

            # Step 9: Construct a valid output file name from the value column name
            valid_value_column = value_column.replace(" ", "_").replace("(", "").replace(")", "").replace("/", "_")
            output_file = f"{self.output_dir}/{valid_value_column}.tif"
            with rasterio.open(output_file, 'w', driver='GTiff', height=height, width=width, count=1, dtype=np.float32,
                               crs=self.crs, transform=transform) as dst:
                dst.write(raster_data, 1)

            # Step 10: Output success message for the current raster file
            print(f'TIFF file successfully created: {output_file}')