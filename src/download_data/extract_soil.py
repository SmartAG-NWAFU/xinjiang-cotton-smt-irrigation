from gee_extractor import GeeExtractor
from pathlib import Path

def soil_extract(data_path, output_path):
    """Extract soil data from SoilGrids using GEE."""
    object_variables = ['sand_mean', 'silt_mean', 'clay_mean', 'bdod_mean', 'soc_mean']

    for obj_var in object_variables:

        extractor = GeeExtractor(data_id=f'projects/soilgrids-isric/{obj_var}', 
                                object_variable= obj_var,
                                point_data_path=data_path, 
                                output_path=output_path, 
                                project="ee-xinjiangcotton")
        extractor.stastic_variables_extract()


def unit_soil_extract():
    """Extract soil data for each grid unit."""
    data_path = "../../data/grid_10km/xinjiang_cotton_units.csv"
    output_path = Path("../../data/grid_10km/soil")
    output_path.mkdir(parents=True, exist_ok=True)
    soil_extract(data_path, output_path)

def site_soil_extract():
    """Extract soil data for each site."""
    data_path = "../../data/xinjiang_zones/experimental_sites.csv"
    output_path = Path("../../data/sites/soil")
    output_path.mkdir(parents=True, exist_ok=True)
    soil_extract(data_path, output_path)


if __name__ == '__main__':
    # site_soil_extract()
    unit_soil_extract()