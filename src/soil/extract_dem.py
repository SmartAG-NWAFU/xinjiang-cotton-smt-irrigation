from gee_extractor import GeeExtractor 


def dem_extract(data_path, output_path):
    '''Extract dem data from srtm90_v4 files'''

    # extract dem data from srtm90_v4 files
    dem_extractor = GeeExtractor(data_id="CGIAR/SRTM90_V4", 
                                    object_variable="elevation",
                                    point_data_path=data_path, 
                                    output_path=output_path, 
                                    project="ee-xinjiangcotton")
    dem_extractor.stastic_variables_extract()



def unit_dem_extract():
    '''Extract dem data for each grid unit'''

    data_path = "../../data/grid_10km/xinjiang_cotton_units.csv"
    output_path = "../../data/grid_10km/"
    dem_extract(data_path, output_path)


def site_dem_extract():
    '''Extract dem data for each site'''

    data_path = "../../data/xinjiang_zones/experimental_sites.csv"
    output_path = "../../data/sites/"
    dem_extract(data_path, output_path)


if __name__ == '__main__':
    # site_dem_extract()
    unit_dem_extract()
