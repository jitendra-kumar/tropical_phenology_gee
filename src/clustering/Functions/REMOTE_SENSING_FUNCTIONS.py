import pandas as pd
import numpy as np
import rasterio
from rasterio.plot import show
from osgeo import gdal


def cut_bounding_box(bbox, df): # min_lon, min_lat, max_lon, max_lat, 
    df_lon = df[(df['Longitude'] < bbox[2]) & (df['Longitude'] > bbox[0])]
    df_lat = df_lon[(df_lon['Latitude'] < bbox[3]) & (df_lon['Latitude'] > bbox[1])]
    return df_lat


def if_all_bands_zero(df):
    df = df[df['B2'] + df['B3'] +
                      df['B4'] + df['B5'] + 
                      df['B6'] + df['B7'] + 
                      df['B8'] + df['B8A'] +
                      df['B11'] + df['B12'] != 0]
    return df

def sample_S2(pd_groupby_obj, show_plots, num_comparisons):

    grouped_dfs = []
    for i in pd_groupby_obj:
        grouped_dfs.append(i[1])
        
    lis = []
    S2_months = list(range(6, num_comparisons+1, 1))
    i = 0

    global shot_coord_list

    for agb_group, S2 in zip(grouped_dfs, S2_months):

        try:
            bands_data = pd.DataFrame()
            wd = '/mnt/locutus/remotesensing/r62/Sentinel_2_Data/Costa_Rica_Monthly/Combined/monthly_{S2}'.format(S2=S2)
            os.chdir(wd)

            df = agb_group
            shot_coord_list = np.array([(x,y) for x,y in zip(df['Longitude'] , df['Latitude'])])
            filepaths = [g for g in os.listdir() if g.endswith('.tif')]
            vrt_path = '/mnt/locutus/remotesensing/r62/Sentinel_2_Data/Costa_Rica_Monthly/Combined/temp.vrt'
            gdal.BuildVRT(vrt_path, filepaths)

            with rasterio.open(vrt_path, 'r') as raster:

                if show_plots == True:
                    figure(figsize=(8, 6), dpi=80)
                    show(raster)

                bands_data['B2'] = [-9999 if np.isnan(sample[0]) else sample[0] for sample in raster.sample(shot_coord_list.astype('float32'))]
                bands_data['B3'] = [-9999 if np.isnan(sample[1]) else sample[1] for sample in raster.sample(shot_coord_list.astype('float32'))]
                bands_data['B4'] = [-9999 if np.isnan(sample[2]) else sample[2] for sample in raster.sample(shot_coord_list.astype('float32'))]
                bands_data['B5'] = [-9999 if np.isnan(sample[3]) else sample[3] for sample in raster.sample(shot_coord_list.astype('float32'))]
                bands_data['B6'] = [-9999 if np.isnan(sample[4]) else sample[4] for sample in raster.sample(shot_coord_list.astype('float32'))]
                bands_data['B7'] = [-9999 if np.isnan(sample[5]) else sample[5] for sample in raster.sample(shot_coord_list.astype('float32'))]
                bands_data['B8'] = [-9999 if np.isnan(sample[6]) else sample[6] for sample in raster.sample(shot_coord_list.astype('float32'))]
                bands_data['B8A'] = [-9999 if np.isnan(sample[7]) else sample[7] for sample in raster.sample(shot_coord_list.astype('float32'))]
                bands_data['B11'] = [-9999 if np.isnan(sample[8]) else sample[8] for sample in raster.sample(shot_coord_list.astype('float32'))]
                bands_data['B12'] = [-9999 if np.isnan(sample[9]) else sample[9] for sample in raster.sample(shot_coord_list.astype('float32'))]
                bands_data['Shot_Number'] = np.array(list((map(str, df['Shot_Number']))))
                bands_data['Group_Number'] = i
                
                i += 1
                lis.append(bands_data)
                print('Iteration:', i, 'complete')

        except:

            i += 1
            print('On Iteration:', i, "there wasn't any GEDI or S2 data at that time interval")

    return pd.concat(lis)


