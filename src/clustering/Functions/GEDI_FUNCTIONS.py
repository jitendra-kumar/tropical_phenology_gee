import requests as r
import pandas as pd
import numpy as np
import datetime
import pickle
import h5py
import sys
import time 



def convert_time(df):
    df = df.sort_values(by = ['Time'])

    list_of_times = []

    for i in df['Time']:

        start_time = datetime.datetime(2018, 1, 1)
        added_seconds = datetime.timedelta(0, i)
        new_datetime = start_time + added_seconds
        list_of_times.append(new_datetime)

    return list_of_times

def seperate_GPV(LIST):
    '''removes the first item of the list that is a duplicate.'''
    
    for i in LIST:
        front = i[0:46]
        matching = [s for s in LIST if s.startswith(front)]
        if len(matching) > 1:
            LIST.remove(matching[0])
    
    return LIST


def get_data(beam, gedi_level, file_string, rh_perc_start=0, rh_perc_stop=100, get_rh_data=False):

    '''Goes into the GEDI files within the working directory and extracts
    data based on whichever variables are selected for below (you need to adjust the
    code accordingly if you wish to extract different variables). 
    The results are: an array with all of the float32 variables, an array with all of the 
    uint8 variables, a one dimensional array with all of the shot numbers (converted to
    string so as to prevent truncation!) and optionally an array of rh_data if '2A' is selected
    (as a dataframe). rh_perc_start and ...stop are the lowest and highest values of rh data
    that you would want to collect, respectively.'''
    
    gedi_file = h5py.File(file_string, 'r')
    
    if gedi_level == '4A':

        shot_num = gedi_file[beam + '/shot_number'][()].astype('U64')     
        
        # float32 columns
        latitude = gedi_file[beam + '/lat_lowestmode' ][()].astype('float32')
        longitude = gedi_file[beam + '/lon_lowestmode' ][()].astype('float32')
        agbd = gedi_file[beam + '/agbd'][()].astype('float32')
        agbd_se = gedi_file[beam + '/agbd_se'][()].astype('float32')
        time = gedi_file[beam + '/delta_time'][()].astype('float32')
        
        # uint8 columns
        beamID = gedi_file[beam + '/beam'][()].astype('uint8')
        plant_type = gedi_file[beam + '/land_cover_data/pft_class'][()].astype('uint8')
        l4A_quality = gedi_file[beam + '/l4_quality_flag'][()].astype('uint8')
        degrade = gedi_file[beam + '/degrade_flag'][()].astype('uint8')
        
        HDF5_arr_float = np.vstack((latitude, longitude, agbd, agbd_se, time))
        HDF5_arr_int = np.vstack((beamID, plant_type, l4A_quality, degrade))
                
        rh_df = None
        
    elif gedi_level == '2A':
        
        shot_num = gedi_file[beam + '/shot_number'][()].astype('U64')
        rh = gedi_file[beam + '/rh'][()].astype('int')
        
        # float32 columns
        latitude = gedi_file[beam + '/lat_lowestmode' ][()].astype('float32')
        longitude = gedi_file[beam + '/lon_lowestmode' ][()].astype('float32')
        time = gedi_file[beam + '/delta_time'][()].astype('float32')
        TX = gedi_file[beam + '/digital_elevation_model'][()].astype('float32')
        lowest_elev = gedi_file[beam + '/elev_lowestmode'][()].astype('float32')
        sensitivity = gedi_file[beam + '/sensitivity'][()].astype('float32')
        
        # uint8 columns
        beamID = gedi_file[beam + '/beam'][()].astype('uint8')
        plant_type = gedi_file[beam + '/land_cover_data/pft_class'][()].astype('uint8')      
        quality = gedi_file[beam + '/quality_flag'][()].astype('uint8')

        HDF5_arr_float = np.vstack((latitude, longitude, time, TX, lowest_elev, sensitivity))
        HDF5_arr_int = np.vstack((beamID, plant_type, quality))
        
        if get_rh_data == True:
            rh_names = ['rh%d'%(r) for r in range(rh_perc_start, rh_perc_stop+1)]
            rh_df = pd.DataFrame(rh, columns=rh_names)
            
        else:
            
            rh_df = None
            
   
	return (HDF5_arr_float, HDF5_arr_int, shot_num, rh_df)


def get_files(doi, cmrurl, bound, start_date, end_date, country_name):

	''' Goes into the ORNL DAAC and pulls out all of 
	the specified L4A GEDI files for our given boundary box. 
	Then populates a text file called GEDI04_A_GranuleList.txt 
	which contains the list of all of the URLs that were pulled. '''

	doisearch = cmrurl + 'collections.json?doi=' + doi
	concept_id = r.get(doisearch).json()['feed']['entry'][0]['id']
	print("The Concept ID is: ", concept_id)


	dt_format = '%Y-%m-%dT%H:%M:%SZ'
	temporal_str = start_date.strftime(dt_format) + ',' + end_date.strftime(dt_format)

	bound_str = ','.join(map(str, bound))

	page_num = 1
	page_size = 2000 # CMR page size limit

	granule_arr = []

	while True:
    
		# defining parameters
		cmr_param = {
			"collection_concept_id": concept_id, 
			"page_size": page_size,
			"page_num": page_num,
			"temporal": temporal_str,
			"bounding_box[]": bound_str
		}
    
		granulesearch = cmrurl + 'granules.json'

		response = r.get(granulesearch, params=cmr_param)
		granules = response.json()['feed']['entry']
    
		if granules:
			for g in granules:
				granule_url = ''
            
				for links in g['links']:
					if 'title' in links and links['title'].startswith('Download') \
					and links['title'].endswith('.h5'):
						granule_url = links['href']
				granule_arr.append(granule_url)
               
			page_num += 1
		else: 
			break
   
	df = pd.DataFrame(granule_arr)
	df.to_csv('GEDI04_A_GranuleList_{country_name}.txt'.format(country_name=country_name), header=None, index=None, sep='\t')

def get_gedi_L1_or_L2_files(product, bbox):
    
    '''Goes into the NASA DAAC and retrieves data corresponding to the 
    boundary box specified for the product specified then populates the 
    URLs for that that data in a text file within the working directory:
    
    Options include 'GEDI01_B.002', 'GEDI02_A.002', 'GEDI02_B.002'
    bounding box coordinates in LL Longitude, LL Latitude, UR Longitude, UR Latitude format
    
    Source:
    https://lpdaac.usgs.gov/resources/e-learning/spatial-querying-of-gedi-version-2-data-in-python/
    '''
    bbox = str(bbox)[1:-1]    
    def gedi_finder(product, bbox):

        # Define the base CMR granule search url, including LPDAAC provider name and max page size (2000 is the max allowed)
        cmr = "https://cmr.earthdata.nasa.gov/search/granules.json?pretty=true&provider=LPDAAC_ECS&page_size=2000&concept_id="

        # Set up dictionary where key is GEDI shortname + version
        concept_ids = {'GEDI01_B.002': 'C1908344278-LPDAAC_ECS', 
                       'GEDI02_A.002': 'C1908348134-LPDAAC_ECS', 
                       'GEDI02_B.002': 'C1908350066-LPDAAC_ECS'}

        # CMR uses pagination for queries with more features returned than the page size
        page = 1
        bbox = bbox.replace(' ', '')  # remove any white spaces
        try:
            # Send GET request to CMR granule search endpoint w/ product concept ID, bbox & page number, format return as json
            cmr_response = r.get(f"{cmr}{concept_ids[product]}&bounding_box={bbox}&pageNum={page}").json()['feed']['entry']

            # If 2000 features are returned, move to the next page and submit another request, and append to the response
            while len(cmr_response) % 2000 == 0:
                page += 1
                cmr_response += r.get(f"{cmr}{concept_ids[product]}&bounding_box={bbox}&pageNum={page}").json()['feed']['entry']

            # CMR returns more info than just the Data Pool links, below use list comprehension to return a list of DP links
            return [c['links'][0]['href'] for c in cmr_response]
        except:
            # If the request did not complete successfully, print out the response from CMR
            print(r.get(f"{cmr}{concept_ids[product]}&bounding_box={bbox.replace(' ', '')}&pageNum={page}").json())

    granules = gedi_finder(product, bbox)
    print(f"{len(granules)} {product} Version 2 granules found.")

    # Set up output text file name using the current datetime
    outName = f"{product.replace('.', '_')}_GranuleList_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}.txt"

    # Open file and write each granule link on a new line
    with open(outName, "w") as gf:
        for g in granules:
            gf.write(f"{g}\n")
    print(f" A file containing links to intersecting {product} Version 2
			data has been saved to:\n {os.getcwd()}/{outName}")


		  
def pickle_GEDI_HDF5(beams, file_list, gedi_level, cols_float, cols_int):
    
    '''You need to know the number of cols that you are extracting that are type float32 and 
    how many that are type uint8 (cols_float and cols_int). This function creates binary files from 
    the GEDI data you are extracting from in the working directory. Each binary file cannot surpass
    900MB in order to prevent memory issues. 
    '''
    
    combined_arrs_float = np.array([np.nan]*cols_float).reshape(cols_float, 1)
    combined_arrs_int = np.array([np.nan]*cols_int).reshape(cols_int, 1)
    shots_combined = []
    
    i = 0
    problem_files = []
    
    for idx, hdf5_file in enumerate(file_list):
        for beam in beams:        
            
            try:
                 
                results = get_data(beam = beam, gedi_level = gedi_level, file_string = hdf5_file, get_rh_data = False)
                array_float = results[0]
                array_int = results[1]
                shot_number = results[2]
                combined_arrs_float = combine_arrays(combined_arrs_float, array_float)
                combined_arrs_int = combine_arrays(combined_arrs_int, array_int)
                shots_combined.extend(shot_number)

                if (sys.getsizeof(shots_combined) + combined_arrs_float.size + combined_arrs_int.size >= 900000000) or ((idx == len(file_list)-1) and (beam == beams[-1])):

                    i += 1
                    bin_name = "./GEDI_{gedi_level}_{number}.pkl".format(gedi_level = gedi_level, number = i)
                    df = pd.DataFrame(np.hstack((combined_arrs_float.T, combined_arrs_int.T)))
                    df = df[1:]
                    df['Shot_Number'] = shots_combined
                    df.to_pickle(bin_name)
                    combined_arrs_float = np.array([np.nan]*cols_float).reshape(cols_float, 1)
                    combined_arrs_int = np.array([np.nan]*cols_int).reshape(cols_int, 1)
                    shots_combined = []    

                elif  sys.getsizeof(shots_combined) + combined_arrs_float.size + combined_arrs_int.size < 900000000:

                    continue


            except Exception as e:
                
                problem_files.append(hdf5_file)
                logger.error('This GEDI file failed because: '+ str(e))
                
    print('The problem files are:', set(problem_files))
		  
def pickle_rh_data(beams, file_list, gedi_level, get_rh_data):
    
    '''Same as above but used exclusively for the rh_data'''
    
    combined_arrs = np.array([np.nan]*101).reshape(101, 1)
    shots = []
    i = 0
    problem_files = []

    for idx, hdf5_file in enumerate(file_list):
        for beam in beams:        
            
            try:
                 
                results = get_data(beam = beam, gedi_level = gedi_level, file_string = hdf5_file, get_rh_data = get_rh_data)
                rh_data = np.array(results[3]).T
                shot_number = results[2]
                combined_arrs = combine_arrays(combined_arrs, rh_data)
                shots.extend(shot_number)
                                
                if (sys.getsizeof(shots) + combined_arrs.size >= 900000000) or ((idx == len(file_list)-1) and (beam == beams[-1])):
                    
                    i += 1
                    bin_name = "./GEDI_2A_rh_data_{number}.pkl".format(number = i)
                    df = pd.DataFrame(combined_arrs.T)
                    df = df[1:]
                    df['Shot_Number'] = shots
                    df.to_pickle(bin_name)
                    combined_arrs = np.array([np.nan]*101).reshape(101, 1)
                    shots = []    
                
                elif  combined_arrs.size + sys.getsizeof(shots) < 900000000:
                    
                    continue
                    

            except Exception as e:
                
                problem_files.append(hdf5_file)
                logger.error('This rh data file failed because: '+ str(e))
                
    print('The problem files are: ', set(problem_files))
		  
def subset_rh_files(rh_files, shot_list_keep):
    
    df_list = []
    for rh_fil in rh_files:
        df = pd.read_pickle(rh_fil)
        df = df[df['Shot_Number'].isin(shot_list_keep)]
        df_list.append(df)
    df = pd.concat(df_list)
    df.to_pickle('./rh_data_final')    
    
    
    
def preprocess_L2A(pkl_files, bbox):
    
    '''Basic preprocessing steps to be taken with the given 
    gedi_level 2A data considering the variables'''
    
    column_names = ['Latitude', 'Longitude', 'Time', 'TanDEMX', 'Lowest_Elevation', 'Sensitivity', 'BeamID', 'PFT', 'Quality', 'Shot_Number']
    dfs = []

    for fil in pkl_files:
    
        df = pd.read_pickle(fil)
        df.columns = column_names
        
        df = cut_bounding_box(bbox, df) #Using CR bbox
        # df = df.where(df['Quality'].ne(0)) # removed this 11/1/22
        df = df[df['Quality'] != 0] # this instead!
        df = df[df['PFT'] != 0]
        df = df[(df['Sensitivity'] >= 0.97) & (df['Sensitivity'] <= 1)] # 12/13/22 made the sensitivity stricter
        # df = df[df['TanDEMX'] != -9.999990e+05] # 12/13/22 stop doing this correction!
        # df = df[abs(df['Lowest_Elevation'] - df['TanDEMX']) <= 3.0] # 12/13/22 stop doing this correction!
        df = df.dropna()
        dfs.append(df)
    
    final = pd.concat(dfs)
    final = final.to_pickle('./Final_Level_2A_Data.pkl')
    return final
		  
def preprocess_L4A(pkl_files):
    
    '''Basic preprocessing steps to be taken with the given 
    gedi_level 4A data considering the variables'''
    
    column_names = ['Latitude', 'Longitude', 'AGB', 'AGB_stand_err', 'Time', 'BeamID', 'PFT', 'L4A_Quality', 'Degrade', 'Shot_Number']
    dfs = []

    for fil in pkl_files:
    
        df = pd.read_pickle(fil)
        df.columns = column_names
        
        df = cut_bounding_box(bbox, df) #Using CR bbox (Left-right, Bottom-left, Right-right, Top-left)
        # df = df.where(df['L4A_Quality'].ne(0)) # 11/1/22
        df = df[df['L4A_Quality'] != 0]            
        df = df[df['Degrade'] == 0]    
        df = df[df['PFT'] != 0]
        df = df[df['AGB'] != -9999.0]
        df = df.dropna()
        dfs.append(df)
    
    final = pd.concat(dfs)
    final = final.to_pickle('./Final_Level_4A_Data.pkl')
    return final