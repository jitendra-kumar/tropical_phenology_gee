import sys
import cdsapi

varname=['2m_temperature', 'surface_solar_radiation_downwards', 'surface_sensible_heat_flux', 'surface_latent_heat_flux', 'total_precipitation', 'runoff', 'volumetric_soil_water_layer_1', 'volumetric_soil_water_layer_2', 'volumetric_soil_water_layer_3', 'volumetric_soil_water_layer_4', 'total_evaporation', 'evaporation_from_vegetation_transpiration', 'potential_evaporation', '2m_dewpoint_temperature']
var=['t2m', 'ssrd', 'sshf', 'slhf', 'tp', 'ro', 'swvl1', 'swvl2', 'swvl3', 'swvl4', 'e', 'evavt', 'pev', '2d']

#varname=['evaporation_from_vegetation_transpiration']
#var=['evavt']


year=int(sys.argv[1])

#c = cdsapi.Client()
#
#c.retrieve(
#    'reanalysis-era5-land',
#    {
#        'format': 'netcdf',
#        'variable': [
##            '2m_temperature', 
##			'surface_solar_radiation_downwards',
##			'surface_sensible_heat_flux', 
##			'surface_latent_heat_flux',
##			'total_precipitation', 
##			'runoff',
##			'volumetric_soil_water_layer_1',
##			'volumetric_soil_water_layer_2',
##			'volumetric_soil_water_layer_3',
##			'volumetric_soil_water_layer_4',
##			'total_evaporation', 
##			'evaporation_from_vegetation_transpiration',
##			'potential_evaporation',
#            '2m_dewpoint_temperature', 
#        ],
#		'expver': '1',
#        'year': '2022',
#        'month': [
#            '01', '02', '03',
#            '04', '05', '06',
#            '07', '08', '09',
#            '10', '11', '12',
#        ],
#        'day': [
#            '01', '02', '03',
#            '04', '05', '06',
#            '07', '08', '09',
#            '10', '11', '12',
#            '13', '14', '15',
#            '16', '17', '18',
#            '19', '20', '21',
#            '22', '23', '24',
#            '25', '26', '27',
#            '28', '29', '30',
#            '31',
#        ],
##       'time': [
##           '00:00', '01:00', '02:00',
##           '03:00', '04:00', '05:00',
##           '06:00', '07:00', '08:00',
##           '09:00', '10:00', '11:00',
##           '12:00', '13:00', '14:00',
##           '15:00', '16:00', '17:00',
##           '18:00', '19:00', '20:00',
##           '21:00', '22:00', '23:00',
##       ],
#        'area': [
#            11.5, -86.5,
#			7, -77.0,
#        ],
#    },
#    '2022_daily_costarica_panama_3.nc')
#

def download_era5land_var(varname, var, year, month):
	c = cdsapi.Client()
	
	c.retrieve(
	    'reanalysis-era5-land',
	    {
	        'format': 'netcdf',
			'variable': varname,
	#       'variable': [
	#            '2m_temperature', 
	#			'surface_solar_radiation_downwards',
	#			'surface_sensible_heat_flux', 
	#			'surface_latent_heat_flux',
	#			'total_precipitation', 
	#			'runoff',
	#			'volumetric_soil_water_layer_1',
	#			'volumetric_soil_water_layer_2',
	#			'volumetric_soil_water_layer_3',
	#			'volumetric_soil_water_layer_4',
	#			'total_evaporation', 
	#			'evaporation_from_vegetation_transpiration',
	#			'potential_evaporation',
	#           '2m_dewpoint_temperature', 
	#       ],
	        'year': '%d'%(year),
	        'month': ['%02d'%(month)
	#           '01', '02', '03',
	#           '04', '05', '06',
	#           '07', '08', '09',
	#           '10', '11', '12',
	        ],
	        'day': [
	            '01', '02', '03',
	            '04', '05', '06',
	            '07', '08', '09',
	            '10', '11', '12',
	            '13', '14', '15',
	            '16', '17', '18',
	            '19', '20', '21',
	            '22', '23', '24',
	            '25', '26', '27',
	            '28', '29', '30',
	            '31',
	        ],
	       'time': [
	           '00:00', '01:00', '02:00',
	           '03:00', '04:00', '05:00',
	           '06:00', '07:00', '08:00',
	           '09:00', '10:00', '11:00',
	           '12:00', '13:00', '14:00',
	           '15:00', '16:00', '17:00',
	           '18:00', '19:00', '20:00',
	           '21:00', '22:00', '23:00',
	       ],
	        'area': [
	            11.5, -86.5,
				7, -77.0,
	        ],
	    },
	    '%d_%02d_daily_costarica_panama_%s.nc'%(year, month, var))
	
nvars=len(var)
for v in range(nvars):
	for month in range(12,12+1):
		download_era5land_var(varname[v], var[v], year, month)

