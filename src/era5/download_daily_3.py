import cdsapi

c = cdsapi.Client()

c.retrieve(
    'reanalysis-era5-land',
    {
        'format': 'netcdf',
        'variable': [
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
            '2m_dewpoint_temperature', 
        ],
        'year': '2021',
        'month': [
            '01', '02', '03',
            '04', '05', '06',
            '07', '08', '09',
            '10', '11', '12',
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
			# Costa Rica + Panama
            11.5, -86.5,
			7, -77.0,
			# Brazil
#            6.0, -76.5,
#			-27.0, -47.5,
			# Puerto Rico
#            19.0, -68.0,
#			18.0, -65.0,
        ],
    },
    '2021_daily_costarica_panama_3.nc')
#    '2021_daily_brazil_3.nc')
#    '2021_daily_puertorico_3.nc')



