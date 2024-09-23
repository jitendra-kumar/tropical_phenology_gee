import xarray as xr
import pandas as pd
import numpy as np
import rioxarray as rio


varname=['2m_temperature', 'surface_solar_radiation_downwards', 'surface_sensible_heat_flux', 'surface_latent_heat_flux', 'total_precipitation', 'runoff', 'volumetric_soil_water_layer_1', 'volumetric_soil_water_layer_2', 'volumetric_soil_water_layer_3', 'volumetric_soil_water_layer_4', 'total_evaporation', 'evaporation_from_vegetation_transpiration', 'potential_evaporation', '2m_dewpoint_temperature']
var=['t2m', 'ssrd', 'sshf', 'slhf', 'tp', 'ro', 'swvl1', 'swvl2', 'swvl3', 'swvl4', 'e', 'evavt', 'pev', 'd2m']

data_dir='/mnt/locutus/remotesensing/tropics/data/era5_data/costarica_panama'
region="costarica_panama"
# loop over years 
for t in range(2022, 2022+1):
	print("Temporally resampling hourly data to 15 days: Year %d Region: %s"%(t,region))

	# Loop over variables 

	inds = xr.Dataset({})
	outds = xr.Dataset({})
	for v in var:
		print("Variable: %s"%(v))
		print('Reading files %s/%d_*_daily_%s_%s.nc'%(data_dir, t, region, v))
		inds1=xr.open_mfdataset('%s/%d_*_daily_%s_%s.nc'%(data_dir, t, region, v))
		if v == 'evavt':
			inds1['longitude'] = inds['longitude']
		#inds=inds.merge(inds1)
		if 'expver' in inds1.dims:
			inds[v] =inds1[v].sel(expver=1, drop=True)
		else:
			inds[v] = inds1[v]

		# resampled and add to output dataset
		outds[v] = inds[v].resample(time='15D').mean()
		outds[v].attrs = inds[v].attrs
		if v == 'd2m':
			# calculate VPD
			# https://bmcnoldy.rsmas.miami.edu/Humidity.html
			rh = 100*(np.exp((17.625*(inds.d2m.values-273.16))/(243.04+(inds.d2m.values-273.16)))/np.exp((17.625*(inds.t2m.values-273.16))/(243.04+(inds.t2m.values-273.16))))
			
			# https://en.wikipedia.org/wiki/Vapour-pressure_deficit
			# Convert kelvin to rankine -- R = 1.8*K
			# Use Arrhenius equation to calculate saturate vapor pressure
		##	vpsat = np.exp((-1.0440397 * math.pow(10, 4)/(inds1.t2m.values*1.8)) +
		##			(-11.29465) +
		##			(-2.7022355*math.pow(10,-2)*(inds1.t2m.values*1.8)) +
		##			(1.289036 * math.pow(10,-5)*(inds1.t2m.values*1.8)*(inds1.t2m.values*1.8)) +
		##			(-2.4780681*math.pow(10,-9)*(inds1.t2m.values*1.8)*(inds1.t2m.values*1.8)*(inds1.t2m.values*1.8)) +
		##			(6.5459673*np.log((inds1.t2m.values*1.8)))
		##			)
			# https://bsapubs.onlinelibrary.wiley.com/doi/pdfdirect/10.3732/ajb.1700247
			#  https://doi.org/10.3732/ajb.1700247
			vpsat = (0.61078 * np.exp(17.27*(inds.t2m.values-273.16) / ((inds.t2m.values-273.16) + 237.3))) 
			vpd =(vpsat * (1- rh/100))
			inds['vpd']=inds['d2m']
			inds['vpd'][:,:,:] = vpd
			inds['vpd'].attrs['long_name'] = 'Vapor Pressure Deficit'
			inds['vpd'].attrs['standard_name'] = 'Vapor_Pressure_Deficit'
			inds['vpd'].attrs['units'] = 'kPa'

			outds['vpd'] = inds['vpd'].resample(time='15D').mean()
			outds['vpd'].attrs = inds['vpd'].attrs
			
            # Create a new variable which is weighted sum of volumetric water content
			# for 0-100cm depths
			outds['swvl_1m'] = (7.0/100.0)*outds['swvl1'] + (21.0/100.0)*outds['swvl2'] + (72.0/100.0)*outds['swvl3']
			outds['swvl_1m'].attrs['long_name'] = 'Volumetric water content for 0-100cm'
			outds['swvl_1m'].attrs['units'] = outds['swvl1'].attrs['units']

	print(inds)

#	# set 1 with variables: t2m (2 metre temperature), 
#	#						ssrd (Surface solar radiation downwards), 
#	#						sshf (Surface sensible heat flux), 
#	#						slhf (Surface latent heat flux), 
#	#						tp (Total precipitation), 
#	#						ro (Runoff)
#
#	# load the file
#	inds1=xr.open_dataset('%d_daily_%s_1.nc'%(t, region))
#	
#	outds = xr.Dataset({})
#	outds['t2m'] = inds1.t2m.resample(time='15D').mean()
#	outds['ssrd'] = inds1.ssrd.resample(time='15D').mean()
#	outds['sshf'] = inds1.sshf.resample(time='15D').mean()
#	outds['slhf'] = inds1.slhf.resample(time='15D').mean()
#	outds['tp'] = inds1.tp.resample(time='15D').sum()
#	outds['ro'] = inds1.ro.resample(time='15D').sum()
#	outds.t2m.attrs = inds1.t2m.attrs
#	outds.ssrd.attrs = inds1.ssrd.attrs
#	outds.sshf.attrs = inds1.sshf.attrs
#	outds.slhf.attrs = inds1.slhf.attrs
#	outds.tp.attrs = inds1.tp.attrs
#	outds.ro.attrs = inds1.ro.attrs
##	outds.rio.write_crs("epsg:4326", inplace=True)
#	
##	outds.to_netcdf('15d_%d_daily_%s_EPSG4326_1.nc'%(t, region),
##			mode='w', format='NETCDF4_CLASSIC')
#
#	# set 2 with variables: swvl1 (Volumetric soil water layer 1),
#   	#						swvl2 (Volumetric soil water layer 2),
#	#						swvl3 (Volumetric soil water layer 3),
#	#						swvl4 (Volumetric soil water layer 4),
#	#						e (Evaporation),
#	#						evavt (Evaporation from vegetation transpiration),
#	#						pev (Potential evaporation)
#
#	# load the file
#	inds2=xr.open_dataset('%d_daily_%s_2.nc'%(t, region))
#	
#	outds['swvl'] = inds2.swvl1.resample(time='15D').mean() + inds2.swvl2.resample(time='15D').mean() + inds2.swvl3.resample(time='15D').mean() + inds2.swvl4.resample(time='15D').mean()
#	outds['swvl1'] =  inds2.swvl1.resample(time='15D').mean()
#	outds['swvl2'] = inds2.swvl1.resample(time='15D').mean()
#	outds['swvl3'] = inds2.swvl1.resample(time='15D').mean()
#	outds['swvl4'] = inds2.swvl1.resample(time='15D').mean()
#	outds['e'] = inds2.e.resample(time='15D').mean()
#	outds['evavt'] = inds2.evavt.resample(time='15D').mean()
#	outds['pev'] = inds2.pev.resample(time='15D').mean()
#	outds.swvl.attrs = inds2.swvl1.attrs
#	outds.swvl.attrs['long_name'] = 'Volumetric soil water layer 1-4'
#	outds.swvl1.attrs = inds2.swvl1.attrs
#	outds.swvl2.attrs = inds2.swvl2.attrs
#	outds.swvl3.attrs = inds2.swvl3.attrs
#	outds.swvl4.attrs = inds2.swvl4.attrs
#	outds.e.attrs = inds2.e.attrs
#	outds.evavt.attrs = inds2.evavt.attrs
#	outds.pev.attrs = inds2.pev.attrs
##	outds.rio.write_crs("epsg:4326", inplace=True)
#	
##	outds.to_netcdf('15d_%d_daily_%s_EPSG4326_2.nc'%(t, region),
##			mode='w', format='NETCDF4_CLASSIC')
#		
#	# set 3 with variables: d2m (2m dew point temperature),
#
#	# load the file
#	inds3=xr.open_dataset('%d_daily_%s_3.nc'%(t, region))
#
#	# calculate vapor pressure deficit
#
#	print(inds1.t2m.shape)
#	print(inds3.d2m.shape)
#	# https://bmcnoldy.rsmas.miami.edu/Humidity.html
#	rh = 100*(np.exp((17.625*(inds3.d2m.values-273.16))/(243.04+(inds3.d2m.values-273.16)))/np.exp((17.625*(inds1.t2m.values-273.16))/(243.04+(inds1.t2m.values-273.16))))
#	
##	# https://en.wikipedia.org/wiki/Vapour-pressure_deficit
##	# Convert kelvin to rankine -- R = 1.8*K
##	# Use Arrhenius equation to calculate saturate vapor pressure
##	vpsat = np.exp((-1.0440397 * math.pow(10, 4)/(inds1.t2m.values*1.8)) +
##			(-11.29465) +
##			(-2.7022355*math.pow(10,-2)*(inds1.t2m.values*1.8)) +
##			(1.289036 * math.pow(10,-5)*(inds1.t2m.values*1.8)*(inds1.t2m.values*1.8)) +
##			(-2.4780681*math.pow(10,-9)*(inds1.t2m.values*1.8)*(inds1.t2m.values*1.8)*(inds1.t2m.values*1.8)) +
##			(6.5459673*np.log((inds1.t2m.values*1.8)))
##			)
#	# https://bsapubs.onlinelibrary.wiley.com/doi/pdfdirect/10.3732/ajb.1700247
#	#  https://doi.org/10.3732/ajb.1700247
#	vpsat = (0.61078 * np.exp(17.27*(inds1.t2m.values-273.16) / ((inds1.t2m.values-273.16) + 237.3))) 
#	vpd =(vpsat * (1- rh/100))
#	inds3['vpd']=inds3.d2m
#	inds3['vpd'][:,:,:] = vpd
##	print(inds3.d2m)
##	print(inds3.time.shape)
##	print(inds3.latitude.shape)
##	print(inds3.longitude.shape)
##	print(vpd.shape)
##	print("Ad VPD to inds3")
##	test=np.zeros((vpd.shape), dtype='float32')	
#
##	print("T2M: Min %f Max %f Median %f"%(inds1.t2m.min() -273.16, inds1.t2m.max() -273.16, inds1.t2m.median() -273.16))
##	print("VPD: Min %f Max %f Median %f"%(inds3.vpd.min(), inds3.vpd.max(), inds3.vpd.median()))
##	print(np.exp(1))
#
#	outds['vpd'] = inds3.vpd.resample(time='15D').mean()
#	outds.vpd.attrs['long_name'] = 'Vapor Pressure Deficit'
#	outds.vpd.attrs['units'] = 'kPa'
#
	outds.rio.write_crs("epsg:4326", inplace=True)
#	print("Write to file")	
	outds.to_netcdf('%s/15d_%d_daily_%s_EPSG4326_allvars_v3.nc'%(data_dir, t, region),
			mode='w', format='NETCDF4_CLASSIC')
