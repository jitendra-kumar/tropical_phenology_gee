import xarray as xr
import pandas as pd
import numpy as np
import rioxarray as rio
import math 

region="costarica_panama"
# loop over years 
for t in range(2017, 2020+1):
	print("Temporally resampling hourly data to 15 days: Year %d Region: %s"%(t,region))
	# set 1 with variables: t2m (2 metre temperature), 
	#						ssrd (Surface solar radiation downwards), 
	#						sshf (Surface sensible heat flux), 
	#						slhf (Surface latent heat flux), 
	#						tp (Total precipitation), 
	#						ro (Runoff)

	# load the file
	inds1=xr.open_dataset('%d_daily_%s_1.nc'%(t, region))
	
	# set 2 with variables: swvl1 (Volumetric soil water layer 1),
   	#						swvl2 (Volumetric soil water layer 2),
	#						swvl3 (Volumetric soil water layer 3),
	#						swvl4 (Volumetric soil water layer 4),
	#						e (Evaporation),
	#						evavt (Evaporation from vegetation transpiration),
	#						pev (Potential evaporation)

	# load the file
#	inds2=xr.open_dataset('%d_daily_%s_2.nc'%(t, region))
		
	# set 3 with variables: d2m (2m dew point temperature),

	# load the file
	inds3=xr.open_dataset('%d_daily_%s_3.nc'%(t, region))

	# calculate vapor pressure deficit

	# https://bmcnoldy.rsmas.miami.edu/Humidity.html
	rh = 100*(np.exp((17.625*(inds3.d2m.values-273.16))/(243.04+(inds3.d2m.values-273.16)))/np.exp((17.625*(inds1.t2m.values-273.16))/(243.04+(inds1.t2m.values-273.16))))
	
#	# https://en.wikipedia.org/wiki/Vapour-pressure_deficit
#	# Convert kelvin to rankine -- R = 1.8*K
#	# Use Arrhenius equation to calculate saturate vapor pressure
#	vpsat = np.exp((-1.0440397 * math.pow(10, 4)/(inds1.t2m.values*1.8)) +
#			(-11.29465) +
#			(-2.7022355*math.pow(10,-2)*(inds1.t2m.values*1.8)) +
#			(1.289036 * math.pow(10,-5)*(inds1.t2m.values*1.8)*(inds1.t2m.values*1.8)) +
#			(-2.4780681*math.pow(10,-9)*(inds1.t2m.values*1.8)*(inds1.t2m.values*1.8)*(inds1.t2m.values*1.8)) +
#			(6.5459673*np.log((inds1.t2m.values*1.8)))
#			)
	# https://bsapubs.onlinelibrary.wiley.com/doi/pdfdirect/10.3732/ajb.1700247
	#  https://doi.org/10.3732/ajb.1700247
	vpsat = (0.61078 * np.exp(17.27*(inds1.t2m.values-273.16) / ((inds1.t2m.values-273.16) + 237.3))) 
	vpd =(vpsat * (1- rh/100))
#	print(inds3.d2m)
	print(inds3.time.shape)
	print(inds3.latitude.shape)
	print(inds3.longitude.shape)
	print(vpd.shape)
	print("Ad VPD to inds3")
	test=np.zeros((vpd.shape), dtype='float32')	
	inds3['vpd']=inds3.d2m
	inds3['vpd'][:,:,:] = vpd

	print("T2M: Min %f Max %f Median %f"%(inds1.t2m.min() -273.16, inds1.t2m.max() -273.16, inds1.t2m.median() -273.16))
	print("VPD: Min %f Max %f Median %f"%(inds3.vpd.min(), inds3.vpd.max(), inds3.vpd.median()))
	print(np.exp(1))

	outds = xr.Dataset({'vpd': inds3.vpd.resample(time='15D').mean()})
	outds.vpd.attrs['long_name'] = 'Vapor Pressure Deficit'
	outds.vpd.attrs['units'] = 'kPa'

	outds.rio.write_crs("epsg:4326", inplace=True)
	print("Write to file")	
	outds.to_netcdf('15d_%d_daily_%s_EPSG4326_VPD.nc'%(t, region),
			mode='w', format='NETCDF4_CLASSIC')
