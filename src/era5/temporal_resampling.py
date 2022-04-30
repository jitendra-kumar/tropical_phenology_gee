import xarray as xr
import pandas as pd
import numpy as np
import rioxarray as rio

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
	inds=xr.open_dataset('%d_daily_%s_1.nc'%(t, region))
	
	outds = xr.Dataset({
				't2m': inds.t2m.resample(time='15D').mean(),
				'ssrd': inds.ssrd.resample(time='15D').mean(),
				'sshf': inds.sshf.resample(time='15D').mean(),
				'slhf': inds.slhf.resample(time='15D').mean(),
				'tp': inds.tp.resample(time='15D').sum(),
				'ro': inds.ro.resample(time='15D').sum()
				})
	outds.t2m.attrs = inds.t2m.attrs
	outds.ssrd.attrs = inds.ssrd.attrs
	outds.sshf.attrs = inds.sshf.attrs
	outds.slhf.attrs = inds.slhf.attrs
	outds.tp.attrs = inds.tp.attrs
	outds.ro.attrs = inds.ro.attrs
	outds.rio.write_crs("epsg:4326", inplace=True)
	
	outds.to_netcdf('15d_%d_daily_%s_EPSG4326_1.nc'%(t, region),
			mode='w', format='NETCDF4_CLASSIC')

	# set 2 with variables: swvl1 (Volumetric soil water layer 1),
   	#						swvl2 (Volumetric soil water layer 2),
	#						swvl3 (Volumetric soil water layer 3),
	#						swvl4 (Volumetric soil water layer 4),
	#						e (Evaporation),
	#						evavt (Evaporation from vegetation transpiration),
	#						pev (Potential evaporation)

	# load the file
	inds=xr.open_dataset('%d_daily_%s_2.nc'%(t, region))
	
	outds = xr.Dataset({
				'swvl': inds.swvl1.resample(time='15D').mean() + inds.swvl2.resample(time='15D').mean() + inds.swvl3.resample(time='15D').mean() + inds.swvl4.resample(time='15D').mean(),
				'e': inds.e.resample(time='15D').mean(),
				'evavt': inds.evavt.resample(time='15D').mean(),
				'pev': inds.pev.resample(time='15D').mean(),
				})
	outds.swvl.attrs = inds.swvl1.attrs
	outds.swvl.attrs['long_name'] = 'Volumetric soil water layer 1-4'
	outds.e.attrs = inds.e.attrs
	outds.evavt.attrs = inds.evavt.attrs
	outds.pev.attrs = inds.pev.attrs
	outds.rio.write_crs("epsg:4326", inplace=True)
	
	outds.to_netcdf('15d_%d_daily_%s_EPSG4326_2.nc'%(t, region),
			mode='w', format='NETCDF4_CLASSIC')

