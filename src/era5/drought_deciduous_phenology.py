import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 

# Read the list of deciduous sites and their NDRE time series
#landscape
#dsites_data = pd.read_csv("/data1/jbk/projects/climate/tropics/marcos_longo/drought_deciduous_sites_MarcosLongo.ndre", delimiter=" ")
#asha
dsites_data = pd.read_csv("/home/jbk/projects/climate/tropics/marcos_longo/drought_deciduous_sites_MarcosLongo.ndre", delimiter="|")

# extract NDRE timesereis for BCI - 25 x 6 = 150 values
bci_ndre = dsites_data[dsites_data.code == "BCI"].loc[:, dsites_data.columns.str.contains('n20')]
pnm_ndre = dsites_data[dsites_data.code == "PNM"].loc[:, dsites_data.columns.str.contains('n20')]

# create list of dates
dates = pd.date_range('2017-01-01', periods=25, freq='15D')
dates = dates.union(pd.date_range('2018-01-01', periods=25, freq='15D'))
dates = dates.union(pd.date_range('2019-01-01', periods=25, freq='15D'))
dates = dates.union(pd.date_range('2020-01-01', periods=25, freq='15D'))
dates = dates.union(pd.date_range('2021-01-01', periods=25, freq='15D'))
dates = dates.union(pd.date_range('2022-01-01', periods=25, freq='15D'))

plt.plot(dates, bci_ndre.to_numpy()[0])
plt.ylim([0,1])
plt.show()
