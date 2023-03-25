import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 

variables=['t2m', 'tp', 'swvl1', 'swvl2', 'swvl3', 'swvl4', 'vpd', 'ndre_1', 'ndre_2', 'ndre_3', 't2m_1', 't2m_2', 't2m_3', 'tp_1', 'tp_2', 'tp_3', 'swvl1_1', 'swvl1_2', 'swvl1_3', 'swvl2_1', 'swvl2_2', 'swvl2_3', 'swvl3_1', 'swvl3_2', 'swvl3_3', 'swvl4_1', 'swvl4_2', 'swvl4_3', 'vpd_1', 'vpd_2', 'vpd_3']

subset=8
scores=pd.read_csv("shap_values_h10_%d.csv"%(subset), delimiter=",", header=None)
orders=pd.read_csv("shap_order_h10_%d.csv"%(subset), delimiter=",", header=None)
orders=orders.astype(int)
counts=orders[0].value_counts()
varname = np.empty([len(counts)], dtype="S10")
print(counts)
summary = pd.DataFrame()
for i in range(len(counts)):
#	print(counts.index[0])
	if counts.index[i] != 9999:
		print(counts.index[i])
		varname[i] = variables[int(counts.index[i])]
		print(varname[i])
	else:
		varname[i] = 'No data'

summary['varid'] = counts.index
summary['varname'] = varname
summary['count'] = counts.iloc[:].values

print(summary)
summary.to_csv("sumary_%d.csv"%(subset))

