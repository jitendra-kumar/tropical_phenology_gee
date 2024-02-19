import pandas as pd 
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

font = {'family' : 'serif',
        'weight' : 'bold',
        'size'   : 18}

matplotlib.rc('font', **font)

# Read phenometrics univar
phenometrics_h10=pd.read_csv("costa_rica_panama.phen_params.seasonality.2022.h10.univar", sep=",")

dict_seasonality = []

for index, row in phenometrics_h10.iterrows():
	dict_seasonality.append({"label" : int(row["label"]), 
			                  "mean": row["mean"], 
							  "med": row["median"], 
							  "q1" : row["first_quart"],
							  "q3": row["third_quart"], 
							  "whislo": row["min"], 
							  "whishi": row["max"], 
							  "cilo": row["perc_10"], 
							  "ciho": row["perc_90"],
							  })
print(dict_seasonality)

fig, ax = plt.subplots(figsize=(20,10))
ax.bxp(dict_seasonality, showmeans=True, meanline=True, showfliers=False)
ax.set_ylim([0,0.2])
ax.set_xlabel("Phenoregions")
ax.set_ylabel("Phenological Seasonality")
fig.tight_layout()
fig.savefig("costa_rica_panama.phen_params.seasonality.2022.h10.univar.png", bbox_inches='tight', dpi=300)


# Read phenometrics univar
phenometrics_probav=pd.read_csv("costa_rica_panama.phen_params.seasonality.2022.probav.univar", sep=",")

dict_seasonality = []

for index, row in phenometrics_probav.iterrows():
	if row["label"] != "None1":
		dict_seasonality.append({"label" : row["label"], 
			                  "mean": row["mean"], 
							  "med": row["median"], 
							  "q1" : row["first_quart"],
							  "q3": row["third_quart"], 
							  "whislo": row["min"], 
							  "whishi": row["max"], 
							  "cilo": row["perc_10"], 
							  "ciho": row["perc_90"],
							  })
print(dict_seasonality)

fig, ax = plt.subplots(figsize=(20,10))
ax.bxp(dict_seasonality, showmeans=True, meanline=True, showfliers=False)
ax.set_ylim([0,0.2])
ax.set_xlabel("Landcover")
ax.set_ylabel("Phenological Seasonality")
fig.tight_layout()
fig.savefig("costa_rica_panama.phen_params.seasonality.2022.probav.univar.png", bbox_inches='tight', dpi=300)
