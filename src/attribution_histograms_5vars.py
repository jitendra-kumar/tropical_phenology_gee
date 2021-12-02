## by bharat sharma
### Date of this code first created: Nov 23, 2021
### for attribution summary plots of analysis with 5 vars

# Import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
from pathlib import Path
import glob
import os

#  Enter the region of interest
region = "puerto_rico"

path_attr = f"/gpfs/alpine/cli137/proj-shared/ud4/{region}_attr_summary/"

#save_path = "/gpfs/alpine/cli137/proj-shared/ud4/costa_rica/"
save_path=path_attr

# Attribution Codes
# ===========

Codes = {
    9999 : "too few values",
    1110 : "neg NDRE ext driven by neg tp anomaly ",
    1120 : "neg NDRE ext driven by pos tp anomaly ",
    1210 : "neg NDRE ext driven by neg t2m anomaly ",
    1220 : "neg NDRE ext driven by pos t2m anomaly ",
    1310 : "neg NDRE ext driven by neg swvl anomaly ",
    1320 : "neg NDRE ext driven by pos swvl anomaly ",
    1410 : "neg NDRE ext driven by neg ssrd anomaly ",
    1420 : "neg NDRE ext driven by pos ssrd anomaly ",
    1510 : "neg NDRE ext driven by neg aet anomaly ",
    1520 : "neg NDRE ext driven by pos aet anomaly ",
    2110 : "pos NDRE ext driven by neg tp anomaly ",
    2120 : "pos NDRE ext driven by pos tp anomaly ",
    2210 : "pos NDRE ext driven by neg t2m anomaly ",
    2220 : "pos NDRE ext driven by pos t2m anomaly ",
    2310 : "pos NDRE ext driven by neg swvl anomaly ",
    2320 : "pos NDRE ext driven by pos swvl anomaly ",
    2410 : "pos NDRE ext driven by neg ssrd anomaly ",
    2420 : "pos NDRE ext driven by pos ssrd anomaly ",
    2510 : "pos NDRE ext driven by neg aet anomaly ",
    2520 : "pos NDRE ext driven by pos aet anomaly ",
}


x_text_no999 = [r"$-$ve CC - $-$ve TP", 
                r"$-$ve CC - $+$ve TP", 
                r"$-$ve CC - $-$ve T2M", 
                r"$-$ve CC - $+$ve T2M", 
                r"$-$ve CC - $-$ve SWVL", 
                r"$-$ve CC - $+$ve SWVL",
                r"$-$ve CC - $-$ve SSRD", 
                r"$-$ve CC - $+$ve SSRD",
                r"$-$ve CC - $-$ve AET", 
                r"$-$ve CC - $+$ve AET",
                r"$+$ve CC - $-$ve TP", 
                r"$+$ve CC - $+$ve TP", 
                r"$+$ve CC - $-$ve T2M", 
                r"$+$ve CC - $+$ve T2M", 
                r"$+$ve CC - $-$ve SWVL", 
                r"$+$ve CC - $+$ve SWVL",
                r"$+$ve CC - $-$ve SSRD", 
                r"$+$ve CC - $+$ve SSRD",
                r"$+$ve CC - $-$ve AET", 
                r"$+$ve CC - $+$ve AET"]

#extract file names
filenames_all = {}
for lag in range(6):
    filenames_all[lag] = save_path + "All_{}_attr_ano_lag_{:02d}.csv".format(region,lag)               

# Summary
data = {}
dict_attr_codes_count = {}
for lag in [0,1,2,3,4,5]: 
    data[lag] = pd.read_csv (filenames_all[lag],
                             sep=' ', header=None) #, dtype=np.int32)
    dict_attr_codes_count["lag "+str(lag)] =  data[lag].sum(axis=0) / (np.array(list(Codes.keys()))+lag)

df_attr_summary_count = pd.DataFrame.from_dict(dict_attr_codes_count)
df_attr_summary_count.index = Codes.keys()
#df_attr_summary_count.to_csv(save_path + "All_CR_Attr_Summary_Count_Corrected.csv", sep=",")
print (df_attr_summary_count)

# Sumamary table
df_attr_summary_count_no999 = df_attr_summary_count.drop(9999)
df_attr_summary_count_no999.columns = ["00 days", "15 days", "30 days" ,
                                       "45 days", "60 days", "75 days"]

df_attr_summary_count_no999 = df_attr_summary_count.drop(9999)

df_attr_summary_count_no999 = df_attr_summary_count.drop(9999)
#df_attr_summary_count_no999.columns = ["00 days", "15 days", "30 days" ]
df_attr_summary_count_no999.columns = ["00 days", "15 days", "30 days" ,
                                       "45 days", "60 days", "75 days"]
#df_attr_summary_count_no999.columns = ["15 days"]

fig,axs = plt.subplots(tight_layout = True, figsize = (12,5))
df_attr_summary_count_no999.plot(ax = axs, kind = 'bar', width = .7, fontsize=12)#, colormap='Purples_r')
axs.grid('--')
axs.set_ylabel (r"Number of Pixels", fontsize =16)
#axs.set_xlabel ("Attr Codes", fontsize =14)
plt.title(f"Attribution of NDRE extremes to Climate Drivers in {region}\n", fontsize=16 )
plt.legend(fontsize=12)
#ax1.set_xticks(x1)
axs.set_xticklabels(x_text_no999, minor=False, rotation=60)
fig.savefig(save_path + f"All_{region}_Attr_Summary_Count_bar_upto_lag{lag}.png")
fig.savefig(save_path + f"All_{region}_Attr_Summary_Count_bar_upto_lag{lag}.pdf")
