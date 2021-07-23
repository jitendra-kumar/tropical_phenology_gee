# Bharat Sharma
# To find anomalies and extremes in NDRE and attribute to climate drivers
# with compound effects and lags
# Designed to run with mpi4py
# Read : Instructions_NDRE.md for setting up the conda env

import numpy as np
import pandas as pd
from scipy import stats
import argparse
import os

from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

parser  = argparse.ArgumentParser()
parser.add_argument('--filename_NDRE'   ,'-f_ndre'    , help = "Filepath/filename of ndre"        , type= str,    default= None       )
parser.add_argument('--filename_Precip' ,'-f_pr'      , help = "Filepath/filename of precip"      , type= str,    default= None       )
parser.add_argument('--filename_Temp'   ,'-f_tas'     , help = "Filepath/filename of temperature" , type= str,    default= None       )
parser.add_argument('--save_ext_loc'    ,'-save_loc'  , help = "Do you want to save loc of ext? (y,n)", type= str,    default= 'n'    )
parser.add_argument('--attribution_type','-attr_typ'  , help = "Attribute using driver ano or ori ts?", type= str,    default= 'ano'  )
parser.add_argument('--save_ano_file'  ,'-save_f_ano', help = "Do you want to save anomaly file?", type= str,    default= 'y'  )
args = parser.parse_args()

# The inputs:
# ===========
file_NDRE  = str   (args.filename_NDRE)
file_PR    = str   (args.filename_Precip)
file_TAS   = str   (args.filename_Temp)
save_ext_loc = str   (args.save_ext_loc)
attr_type = str   (args.attribution_type)
save_ano_file = str   (args.save_ano_file)

# Running the file
#New Version : `detection_disturbance_with_loc.py`
#  * Saves locations of extremes if required , `-save_loc y` or `-save_loc yes`
#  * will gerenetate different directory for smooth and interp files
#time srun -n 32 python /gpfs/alpine/cli137/proj-shared/6ru/proj_analysis/Detection_Attribution/detection_disturbance_with_loc.py \
#        -f_ndre /gpfs/alpine/cli137/proj-shared/6ru/proj_analysis/am_0.smooth \
#        -f_pr /gpfs/alpine/cli137/proj-shared/6ru/proj_analysis/am_0.precip \
#        -f_tas /gpfs/alpine/cli137/proj-shared/6ru/proj_analysis/am_0.temp \
#        -save_loc no \
#        -save_f_ano yes



# Reading the data
# ================
data_NDRE= np.genfromtxt(file_NDRE, dtype = np.float16, delimiter=" ")
data_pr= np.genfromtxt  (file_PR, dtype = np.float16, delimiter=" ")
data_temp= np.genfromtxt(file_TAS, dtype = np.float16, delimiter=" ")


# Function to calculate anomalies
# ===============================
def Calc_Anomalies (data_ar):
    """
    Input : np array of shape (n,100)
    where 'n' is number of pixels
    100: is biweekly values
    
    Output: Anomalies
    i.e. TS of first 3 years - benchmark TS
    benchmark TS = av of (p,p+25,p+50, p+1,p+26,p+51, ..... , p+24,p+49,p+74) x3 
    """
    # Checking the shape of the input matrix
    np.testing.assert_array_equal(data_ar.shape, (data_ar.shape[0], 100),
                                  err_msg= 'The shape of the Input matrix should be (n , 100)')
    
    # Only keeping first 75 values i.e. 3 years of data
    data_ar = data_ar[:,:75] 
    # Masking -ve values 
    data_ar = np.ma.masked_less_equal(data_ar, 0)
    # data_NDRE = tmp.data * ~tmp.mask # if masking with zeros
    
    # Aim: to calculate the benchmark ts per pixel ...
    # ... average annual ts of 3 years
    data_ar_re = np.reshape(data_ar,(data_ar.shape[0],25,3))
    data_ar_av_1yr = np.ma.average(data_ar_re,axis=2)
    data_ar_bench = np.concatenate((data_ar_av_1yr,data_ar_av_1yr,data_ar_av_1yr), axis=1)
    # anomalies = NDRE -  3year average yearly TS 
    data_ar_anomalies = data_ar - data_ar_bench
    return data_ar_anomalies

# Calculation of anomalies
# ========================
if attr_type == "ano":
    data_NDRE_anomalies = Calc_Anomalies (data_NDRE)
    data_PR_anomalies   = Calc_Anomalies (data_pr)
    data_TAS_anomalies = Calc_Anomalies (data_temp)
elif attr_type == "ori":
    data_NDRE_anomalies = Calc_Anomalies (data_NDRE)
    data_PR_anomalies   = data_pr [:,:75]
    data_TAS_anomalies  = data_temp [:,:75]


if (rank == 0) and (attr_type == "ano") and (save_ano_file in ['y','yes']):
    # Saving anomalies
    # ================
    filename_ano_ndre =  '.'.join(file_NDRE.split('.')[:-1]) + '_ano.' +file_NDRE.split('.')[-1]
    filename_ano_pr   =  '.'.join(file_PR.split('.')[:-1]) + '_ano.' +file_PR.split('.')[-1]
    filename_ano_tas  =  '.'.join(file_TAS.split('.')[:-1]) + '_ano.' +file_TAS.split('.')[-1]


    # NDRE
    np.savetxt(filename_ano_ndre,
           data_NDRE_anomalies,
           fmt='%10.4f',
           delimiter = ",")
    # Pr
    np.savetxt(filename_ano_pr,
           data_PR_anomalies,
           fmt='%10.4f',
           delimiter = ",")
    # Tas
    np.savetxt(filename_ano_tas,
           data_TAS_anomalies,
           fmt='%10.4f',  
           delimiter = ",")

if (rank == 0) and (attr_type == "ori") :
    # Saving anomalies
    # ================
    filename_ano_ndre =  '.'.join(file_NDRE.split('.')[:-1]) + '_ano.' +file_NDRE.split('.')[-1]
    # NDRE
    np.savetxt(filename_ano_ndre,
           data_NDRE_anomalies,
           fmt='%10.4f',
           delimiter = ",")


# Attribution
# ===========

Codes = {
    99 : "too few values",
    10 : "neg NDRE ext driven by dry",
    20 : "neg NDRE ext driven by hot",
    30 : "neg NDRE ext driven by wet",
    40 : "neg NDRE ext driven by cold",
    60 : "pos NDRE ext driven by dry",
    70 : "pos NDRE ext driven by hot",
    80 : "pos NDRE ext driven by wet",
    90 : "pos NDRE ext driven by cold",
    100: "neg NDRE ext driven by dry and dry",
    110: "pos NDRE ext driven by wet and cold",
    
}



def Attribution_Drivers_Codes_1025 (ts_ndre_ano, ts_pr_ano, ts_tas_ano,lag=0):
    """
    Similar to Attribution_Drivers_Codes, but here we compare the driver during 10p of NDRE with 25q of its own value and remove all grids with negative values
    
    Checks the climatic conditions during negative and postive NDRE extremes at a pixel.
    Assigns codes of attributions.
     
     999     : 'too few values',
     10 + lag: 'neg NDRE ext driven by dry',
     20 + lag: 'neg NDRE ext driven by hot',
     30 + lag: 'neg NDRE ext driven by wet',
     40 + lag: 'neg NDRE ext driven by cold',
     60 + lag: 'pos NDRE ext driven by dry',
     70 + lag: 'pos NDRE ext driven by hot',
     80 + lag: 'pos NDRE ext driven by wet',
     90 + lag: 'pos NDRE ext driven by cold',
     100+ lag: 'neg NDRE ext driven by dry and dry',
     110+ lag: 'pos NDRE ext driven by wet and cold'
     
     
    Parameters
    ----------
    ts_ndre_ano : ts of a pixel of NDRE
    ts_pr_ano : ts of a pixel of NDRE
    ts_tas_ano : ts of a pixel of NDRE
    lag: lag timestep, default =0
    
    Returns
    -------
    The attributions codes array of a pixel
    """
    # To include the effect of lag
    if lag > 0 :
        ts_ndre_ano = ts_ndre_ano[lag:]
        ts_pr_ano   = ts_pr_ano  [:-lag]
        ts_tas_ano  = ts_tas_ano [:-lag]

    # initializing the codes array per pixel with zeros and will fill as conditions are met
    ts_codes_px = np.zeros((len(Codes)))

    ts_ndre_ano_non_mask_vals = ts_ndre_ano[~ts_ndre_ano.mask]
    if ts_ndre_ano_non_mask_vals.size < 75-int(lag) : 
        code_px = 999
        ts_codes_px[0] = code_px
        # Printing zeros for location of extremes
        loc_10q = np.zeros((75-int(lag)))
        loc_90q = np.zeros((75-int(lag)))
    else:
        # Interannual Variability = standard deviation of anomalies
        iav_px = np.array(ts_ndre_ano_non_mask_vals).std()
        # Mean value of NDRE anomalies of 10th quarlite
        loc_10q = ts_ndre_ano<np.percentile(ts_ndre_ano_non_mask_vals,10)
        px_ndre_10q = ts_ndre_ano[loc_10q].mean()
        # Mean value of NDRE anomalies of 90th quarlite
        loc_90q = ts_ndre_ano>np.percentile(ts_ndre_ano_non_mask_vals,90)
        px_ndre_90q = ts_ndre_ano[loc_90q].mean()

        # Mean precipiration and temperature during < 10q NDRE
        pr_du_neg  = ts_pr_ano[loc_10q].mean()
        tas_du_neg = ts_tas_ano[loc_10q].mean()
        # Mean precipiration and temperature during > 90q NDRE
        pr_du_pos  = ts_pr_ano[loc_90q].mean()
        tas_du_pos = ts_tas_ano[loc_90q].mean()

        # 25q of precipitation
        pr_25q = np.percentile(ts_pr_ano,25)
        # 75q of precipitation
        pr_75q = np.percentile(ts_pr_ano,75)

        # 25q of tas
        tas_25q = np.percentile(ts_tas_ano,25)
        # 75q of tas
        tas_75q = np.percentile(ts_tas_ano,75)

        if pr_du_neg < pr_25q :
            # neg NDRE ext driven by dry : 10+ lag
            code_px = 10+ lag
            ts_codes_px[1] = code_px
        if tas_du_neg > tas_75q :
            # neg NDRE ext driven by hot : 20+ lag
            code_px = 20+ lag
            ts_codes_px[2] = code_px
        if pr_du_neg > pr_75q :
            # neg NDRE ext driven by wet : 30+ lag
            code_px = 30+ lag
            ts_codes_px[3] = code_px
        if tas_du_neg < tas_25q :
            # neg NDRE ext driven by cold : 40+ lag
            code_px = 40+ lag
            ts_codes_px[4] = code_px

        if pr_du_pos < pr_25q :
            # pos NDRE ext driven by dry : 60+ lag
            code_px = 60+ lag
            ts_codes_px[5] = code_px
        if tas_du_pos > tas_75q :
            # pos NDRE ext driven by hot : 70+ lag
            code_px = 70+ lag
            ts_codes_px[6] = code_px
        if pr_du_pos > pr_75q :
            # pos NDRE ext driven by wet : 80+ lag
            code_px = 80+ lag
            ts_codes_px[7] = code_px
        if tas_du_pos < tas_25q :
            # pos NDRE ext driven by cold : 90+ lag
            code_px = 90+ lag
            ts_codes_px[8] = code_px

        if (10+ lag in ts_codes_px) and (20+ lag in ts_codes_px):
            # neg NDRE ext driven by dry and dry : 100 + lag
            code_px = 100+ lag
            ts_codes_px[9] = code_px

        if (80+ lag in ts_codes_px) and (90+ lag in ts_codes_px):
            # pos NDRE ext driven by wet and cold : 110 + lag
            code_px = 110 + lag
            ts_codes_px[10] = code_px

    return ts_codes_px, loc_10q, loc_90q,iav_px


# After this every rank will do a part of the calculations
# ========================================================   

if data_NDRE_anomalies.shape[0]%size == 0:
    px_per_rank = data_NDRE_anomalies.shape[0]//size
    load_divisor = size
else:
    load_divisor = size -1
    if rank < size-1:
        px_per_rank = data_NDRE_anomalies.shape[0]//(size-1)
    elif rank == size-1:
        px_per_rank = data_NDRE_anomalies.shape[0]%(size-1)
    
for lag in range(6):
    attr_ar = np.zeros((px_per_rank, len(Codes)))
    ar_loc_neg = np.zeros((px_per_rank,75- int(lag)))
    ar_loc_pos = np.zeros((px_per_rank,75- int(lag)))
    iav_ndre   = np.zeros((px_per_rank,1))
    for i in range(px_per_rank):
        ts_ndre_ano = data_NDRE_anomalies[rank*int(data_NDRE_anomalies.shape[0]/load_divisor)+i,:]
        ts_pr_ano   = data_PR_anomalies  [rank*int(data_NDRE_anomalies.shape[0]/load_divisor)+i,:]
        ts_tas_ano  = data_TAS_anomalies[rank*int(data_NDRE_anomalies.shape[0]/load_divisor)+i,:]
        attr_ar[i],ar_loc_neg[i],ar_loc_pos[i], iav_ndre[i] = Attribution_Drivers_Codes_1025 (ts_ndre_ano, ts_pr_ano, ts_tas_ano, lag=lag)

        # Saving attribution data
        # -----------------------
        path_attr = ("/").join(file_NDRE.split('/')[:-1]) + '/' +  ("_").join((file_NDRE.split('/')[-1]).split("."))+'/'
        if os.path.isdir(path_attr) == False:
            os.makedirs(path_attr)
        # Attr
        filename_attr = path_attr + f"attr_{attr_type}_lag_{str(lag).zfill(2)}_rank_{str(rank).zfill(3)}.csv"
        np.savetxt(filename_attr,
               attr_ar,
               fmt='%i',
               delimiter = " ")
        #IAV
        filename_iav = path_attr + f"iav_ndre_ano_{attr_type}_lag_{str(lag).zfill(2)}_rank_{str(rank).zfill(3)}.csv"
        np.savetxt(filename_iav,
               iav_ndre,
               delimiter = " ")

        if save_ext_loc in ['y','yes','YES','Y']:
            filename_neg_loc = path_attr + f"neg_loc_{attr_type}_lag_{str(lag).zfill(2)}_rank_{str(rank).zfill(3)}.csv"
            np.savetxt(filename_neg_loc,
                   ar_loc_neg,
                   fmt='%i',
                   delimiter = " ")
            filename_pos_loc = path_attr + f"pos_loc_{attr_type}_lag_{str(lag).zfill(2)}_rank_{str(rank).zfill(3)}.csv"
            np.savetxt(filename_pos_loc,
                   ar_loc_pos,
                   fmt='%i',
                   delimiter = " ")

        
print ("Success : ", rank )
