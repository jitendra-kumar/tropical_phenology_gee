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
parser.add_argument('--filename_NDRE'       ,'-f_ndre'  , help = "Filepath/filename of ndre"            , type= str,    default= None       )
parser.add_argument('--filename_Precip'     ,'-f_tp'    , help = "Filepath/filename of precip"          , type= str,    default= None       )
parser.add_argument('--filename_temp'       ,'-f_t2m'   , help = "Filepath/filename of temperature"     , type= str,    default= None       )
parser.add_argument('--filename_soilMoist'  ,'-f_swvl'  , help = "Filepath/filename of soil moisture"   , type= str,    default= None       )
parser.add_argument('--filename_shortRad'   ,'-f_ssrd'  , help = "Filepath/filename of short radiation" , type= str,    default= None       )
parser.add_argument('--filename_AET'        ,'-f_aet'   , help = "Filepath/filename of evap-trans"      , type= str,    default= None       )
parser.add_argument('--filepath_out'        ,'-f_pathout' , help = "Filepath of output attr files in tile split combination"           , type= str,    default= None       )
parser.add_argument('--save_ext_loc'        ,'-save_loc'  , help = "Do you want to save loc of ext? (y,n)", type= str,    default= 'y'    )
parser.add_argument('--attribution_type'    ,'-attr_typ'  , help = "Attribute using driver ano or ori ts?", type= str,    default= 'ano'  )
parser.add_argument('--save_ano_file'       ,'-save_f_ano', help = "Do you want to save anomaly file?", type= str,    default= 'y'  )
args = parser.parse_args()

# The inputs:
# ===========
file_NDRE  = str   (args.filename_NDRE)
file_tp    = str   (args.filename_Precip)
file_t2m   = str   (args.filename_temp)
file_swvl    = str   (args.filename_soilMoist)
file_ssrd    = str   (args.filename_shortRad)
file_aet    = str   (args.filename_AET)
fpath_out  = str   (args.filepath_out)
save_ext_loc = str   (args.save_ext_loc)
attr_type = str   (args.attribution_type)
save_ano_file = str   (args.save_ano_file)


if os.path.isdir(fpath_out) == False:
            os.makedirs(fpath_out)
if os.path.isdir(fpath_out+'anomalies/') == False:
            os.makedirs(fpath_out+'anomalies/')

# Running the file
#New Version : `detection_disturbance_with_loc.py`
#  * Saves locations of extremes if required , `-save_loc y` or `-save_loc yes`
#  * will gerenetate different directory for smooth and interp files
#time srun -n 32 python /gpfs/alpine/cli137/proj-shared/6ru/proj_analysis/Detection_Attribution/detection_disturbance_with_loc_5vars.py \
#        -f_ndre /mnt/locutus/remotesensing/tropics/data/s2_data_raw/costa_rica_ascii/costa_rica_biweekly_1/costa_rica_biweekly_1_timeseries.split.11.smooth \
#        -f_tp /mnt/locutus/remotesensing/tropics/costa_rica/costa_rica_era5_vars/costa_rica_biweekly_1_timeseries.split.11.tp \
#        -f_t2m /mnt/locutus/remotesensing/tropics/costa_rica/costa_rica_era5_vars/costa_rica_biweekly_1_timeseries.split.11.t2m \
#        -f_swvl /mnt/locutus/remotesensing/tropics/costa_rica/costa_rica_era5_vars/costa_rica_biweekly_1_timeseries.split.11.swvl \
#        -f_ssrd /mnt/locutus/remotesensing/tropics/costa_rica/costa_rica_era5_vars/costa_rica_biweekly_1_timeseries.split.11.ssrd \
#        -f_aet /mnt/locutus/remotesensing/tropics/costa_rica/costa_rica_era5_vars/costa_rica_biweekly_1_timeseries.split.11.aet \
#        -f_pathout /mnt/locutus/remotesensing/tropics/costa_rica/attr_codes_t_001_s_011/ \
#        -save_loc y \
#        -save_f_ano yes



# Reading the data
# ================
data_NDRE= np.genfromtxt(file_NDRE, dtype = np.float16, delimiter=" ")
data_tp= np.genfromtxt  (file_tp, dtype = np.float16, delimiter=" ")
data_t2m= np.genfromtxt(file_t2m, dtype = np.float16, delimiter=" ")
data_swvl= np.genfromtxt  (file_swvl, dtype = np.float16, delimiter=" ")
data_ssrd= np.genfromtxt  (file_ssrd, dtype = np.float16, delimiter=" ")
data_aet= np.genfromtxt  (file_aet, dtype = np.float16, delimiter=" ")

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
    data_tp_anomalies   = Calc_Anomalies (data_tp)
    data_t2m_anomalies  = Calc_Anomalies (data_t2m)
    data_swvl_anomalies = Calc_Anomalies (data_swvl)
    data_ssrd_anomalies = Calc_Anomalies (data_ssrd)
    data_aet_anomalies  = Calc_Anomalies (data_aet)

elif attr_type == "ori":
    data_NDRE_anomalies = Calc_Anomalies (data_NDRE)
    data_tp_anomalies   = data_tp [:,:75]
    data_t2m_anomalies  = data_t2m [:,:75] 
    data_swvl_anomalies = data_swvl [:,:75] 
    data_ssrd_anomalies = data_ssrd [:,:75] 
    data_aet_anomalies  = data_aet [:,:75] 


if (rank == 0) and (attr_type == "ano") and (save_ano_file in ['y','yes']):
    # Saving anomalies
    # ================
    filename_ano_ndre =  '.'.join(file_NDRE.split('/')[-1].split('.')[:-1]) + '_ano.' +file_NDRE.split('/')[-1].split('.')[-1]
    filename_ano_tp   =  '.'.join(file_tp.split('/')[-1].split('.')[:-1]) + '_ano.' +file_tp.split('/')[-1].split('.')[-1]
    filename_ano_t2m   =  '.'.join(file_t2m.split('/')[-1].split('.')[:-1]) + '_ano.' +file_t2m.split('/')[-1].split('.')[-1]
    filename_ano_swvl   =  '.'.join(file_swvl.split('/')[-1].split('.')[:-1]) + '_ano.' +file_swvl.split('/')[-1].split('.')[-1]
    filename_ano_ssrd   =  '.'.join(file_ssrd.split('/')[-1].split('.')[:-1]) + '_ano.' +file_ssrd.split('/')[-1].split('.')[-1]
    filename_ano_aet   =  '.'.join(file_aet.split('/')[-1].split('.')[:-1]) + '_ano.' +file_aet.split('/')[-1].split('.')[-1]


    # NDRE
    np.savetxt(fpath_out+'anomalies/' + filename_ano_ndre,
           data_NDRE_anomalies,
           fmt='%10.4f',
           delimiter = ",")
    # Pr
    np.savetxt(fpath_out+'anomalies/' + filename_ano_tp,
           data_tp_anomalies,
           fmt='%10.4f',
           delimiter = ",")
    # Tas
    np.savetxt(fpath_out+'anomalies/' + filename_ano_t2m,
           data_t2m_anomalies,
           fmt='%10.4f',  
           delimiter = ",")
    # swvl
    np.savetxt(fpath_out+'anomalies/' + filename_ano_swvl,
           data_swvl_anomalies,
           fmt='%10.4f',
           delimiter = ",")
    # ssrd
    np.savetxt(fpath_out +'anomalies/'+ filename_ano_ssrd,
           data_ssrd_anomalies,
           fmt='%10.4f',
           delimiter = ",")
    # aet
    np.savetxt(fpath_out +'anomalies/'+ filename_ano_aet,
           data_aet_anomalies,
           fmt='%10.4f',
           delimiter = ",")

if (rank == 0) and (attr_type == "ori") :
    # Saving anomalies
    # ================
    filename_ano_ndre =  '.'.join(file_NDRE.split('/')[-1].split('.')[:-1]) + '_ano.' +file_NDRE.split('/')[-1].split('.')[-1]
    # NDRE
    np.savetxt(fpath_out+'anomalies/' + filename_ano_ndre,
           data_NDRE_anomalies,
           fmt='%10.4f',
           delimiter = ",")


# Attribution
# ===========

codes_description = """NDRE Extreme type (neg/pos at 1000's) + Driver code (100's) + Driver anomaly (neg/pos at 10's) + lag (0to9 at 1's) ...
                    Driver code: tp (1), t2m(2), swvl (3), ssrd (4), and aet(5)
                    Driver anomaly type (neg/pos) i.e. 10/20
                    NDRE Extreme type: neg/pos i.e. 1000/2000

                    e.g. neg ndre extreme driven by neg temp anomaly at lag 1 : 1000+200+10+1 = 1211 code
                    e.g. pos ndre extreme driven by neg aet anomaly at lag 1  : 2000+500+10+1 = 2511 code
                    """




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



def Attribution_Drivers_Codes_1025 (ts_ndre_ano, ts_tp_ano, ts_t2m_ano, ts_swvl_ano, ts_ssrd_ano, ts_aet_ano, lag=0):
    """
    Similar to Attribution_Drivers_Codes, but here we compare the driver during 10p of NDRE with 25q of its own value and remove all grids with negative values
    
    Checks the climatic conditions during negative and postive NDRE extremes at a pixel.
    Assigns codes of attributions.
     
     9999     : "too few values",
     1110 + lag: "neg NDRE ext driven by neg tp anomaly ",
     1120 + lag: "neg NDRE ext driven by pos tp anomaly ",
     1210 + lag: "neg NDRE ext driven by neg t2m anomaly ",
     1220 + lag: "neg NDRE ext driven by pos t2m anomaly ",
     1310 + lag: "neg NDRE ext driven by neg swvl anomaly ",
     1320 + lag: "neg NDRE ext driven by pos swvl anomaly ",
     1410 + lag: "neg NDRE ext driven by neg ssrd anomaly ",
     1420 + lag: "neg NDRE ext driven by pos ssrd anomaly ",
     1510 + lag: "neg NDRE ext driven by neg aet anomaly ",
     1520 + lag: "neg NDRE ext driven by pos aet anomaly ",
     2110 + lag: "pos NDRE ext driven by neg tp anomaly ",
     2120 + lag: "pos NDRE ext driven by pos tp anomaly ",
     2210 + lag: "pos NDRE ext driven by neg t2m anomaly ",
     2220 + lag: "pos NDRE ext driven by pos t2m anomaly ",
     2310 + lag: "pos NDRE ext driven by neg swvl anomaly ",
     2320 + lag: "pos NDRE ext driven by pos swvl anomaly ",
     2410 + lag: "pos NDRE ext driven by neg ssrd anomaly ",
     2420 + lag: "pos NDRE ext driven by pos ssrd anomaly ",
     2510 + lag: "pos NDRE ext driven by neg aet anomaly ",
     2520 + lag: "pos NDRE ext driven by pos aet anomaly ",

     
    Parameters
    ----------
    ts_ndre_ano : ts of a pixel of NDRE
    ts_tp_ano : ts of a pixel of tp
    ts_t2m_ano : ts of a pixel of t2m
    ts_swvl_ano : ts of a pixel of swvl
    ts_ssrd_ano :ts of a pixel of ssrd
    ts_aet_ano : ts of a pixel of aet
    lag: lag timestep, default =0
    
    Returns
    -------
    The attributions codes array of a pixel
    """
    # To include the effect of lag
    if lag > 0 :
        ts_ndre_ano     = ts_ndre_ano[lag:]
        ts_tp_ano       = ts_tp_ano  [:-lag]
        ts_t2m_ano      = ts_t2m_ano [:-lag]
        ts_swvl_ano     = ts_swvl_ano  [:-lag]
        ts_ssrd_ano     = ts_ssrd_ano  [:-lag]
        ts_aet_ano      = ts_aet_ano  [:-lag]

    # initializing the codes array per pixel with zeros and will fill as conditions are met
    ts_codes_px = np.zeros((len(Codes)))

    ts_ndre_ano_non_mask_vals = ts_ndre_ano[~ts_ndre_ano.mask]
    if ts_ndre_ano_non_mask_vals.size < 75-int(lag) : 
        code_px = 9999
        ts_codes_px[0] = code_px
        # Printing zeros for location of extremes
        loc_10q = np.zeros((75-int(lag)))
        loc_90q = np.zeros((75-int(lag)))
        iav_px  = 0
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
        tp_du_neg   = ts_tp_ano  [loc_10q].mean()
        t2m_du_neg  = ts_t2m_ano [loc_10q].mean()
        swvl_du_neg = ts_swvl_ano[loc_10q].mean()
        ssrd_du_neg = ts_ssrd_ano[loc_10q].mean()
        aet_du_neg  = ts_aet_ano [loc_10q].mean()

        # Mean precipiration and temperature during > 90q NDRE
        tp_du_pos   = ts_tp_ano  [loc_90q].mean()
        t2m_du_pos  = ts_t2m_ano [loc_90q].mean()
        swvl_du_pos = ts_swvl_ano[loc_90q].mean()
        ssrd_du_pos = ts_ssrd_ano[loc_90q].mean()
        aet_du_pos  = ts_aet_ano [loc_90q].mean()

        # 25q of precipitation
        tp_25q = np.percentile(ts_tp_ano,25)
        # 75q of precipitation
        tp_75q = np.percentile(ts_tp_ano,75)

        # 25q of tas
        t2m_25q = np.percentile(t2m_du_neg,25)
        # 75q of tas
        t2m_75q = np.percentile(t2m_du_neg,75)

        # 25q of swvl
        swvl_25q = np.percentile(ts_swvl_ano,25)
        # 75q of swvl
        swvl_75q = np.percentile(ts_swvl_ano,75)

        # 25q of ssrd
        ssrd_25q = np.percentile(ts_ssrd_ano,25)
        # 75q of ssrd
        ssrd_75q = np.percentile(ts_ssrd_ano,75)

        # 25q of aet
        aet_25q = np.percentile(ts_aet_ano,25)
        # 75q of aet
        aet_75q = np.percentile(ts_aet_ano,75)

        # 1110 + lag: "neg NDRE ext driven by neg tp anomaly "
        if tp_du_neg < tp_25q :
            code_px = 1110+ lag
            ts_codes_px[1] = code_px
        
        # 1120 + lag: "neg NDRE ext driven by pos tp anomaly ",
        if tp_du_neg > tp_75q :
            code_px = 1120+ lag
            ts_codes_px[2] = code_px

        # 1210 + lag: "neg NDRE ext driven by neg t2m anomaly ",
        if t2m_du_neg < t2m_25q :
            code_px = 1210+ lag
            ts_codes_px[3] = code_px
        
        # 1220 + lag: "neg NDRE ext driven by pos t2m anomaly ",
        if t2m_du_neg > t2m_75q :
            code_px = 1220+ lag
            ts_codes_px[4] = code_px

        # 1310 + lag: "neg NDRE ext driven by neg swvl anomaly ",
        if swvl_du_neg < swvl_25q :
            code_px = 1310+ lag
            ts_codes_px[5] = code_px
        
        # 1320 + lag: "neg NDRE ext driven by pos swvl anomaly ",
        if swvl_du_neg > swvl_75q :
            code_px = 1320+ lag
            ts_codes_px[6] = code_px

        # 1410 + lag: "neg NDRE ext driven by neg ssrd anomaly ",
        if ssrd_du_neg < ssrd_25q :
            code_px = 1410+ lag
            ts_codes_px[7] = code_px
        
        # 1420 + lag: "neg NDRE ext driven by pos ssrd anomaly ",
        if ssrd_du_neg > ssrd_75q :
            code_px = 1420+ lag
            ts_codes_px[8] = code_px

        # 1510 + lag: "neg NDRE ext driven by neg aet anomaly ",
        if aet_du_neg < aet_25q :
            code_px = 1510+ lag
            ts_codes_px[9] = code_px
        
        # 1520 + lag: "neg NDRE ext driven by pos aet anomaly ",
        if aet_du_neg > aet_75q :
            code_px = 1520+ lag
            ts_codes_px[10] = code_px
        
        # 2110 + lag: "pos NDRE ext driven by neg tp anomaly "
        if tp_du_pos < tp_25q :
            code_px = 2110+ lag
            ts_codes_px[11] = code_px
        
        # 2120 + lag: "pos NDRE ext driven by pos tp anomaly ",
        if tp_du_pos > tp_75q :
            code_px = 2120+ lag
            ts_codes_px[12] = code_px
        
        # 2210 + lag: "pos NDRE ext driven by neg t2m anomaly ",
        if t2m_du_pos < t2m_25q :
            code_px = 2210+ lag
            ts_codes_px[13] = code_px
        
        # 2220 + lag: "pos NDRE ext driven by pos t2m anomaly ",
        if t2m_du_pos > t2m_75q :
            code_px = 2220+ lag
            ts_codes_px[14] = code_px

        # 2310 + lag: "pos NDRE ext driven by neg swvl anomaly ",
        if swvl_du_pos < swvl_25q :
            code_px = 2310+ lag
            ts_codes_px[15] = code_px
        
        # 2320 + lag: "pos NDRE ext driven by pos swvl anomaly ",
        if swvl_du_pos > swvl_75q :
            code_px = 2320+ lag
            ts_codes_px[16] = code_px

        # 2410 + lag: "pos NDRE ext driven by neg ssrd anomaly ",
        if ssrd_du_pos < ssrd_25q :
            code_px = 2410+ lag
            ts_codes_px[17] = code_px
        
        # 2420 + lag: "pos NDRE ext driven by pos ssrd anomaly ",
        if ssrd_du_pos > ssrd_75q :
            code_px = 2420+ lag
            ts_codes_px[18] = code_px

        # 2510 + lag: "pos NDRE ext driven by neg aet anomaly ",
        if aet_du_pos < aet_25q :
            code_px = 2510+ lag
            ts_codes_px[19] = code_px
        
        # 2520 + lag: "pos NDRE ext driven by pos aet anomaly ",
        if aet_du_pos > aet_75q :
            code_px = 2520+ lag
            ts_codes_px[20] = code_px

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
        ts_ndre_ano = data_NDRE_anomalies  [rank*int(data_NDRE_anomalies.shape[0]/load_divisor)+i,:]
        ts_tp_ano   = data_tp_anomalies    [rank*int(data_NDRE_anomalies.shape[0]/load_divisor)+i,:]
        t2m_du_neg  = data_t2m_anomalies   [rank*int(data_NDRE_anomalies.shape[0]/load_divisor)+i,:]
        ts_swvl_ano = data_swvl_anomalies  [rank*int(data_NDRE_anomalies.shape[0]/load_divisor)+i,:]
        ts_ssrd_ano = data_ssrd_anomalies  [rank*int(data_NDRE_anomalies.shape[0]/load_divisor)+i,:]
        ts_aet_ano  = data_aet_anomalies   [rank*int(data_NDRE_anomalies.shape[0]/load_divisor)+i,:]

        attr_ar[i],ar_loc_neg[i],ar_loc_pos[i], iav_ndre[i] = Attribution_Drivers_Codes_1025 (ts_ndre_ano, 
                                                                ts_tp_ano, 
                                                                t2m_du_neg, 
                                                                ts_swvl_ano,
                                                                ts_ssrd_ano,
                                                                ts_aet_ano,
                                                                lag=lag)

        # Saving attribution data
        # -----------------------
        #path_attr = ("/").join(file_NDRE.split('/')[:-1]) + '/' +  ("_").join((file_NDRE.split('/')[-1]).split("."))+'/'
        path_attr = fpath_out
        if os.path.isdir(path_attr) == False:
            os.makedirs(path_attr)
        if os.path.isdir(path_attr+'attribution/') == False:
            os.makedirs(path_attr+'attribution/')
        # Attr
        filename_attr = path_attr+'attribution/' + f"attr_{attr_type}_lag_{str(lag).zfill(2)}_rank_{str(rank).zfill(3)}.csv"
        np.savetxt(filename_attr,
               attr_ar,
               fmt='%i',
               delimiter = " ")
        #IAV
        filename_iav = path_attr+'attribution/' + f"iav_ndre_ano_{attr_type}_lag_{str(lag).zfill(2)}_rank_{str(rank).zfill(3)}.csv"
        np.savetxt(filename_iav,
               iav_ndre,
               fmt='%10.4f',
               delimiter = " ")

        if save_ext_loc in ['y','yes','YES','Y']:
            filename_neg_loc = path_attr+'attribution/' + f"neg_loc_{attr_type}_lag_{str(lag).zfill(2)}_rank_{str(rank).zfill(3)}.csv"
            np.savetxt(filename_neg_loc,
                   ar_loc_neg,
                   fmt='%i',
                   delimiter = " ")
            filename_pos_loc = path_attr +'attribution/'+ f"pos_loc_{attr_type}_lag_{str(lag).zfill(2)}_rank_{str(rank).zfill(3)}.csv"
            np.savetxt(filename_pos_loc,
                   ar_loc_pos,
                   fmt='%i',
                   delimiter = " ")

        
print ("Success : ", rank )
