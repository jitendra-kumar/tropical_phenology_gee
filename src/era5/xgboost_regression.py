import xarray as xr
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

#from pandas.plotting import lag_plot
#from statsmodels.tsa.vector_ar.var_model import VAR
#from statsmodels.tsa.stattools import adfuller, kpss, grangercausalitytests
import warnings
warnings.filterwarnings("ignore")

from pickle import dump, load
import os
import sys

import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as MSE
from sklearn.metrics import r2_score as r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
import shap

from csv import writer
############################################################################
############################################################################

#diff=True
diff=False
#lag=False
lag=True  # using lag data gives better r2

clust=int(sys.argv[1])

# xgboost model
def fit_xgboost(train_X, train_Y, test_X, test_Y): #, filename, logfile):

#    print(train_X)
#    print(train_Y)
    params = {'max_depth': [3, 6, 10, 15],
              'learning_rate': [0.01, 0.1, 0.2, 0.3, 0.4],
              'subsample': np.arange(0.5, 1.0, 0.1),
              'colsample_bytree': np.arange(0.5, 1.0, 0.1),
              'colsample_bylevel': np.arange(0.5, 1.0, 0.1),
              'n_estimators': [100, 250, 500, 750],
              'num_class': [10]
    }

    xgb_r = xgb.XGBRegressor()
    parameters = {'nthread':[2], #when use hyperthread, xgboost may become slower
#             'objective':['reg:squarederror'],
              'learning_rate': [.03, 0.05, .07, 0.1, 0.2, 0.3, 0.4, 0.5], #so called `eta` value
              'max_depth': [5, 6, 7, 10, 20, 50],
#              'min_child_weight': [4],
#              'silent': [1],
              'subsample': [0.7],
#              'colsample_bytree': [0.7],
              'n_estimators': [50, 100, 500]}

    parameters1 = {'nthread':[4], #when use hyperthread, xgboost may become slower
              'objective':['reg:squarederror'],
              'learning_rate': [.03, 0.05, .07, 0.1, 0.2], #so called `eta` value
              'max_depth': [5, 6, 7],
              'min_child_weight': [4],
              'subsample': np.arange(0.5, 1.0, 0.1),
              'colsample_bytree': np.arange(0.5, 1.0, 0.1),
              'colsample_bylevel': np.arange(0.5, 1.0, 0.1),
              'n_estimators': [10, 50, 100, 200, 500]}

    xgb_grid = GridSearchCV(xgb_r,
                        parameters,
                        cv = 5,
                        n_jobs = 5,
                        verbose=True)

    xgb_grid.fit(train_X, train_Y)

    print("Best score")
    print(xgb_grid.best_score_)
    print("Best params")
    print(xgb_grid.best_params_)
#    logfile.write("Best score")
#    logfile.write(str(xgb_grid.best_score_))
#    logfile.write("Best params")
#    logfile.write(str(xgb_grid.best_params_))

    # Predict the model
    pred = xgb_grid.predict(test_X)

    # RMSE Computation
    rmse = np.sqrt(MSE(test_Y, pred))
    r2 = r2_score(test_Y, pred)
    print("XGBOOST: RMSE : % f R2: %f" %(rmse, r2))
#    logfile.write("XGBOOST: RMSE : % f R2: %f" %(rmse, r2))

#    # save model to a pickle file, along with train and test data
#    dump([xgb_grid, train_X, test_X, train_Y, test_Y], open(filename, 'wb'))

    return xgb_grid, r2, rmse

############################################################################

# read ERA5 data for Costa Rica + Panama for 2017-2022
era5 = xr.open_mfdataset('/home/jbk/projects/climate/tropics/costa_rica_panama/era5/15d_20*_daily_costarica_panama_EPSG4326_allvars.nc')
# list of era5 variables
evars=["t2m", "ssrd", "sshf", "slhf", "tp", "ro", "swvl1", "swvl2", "swvl3", "swvl4", "e", "evavt", "pev", "d2m", "vpd", "ndre"]

# read NDRE time seris for 2017-2022 for Costa Rica
# 133858253
chunksize=1000
crdf = pd.read_csv('/home/jbk/projects/climate/tropics/costa_rica_panama/era5/subset/cr_pn_h10_%d.report'%(clust), delimiter="|", chunksize=chunksize)

fsv=open("shap_values_h10_%d.csv"%(clust), "a")
wfsv=writer(fsv)
fso=open("shap_order_h10_%d.csv"%(clust), "a")
wfso=writer(fso)

chunks=0
for data in crdf:
    chunks=chunks+1
    for p in range(len(data)):
#    for p in range(25,26):
        print("Chunk %d %d of %d"%(chunks, p+1, len(data)))
#               print(data.iloc[p,2:152])
        subset = era5.sel(latitude=data['y'].iloc[p], longitude=data['x'].iloc[p], method='nearest')
#        subset['ndre'] = xr.DataArray(data=data.iloc[p,2:2+150], coords=[era5.time], dims='time') #.values
        # let's only use last two years 2021-2022
        subset['ndre'] = xr.DataArray(data=data.filter(regex=("n20*")).iloc[p,:], coords=[era5.time], dims='time') #.values
#        print("NDRE shape")
#        print(subset['ndre'].shape)
        df_subset = subset[evars].to_dataframe()
#        print(df_subset)
        df_subset = df_subset.dropna()
#        print(df_subset)
#               print(subset)
#               print(df_subset)

        # prepare training vectors
        XY_all = df_subset[["t2m", "tp", "swvl1", "swvl2", "swvl3", "swvl4", "vpd", "ndre"]]
        XY_all['ndre_1'] = XY_all['ndre'].shift(1)
        XY_all['ndre_2'] = XY_all['ndre'].shift(2)
        XY_all['ndre_3'] = XY_all['ndre'].shift(3)
#               print(XY_all.diff(periods=1)['t2m'])
        if lag:
            print("Using lag values")
            XY_all['t2m_1'] = XY_all['t2m'].shift(1)
            XY_all['t2m_2'] = XY_all['t2m'].shift(2)
            XY_all['t2m_3'] = XY_all['t2m'].shift(3)

            XY_all['tp_1'] = XY_all['tp'].shift(1)
            XY_all['tp_2'] = XY_all['tp'].shift(2)
            XY_all['tp_3'] = XY_all['tp'].shift(3)

            XY_all['swvl1_1'] = XY_all['swvl1'].shift(1)
            XY_all['swvl1_2'] = XY_all['swvl1'].shift(2)
            XY_all['swvl1_3'] = XY_all['swvl1'].shift(3)

            XY_all['swvl2_1'] = XY_all['swvl2'].shift(1)
            XY_all['swvl2_2'] = XY_all['swvl2'].shift(2)
            XY_all['swvl2_3'] = XY_all['swvl2'].shift(3)

            XY_all['swvl3_1'] = XY_all['swvl3'].shift(1)
            XY_all['swvl3_2'] = XY_all['swvl3'].shift(2)
            XY_all['swvl3_3'] = XY_all['swvl3'].shift(3)

            XY_all['swvl4_1'] = XY_all['swvl4'].shift(1)
            XY_all['swvl4_2'] = XY_all['swvl4'].shift(2)
            XY_all['swvl4_3'] = XY_all['swvl4'].shift(3)

            XY_all['vpd_1'] = XY_all['vpd'].shift(1)
            XY_all['vpd_2'] = XY_all['vpd'].shift(2)
            XY_all['vpd_3'] = XY_all['vpd'].shift(3)

        if diff:
            print("Usig diff values")
            XY_all['t2m_1'] = XY_all.diff(periods=1)['t2m']
            XY_all['t2m_2'] = XY_all.diff(periods=1)['t2m_1']
            XY_all['t2m_3'] = XY_all.diff(periods=1)['t2m_2']

            XY_all['tp_1'] = XY_all.diff(periods=1)['tp']
            XY_all['tp_2'] = XY_all.diff(periods=1)['tp_1']
            XY_all['tp_3'] = XY_all.diff(periods=1)['tp_2']

            XY_all['swvl1_1'] = XY_all.diff(periods=1)['swvl1']
            XY_all['swvl1_2'] = XY_all.diff(periods=1)['swvl1_1']
            XY_all['swvl1_3'] = XY_all.diff(periods=1)['swvl1_2']

            XY_all['swvl2_1'] = XY_all.diff(periods=1)['swvl2']
            XY_all['swvl2_2'] = XY_all.diff(periods=1)['swvl2_1']
            XY_all['swvl2_3'] = XY_all.diff(periods=1)['swvl2_2']

            XY_all['swvl3_1'] = XY_all.diff(periods=1)['swvl3']
            XY_all['swvl3_2'] = XY_all.diff(periods=1)['swvl3_1']
            XY_all['swvl3_3'] = XY_all.diff(periods=1)['swvl3_2']

            XY_all['swvl4_1'] = XY_all.diff(periods=1)['swvl4']
            XY_all['swvl4_2'] = XY_all.diff(periods=1)['swvl4_1']
            XY_all['swvl4_3'] = XY_all.diff(periods=1)['swvl4_2']

            XY_all['vpd_1'] = XY_all.diff(periods=1)['vpd']
            XY_all['vpd_2'] = XY_all.diff(periods=1)['vpd_1']
            XY_all['vpd_3'] = XY_all.diff(periods=1)['vpd_2']
#               XY_all = XY_all.dropna()
#               print(XY_all)
        print(XY_all.shape)
        # =======================================================================
        # Split data in train and test sets
        # =======================================================================
        predictors = XY_all.loc[:, XY_all.columns != "ndre"]
        target = XY_all["ndre"]
#        print(predictors.columns)
#        print("Len predictors %d"%(len(predictors)))
#        print("Len target %d"%(len(target)))
        if len(target) >0:
            train_X, test_X, train_Y, test_Y = train_test_split(predictors, target,
                  test_size = 0.3, random_state = 123)
            # =======================================================================
            # Fit XGBOOST model
            print("%d Fit XGBOOST"%(p))
            xgb_model, r2, rmse = fit_xgboost(train_X, train_Y, test_X, test_Y)
            # =======================================================================

            # =============================================================================
            # Shapley Values for Feature Importance
            # =============================================================================
            # Fit relevant explainer
            explainer = shap.TreeExplainer(xgb_model.best_estimator_)
            shap_values = explainer.shap_values(train_X)
#            print(shap_values)
#            print(shap_values.shape)
            # save shap values to file
            wfsv.writerow(np.append(np.absolute(shap_values).mean(axis=0), [r2, p]))
            fsv.flush()
            # save relative importance order to file
            wfso.writerow(np.append(np.absolute(shap_values).mean(axis=0).argsort(),[p]))
            fso.flush()
        else:
            print("%d NANs encountered.. skipping xgboost modeling"%(p))
            shap_values = np.ones((1, 31))*9999
            # save shap values to file
            wfsv.writerow(np.append(shap_values.mean(axis=0), [r2, p]))
            fsv.flush()
            # save relative importance order to file
            wfso.writerow(np.append(shap_values.mean(axis=0),[p]))
            fso.flush()
            print(shap_values)
        # View shap values
#        print(shap_values)
#        print(shap_values.sum())
#        print(shap_values.min())
#        print(shap_values.max())

        # =============================================================================
        # # Get global variable importance plot
        # =============================================================================
#        plt_shap = shap.summary_plot(shap_values, #Use Shap values array
#                     features=train_X, # Use training set features
#                     feature_names=train_X.columns, #Use column names
#                     show=False, #Set to false to output to folder
#                     plot_size=(30,15)) # Change plot size

#        # Save my figure to a directory
#        plt.savefig("global_shap.png")

#        break
    break
print("Total chunks: %d"%(chunks))
fsv.close()
fso.close()
