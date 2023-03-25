import pandas as pd
import numpy as np
import random
import os 
import sys
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from sklearn import manifold
import copy


def read_pickle(filepath):
    with open(filepath, 'rb') as handle:
        data_dict = pickle.load(handle)
    return data_dict  

def write_pickle(path, pickle_dictionary):
    with open(path, 'wb') as handle:
        pickle.dump(pickle_dictionary, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
def normalize_df(df, col_list):
    for col in col_list:
        df[col] = df[col]/df[col].max()
    return df

def standardize_df(df, col_list):
    DF = copy.copy(df)
	
    if type(col_list) == str and col_list.upper() == 'ALL':
        col_list = list(DF.columns)
    
    for j in col_list:
        col = DF[j]
        mean = col.mean()
        std = col.std()
        DF[j] = (np.array(col) - np.array(mean))/np.array(std)
        
    return DF

def remove_outliers(df, col_list, num_of_stds):
    for col in col_list:
        df = df[(df[col] > -num_of_stds) & (df[col] < num_of_stds)]
    return df
    
    
def remove_n_vals_from_df(df, n):
    drop_indices = np.random.choice(df.index, n, replace=False)
    df = df.drop(drop_indices)
    return df


def create_cluster_cv_indices(train, train_labels, cluster_col, num_clusters):
    
    group_kfold = GroupKFold(n_splits=num_clusters) 
    clusters = train[cluster_col]
    cluster_kfold = group_kfold.split(train, train_labels, clusters)  
    train_indices, val_indices = [list(trainval) for trainval in zip(*cluster_kfold)]
    cluster_cv = [*zip(train_indices, val_indices)]
    
    return cluster_cv




