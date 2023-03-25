#!/usr/bin/python3

import pandas as pd
import numpy as np
import os
from numba import jit
import pickle
import matplotlib.image as mpimg

def describe_df(df):
    print('******************************')
    print('NUMBER OF VALUES THAT ARE NA:')
    print('******************************')

    print(df.isnull().sum())
    print('******************************')
    print('SHAPE AND SIZE OF DATAFRAME:')
    print('******************************')

    print(df.shape, df.size, 'BYTES')
    print('******************************')
    print('DATATYPES OF DATAFRAME:')
    print('******************************')

    print(df.dtypes)
    print('******************************')   
    print('CONTAINS', len(df) - len(df.drop_duplicates()), 'DUPLICATES')
    print('******************************') 

	
def write_pickle(path, pickle_dictionary):
    with open(path, 'wb') as handle:
        pickle.dump(pickle_dictionary, handle, protocol=pickle.HIGHEST_PROTOCOL)

        
def read_pickle(filepath):
    with open(filepath, 'rb') as handle:
        data_dict = pickle.load(handle)
    return data_dict        

def make_lists_equal_length(list1, list2):
    
    if len(list1) >= len(list2):
        newlist2 = list2*(math.ceil(len(list1)/len(list2)))
        lists_zipped = zip(list1, newlist2)
        
    if len(list2) > len(list1):
        newlist1 = list1*(math.ceil(len(list2)/len(list1)))
        lists_zipped = zip(newlist1, list2)
    
    return lists_zipped


def display_png(filename):
    img = mpimg.imread(filename)
    plt.imshow(img)
    plt.show()


@jit(nopython=True)
def combine_arrays(arr1, arr2):
    return np.hstack((arr1, arr2))    
    
  

