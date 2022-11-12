#!/usr/bin/env python
"""
Created on Tue Aug  8 15:19:50 2017
@author: HGY
"""

import numpy as np
from scipy import stats

#%% functions
def checkValid(d, keep_dim=False):
    if not keep_dim:
        return ~((np.isnan(d).any()) or (np.isinf(d).any()))

def getConstDim(df):
    x = df.values
    constD = []
    for xIdx in range(x.shape[1]):
        if len(set(x[:,xIdx]))==1:
            constD.append(df.columns[xIdx])
    #return [t for t in range(x.shape[1]) if t not in constD]
    return constD

def getValidCol(df):
    d = df.values
    invalidCols = []
    tmp = np.argwhere(np.isinf(d))
    if tmp.size != 0:
        invalidCols.extend(list(tmp[:,1]))
    tmp = np.argwhere(np.isnan(d))
    if tmp.size != 0:
        invalidCols.extend(list(tmp[:,1]))
    invalidCols.extend(getConstDim(df))
    invalidCols = list(set(invalidCols))
    validCols = [df.columns[idx] for idx in range(d.shape[1])  if idx not in invalidCols]
    return validCols



#%%
def getFunctional15(data):    
    """Functional list: ['max','min','mean','median','standard_deviation','1_percentile','99_percentile',
                         '99minus1_percentile','skewneww','kurtosis','min_pos','max_pos','low_quar',
                         'up_quar','quartile_range'] """
    Functional = []
    #0 max
    Functional.append(np.max(data, axis = 0))
    #1 min    
    Functional.append(np.min(data, axis = 0))
    #2 mean
    Functional.append(np.mean(data, axis = 0))
    #3 median    
    Functional.append(np.median(data, axis = 0))
    #4 standard deviation
    Functional.append(np.std(data, axis = 0) )         
    #5 1st_percentile   
    Functional.append(np.percentile(data, 1, axis = 0))
    #6 99th percentile
    Functional.append(np.percentile(data, 99, axis = 0))
    #7 99th percentile - 1st percentile
    Functional.append(Functional[-1]-Functional[-2])
    #8 skewness
    Functional.append(stats.skew(data, axis=0))
    #9 kurtosis
    Functional.append(stats.kurtosis(data, axis=0))
    #10 minmum position
    Functional.append((np.argmin(data, axis=0)).astype(float)/len(data))
    #11 maximum position
    Functional.append((np.argmax(data, axis=0)).astype(float)/len(data))
    #12 lower quartile
    Functional.append(np.percentile(data, 25, axis = 0))
    #13 upper quartile
    Functional.append(np.percentile(data, 75, axis = 0))
    #14 interqyartile range
    Functional.append(Functional[-1]-Functional[-2])
    return np.asarray(Functional).reshape(-1)
    #return np.vstack(Functional).reshape(1, -1)
    
def getFunctional15Dict(data,name):    
    Functional = {}
    Functional[name+'_max'] = np.max(data, axis = 0)
    Functional[name+'_min'] = (np.min(data, axis = 0))
    Functional[name+'_mean'] = (np.mean(data, axis = 0))
    Functional[name+'_median'] = (np.median(data, axis = 0))
    Functional[name+'_std'] = (np.std(data, axis = 0) )         
    Functional[name+'_1_per'] = (np.percentile(data, 1, axis = 0))
    Functional[name+'_99_per'] = (np.percentile(data, 99, axis = 0))
    Functional[name+'_99minus1_per'] = (Functional[name+'_99_per']-Functional[name+'_1_per'])
    Functional[name+'_skewneww'] = (stats.skew(data, axis=0))
    Functional[name+'_kurtosis'] = (stats.kurtosis(data, axis=0))
    Functional[name+'_min_pos'] = ((np.argmin(data, axis=0)).astype(float)/len(data))
    Functional[name+'_max_pos'] = ((np.argmax(data, axis=0)).astype(float)/len(data))
    Functional[name+'_low_quar'] = (np.percentile(data, 25, axis = 0))
    Functional[name+'_up_quar'] = (np.percentile(data, 75, axis = 0))
    Functional[name+'_quartile_range'] = (Functional[name+'_up_quar']-Functional[name+'_low_quar'])
    return Functional



def getFunctional10Dict(data,name):    
    Functional = {}
    Functional[name+'_max'] = np.max(data, axis = 0)
    Functional[name+'_min'] = (np.min(data, axis = 0))
    Functional[name+'_mean'] = (np.mean(data, axis = 0))
    Functional[name+'_median'] = (np.median(data, axis = 0))
    Functional[name+'_std'] = (np.std(data, axis = 0) )         
    Functional[name+'_skewneww'] = (stats.skew(data, axis=0))
    Functional[name+'_kurtosis'] = (stats.kurtosis(data, axis=0))
    Functional[name+'_low_quar'] = (np.percentile(data, 25, axis = 0))
    Functional[name+'_up_quar'] = (np.percentile(data, 75, axis = 0))
    Functional[name+'_quartile_range'] = (Functional[name+'_up_quar']-Functional[name+'_low_quar'])
    return Functional




def cm2UAR(cm):
    return np.mean([cm[rowIdx,rowIdx]/sum(cm[rowIdx,:]) for rowIdx in range(len(cm))])
def cm2Recall(cm, precision=3):
    return [cm[rowIdx,rowIdx]/sum(cm[rowIdx,:]) for rowIdx in range(len(cm))]