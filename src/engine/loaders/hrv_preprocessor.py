#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 14 10:55:19 2021

@author: chadyang
"""


import coloredlogs, logging
coloredlogs.install()
logger = logging.getLogger(__name__)  

from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from collections import defaultdict

import pandas as pd
import numpy as np
from copy import deepcopy
pd.set_option('mode.chained_assignment', None)

#%%
class Preprocessor:
    DEFAULTS = {}   
    def __init__(self, config, fea_lead_all):
        self.param_loader = deepcopy(config.param_loader)
        self.param_preprocess = deepcopy(config.param_preprocess)
        self.leads = fea_lead_all.keys()
        self.all_lead_preprocessors = {lead:defaultdict() for lead in self.leads}
        
        logger.info('-'*60)
        logger.info("Initialize Preprocessor ...")
        
        if config.param_feature.hrv_nk2:
            # if self.param_preprocess["impute"] in ["median", "mean", "most_frequent"]: fea_lead_all = self._fit_transform_imputer(fea_lead_all)
            if self.param_preprocess.impute in ["median", "mean", "most_frequent"]: self._fit_transform_imputer(fea_lead_all)
            if self.param_preprocess.scaler == "zscore": self._fit_transform_scaler(fea_lead_all)


    #%% imputation normal features -> fea_imputed (n*d array)
    def _fit_transform_imputer(self, fea_lead_all):
        logger.info("[Imputation]: "+self.param_preprocess["impute"])

        for lead in self.leads:
            fea_lead = fea_lead_all[lead]
            
            # continuous columns (feature without gender)
            col_conti = [c for c in fea_lead.columns if c!="sex"]
            fea_lead_conti = fea_lead[col_conti]
            fea_lead_conti.replace([np.inf, -np.inf], np.nan, inplace=True)
            imp_conti = SimpleImputer(missing_values=np.nan, strategy=self.param_preprocess.impute).fit(fea_lead_conti)
            fea_lead[col_conti] = imp_conti.transform(fea_lead_conti)
            self.all_lead_preprocessors[lead]["imp_conti"] = [col_conti, imp_conti]
            
            # categorical columns
            col_cat = [c for c in fea_lead.columns if c=="sex"]
            fea_lead_cat = fea_lead[col_cat]
            fea_lead_cat.replace([np.inf, -np.inf], np.nan, inplace=True)
            imp_cat = SimpleImputer(missing_values=np.nan, strategy='most_frequent').fit(fea_lead_cat)
            fea_lead[col_cat] = imp_cat.transform(fea_lead_cat)
            self.all_lead_preprocessors[lead]["imp_cat"] = [col_cat, imp_cat]
        return fea_lead_all


    #%% normalization           
    def _fit_transform_scaler(self, fea_lead_all):
        logger.info("[Scaler]: "+self.param_preprocess.scaler)
        for lead in self.leads:
            fea_lead = fea_lead_all[lead]

            # continuous columns (feature without gender)
            col_conti = [c for c in fea_lead.columns if c!="sex"]
            fea_lead_conti = fea_lead[col_conti]
            scaler = StandardScaler().fit(fea_lead_conti)
            fea_lead[col_conti] = scaler.transform(fea_lead_conti)
            self.all_lead_preprocessors[lead]["scalar_conti"] = [col_conti, scaler]
            


    #%%
    def transform(self, fea_lead_all):
        fea_lead_all = fea_lead_all.copy()
        if self.param_loader.fea=="hrv":

            # imputer
            if self.param_preprocess.impute in ["median", "mean", "most_frequent"]:
                for lead in self.leads:
                    fea_lead = fea_lead_all[lead]

                    col_conti, imp_conti = self.all_lead_preprocessors[lead]["imp_conti"]
                    fea_lead_conti = fea_lead[col_conti]
                    fea_lead_conti.replace([np.inf, -np.inf], np.nan, inplace=True)
                    fea_lead[col_conti] = imp_conti.transform(fea_lead_conti)
                    
                    col_cat, imp_cat = self.all_lead_preprocessors[lead]["imp_cat"]
                    fea_lead_cat = fea_lead[col_cat]
                    fea_lead_cat.replace([np.inf, -np.inf], np.nan, inplace=True)
                    fea_lead[col_cat] = imp_cat.transform(fea_lead_cat)
                    fea_lead_all[lead] = fea_lead
                
            # scaler
            if self.param_preprocess.scaler== "zscore":
                for lead in self.leads:
                    fea_lead = fea_lead_all[lead]
                    col_conti, scaler = self.all_lead_preprocessors[lead]["scalar_conti"]
                    fea_lead_conti = fea_lead[col_conti]
                    fea_lead[col_conti] = scaler.transform(fea_lead_conti)
                    fea_lead_all[lead] = fea_lead
        return fea_lead_all


#%% for ease of debug
# if __name__ == "__main__":
#     preprocessor = Preprocessor(config, fea_lead_all_1)
#     self=preprocessor
#     preprocessor.transform(fea_lead_all_2)
