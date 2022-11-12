# -*- coding: utf-8 -*-

import joblib
from pathlib import Path
import pandas as pd
import coloredlogs, logging
from src.engine.loaders.hrv_loader import load_hrv_pkls
from lib.evaluation_2021.evaluate_model import *
from src.engine.utils import NORMAL_CLASS, CLASSES, WEIGHTS, SCORED_CLASSES

coloredlogs.install()
logger = logging.getLogger(__name__)

EQ_CLASSES = [[['733534002','164909002'],'733534002|164909002'], [['713427006','59118001'],'713427006|59118001'], [['284470004', '63593006'], '284470004|63593006'], [['427172004', '17338001'],'427172004|17338001']]
ALL_SITES = ["WFDB_CPSC2018", "WFDB_CPSC2018_2", "WFDB_Ga", "WFDB_PTB", "WFDB_PTBXL", "WFDB_StPetersburg"]
twelve_leads = ('I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6')


def general_loader(config):
    # hrv features with concatenated features
    if config.param_feature.hrv_nk2 and config.param_feature.hrv_nk2.concat==True:
        
        sites = ALL_SITES if config.exp.sites=="all" else config.exp.sites
        fea_lead_all_site, label_bin_all_site = {lead:[] for lead in twelve_leads}, []
        for site in sites:
            # load exist "hrv_all.pkl"
            pklDir = Path("./datasets/processed/nk2/hrv")/site
            data_path = pklDir/"hrv_all.pkl"
            site_pkl_paths = list(pklDir.rglob("*.pkl"))
            if data_path.exists():
                fea_lead_all, label_all_tmp = joblib.load(data_path.absolute())

            # transform site's "*.pkl" into single "hrv_all.pkl"
            elif len(site_pkl_paths)>0:
                logger.warning(f"Aggregate {site} data {len(site_pkl_paths)} pkls into single hrv_all.pkl")
                fea_lead_all, label_all_tmp = load_hrv_pkls(config, site_pkl_paths, twelve_leads)
                joblib.dump([fea_lead_all, label_all_tmp], data_path)
            
            # feature extract into "hrv_all.pkl""
            else:
                logger.warning(f"Didn't Find Any *.pkls in {pklDir.absolute()}. Start Extract Nk2 Features ...")

            # parse label
            if config.param_loader.replace_eq: 
                label_all_tmp = {k:replace_equivalent_classes(l, EQ_CLASSES) for k,l in label_all_tmp.items()}
                label_all_tmp = {k:[l for l in labels if l in SCORED_CLASSES] for k,labels in label_all_tmp.items()}
                label_bin = []
                for sampleId,labels in label_all_tmp.items():
                    label_bin_tmp = pd.DataFrame(np.zeros((1,len(SCORED_CLASSES))), columns=SCORED_CLASSES, index=[sampleId])
                    label_bin_tmp[labels] = 1
                    label_bin.append(label_bin_tmp)
                label_bin = pd.concat(label_bin).astype("int")
                
            # aggregate all sites
            for k,v in fea_lead_all.items():
                fea_lead_all_site[k].append(v)
            label_bin_all_site.append(label_bin)

        fea_lead_all_site = {k:pd.concat(v) for k,v in fea_lead_all_site.items()}
        label_bin_all_site = pd.concat(label_bin_all_site)
        sampleId = fea_lead_all_site[list(fea_lead_all_site.keys())[0]].index
        label_bin_all_site = label_bin_all_site.loc[sampleId]
        logger.info(f"Now Process Sites {config.exp.sites}. Total Number of Smpales :{len(label_bin_all_site)}")
        return fea_lead_all_site, label_bin_all_site
    
    
#%%
# For each set of equivalent classes, replace each class with the representative class for the set.
def replace_equivalent_classes(classes, equivalent_classes):
    for j, x in enumerate(classes):
        for multiple_classes in equivalent_classes:
            if x in multiple_classes[0]:
                classes[j] = multiple_classes[1] # Use the first class as the representative class.
    return classes