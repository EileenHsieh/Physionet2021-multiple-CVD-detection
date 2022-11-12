#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  5 15:27:53 2021

@author: chadyang

input:
    site: str
    N_fold: int
    ORDER: int
    
step:
    load labels from raw data
    load target labels from weights.csv
    
output:
    k_fold data list
"""
#%%
from pathlib import Path
import sys, os, json

PATHS = json.load(open(Path('.').absolute()/"paths.json"))
for k in PATHS:
    sys.path.append(PATHS[k])
    
from sklearn.model_selection import KFold
from skmultilearn.model_selection import IterativeStratification
from lib.evaluation_2021.evaluate_model import *
from src.engine.utils import replace_equivalent_classes
from src.engine.utils import NORMAL_CLASS, CLASSES, WEIGHTS, SCORED_CLASSES, EQ_CLASSES

from helper_code import find_challenge_files, get_labels, load_recording
from tqdm import tqdm
import pandas as pd
import numpy as np

#%%
RAW_DATA_ROOT = "./datasets/raw"
CV_LIST_SAVE_ROOT = Path("./datasets/CVFolds")
site = "WFDB_Ningbo" # [WFDB_CPSC2018, WFDB_CPSC2018_2, WFDB_StPetersburg, WFDB_PTB, WFDB_PTBXL, WFDB_Ga, WFDB_ChapmanShaoxing, WFDB_Ningbo]

#%%



#%% params
RANDOM_STATE = 100
N_fold = 5
ORDER = 3
cv_save_dir = CV_LIST_SAVE_ROOT/f"cv-{N_fold}_order-{ORDER}"/site
os.makedirs(cv_save_dir, exist_ok=True)

#%%
data_directory = Path(RAW_DATA_ROOT)/site
header_files, recording_files = find_challenge_files(data_directory)
num_recordings = len(recording_files)

#%%
## load all samples label
all_labels_scroed, all_labels_non_scroed = [], []
for dataIdx in tqdm(range(num_recordings)):
    # dataIdx = 21
    header = load_header(header_files[dataIdx])
    recording = load_recording(recording_files[dataIdx])
    sampleId = header_files[dataIdx].split('/')[-1].split('.')[0]
    labels = replace_equivalent_classes(get_labels(header), EQ_CLASSES)
    if any(label in SCORED_CLASSES for label in labels):
        all_labels_scroed.append([sampleId, labels])
    else:
        all_labels_non_scroed.append([sampleId, labels])

#%% scored labels into df
df_all_labels_scroed = []
for sampleId,labels in all_labels_scroed:
    label_bin_tmp = pd.DataFrame(np.zeros((1,len(SCORED_CLASSES))), columns=SCORED_CLASSES, index=[sampleId])
    labels = [l for l in labels if l in SCORED_CLASSES ]
    label_bin_tmp[labels] = 1
    df_all_labels_scroed.append(label_bin_tmp)
df_all_labels_scroed = pd.concat(df_all_labels_scroed).astype("int")


#%% stratify on scored labels
k_fold = IterativeStratification(n_splits=N_fold, order=ORDER)
for foldIdx, (trainIdxs, testIdxs) in enumerate(k_fold.split(df_all_labels_scroed, df_all_labels_scroed)):
    fold_sample_Ids = [df_all_labels_scroed.index[idx] for idx in testIdxs]
    cv_save_path = cv_save_dir/f"fold_{foldIdx}_scored.json"
    with open(cv_save_path, "w") as f:
        json.dump(fold_sample_Ids, f, sort_keys=True, indent=4)

#%% random sample on unscroed labels
k_fold = KFold(n_splits=N_fold, shuffle=True, random_state=RANDOM_STATE)
for foldIdx, (trainIdxs, testIdxs) in enumerate(k_fold.split(all_labels_non_scroed)):
    fold_sample_Ids = [all_labels_non_scroed[idx][0] for idx in testIdxs]
    cv_save_path = cv_save_dir/f"fold_{foldIdx}_non_scored.json"
    with open(cv_save_path, "w") as f:
        json.dump(fold_sample_Ids, f, sort_keys=True, indent=4)


#%% Reload the generated jsons to check label balance in each folds
"""
for foldIdx in range(N_fold):
    cv_save_path = cv_save_dir/f"fold_{foldIdx}_scored.json"
    with open(cv_save_path, "r") as f:
        sampleIds = json.load(f)
    
    header_files = [data_directory/(sampleId+".hea") for sampleId in sampleIds]

    all_labels_scroed = []
    for header_file in header_files:
        header = load_header(header_file)
        labels = replace_equivalent_classes(get_labels(header), EQ_CLASSES)
        if any(label in SCORED_CLASSES for label in labels):
            all_labels_scroed.append([sampleId, labels])
    
    df_all_labels_scroed = []
    for sampleId,labels in all_labels_scroed:
        label_bin_tmp = pd.DataFrame(np.zeros((1,len(SCORED_CLASSES))), columns=SCORED_CLASSES, index=[sampleId])
        labels = [l for l in labels if l in SCORED_CLASSES ]
        label_bin_tmp[labels] = 1
        df_all_labels_scroed.append(label_bin_tmp)
    df_all_labels_scroed = pd.concat(df_all_labels_scroed).astype("int")

    print(df_all_labels_scroed.mean())
"""
