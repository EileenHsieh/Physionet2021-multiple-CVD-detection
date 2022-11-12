#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 26 21:51:03 2021
@author: chadyang
"""
import sys, os
from helper_code import find_challenge_files, load_header, load_recording

from lib.evaluation_2021.evaluate_model import *
from lib.signal_process.neurokit2 import extract_neurokit2_data
from tqdm import tqdm
import joblib
import json

import multiprocessing
from multiprocessing import Pool
from utils import istarmap
from functools import partial

import time
from pathlib import Path
import coloredlogs, logging
coloredlogs.install()
logger = logging.getLogger(__name__)  


#%%
OUT_HRV_ROOT = Path("./datasets/processed/nk2/hrv").absolute()
OUT_TIME_ROOT = Path("./datasets/processed/nk2/time").absolute()
RAW_DATA_ROOT = Path("./datasets/raw").absolute()
FEA_NAMES = json.load(open(Path(__file__).parent.absolute()/"./FEA_NAMES_HRV.json"))




#%%
def extract_data(dataIdx, config=None, header_files=None, recording_files=None, leads=None):
    header = load_header(header_files[dataIdx])
    recording = load_recording(recording_files[dataIdx])
    sampleId = header_files[dataIdx].split('/')[-1].split('.')[0]
    site = header_files[dataIdx].split('/')[-2]
    
    # extract feature and label
    data_hrv, data_time, failed_list = extract_neurokit2_data(header, recording, sampleId, leads, config.param_feature.hrv_nk2.fea_types)
    data_hrv.update({'sampleId':sampleId, 'site':site})
    data_time.update({'sampleId':sampleId, 'site':site})
    
    # parse multilead featrue
    if config.param_feature.hrv_nk2:
        fea_lead_all = {}
        for lead in leads:
            fea_multi_lead = data_hrv['fea_ECG_multi_lead'][f'{sampleId}-{lead}']
            if fea_multi_lead is not None:
                fea_lead = [f for f_name, f in fea_multi_lead.items() if f_name in fea_types]
                fea_lead = pd.concat(fea_lead,  axis=1)
                fea_lead.columns = [f'{lead}-{f_col_name}' for f_col_name in fea_lead.columns]
                fea_lead.index = [sampleId]
            
            else: # insert nan  to empty lead features
                fea_lead = []
                for fea_type in fea_types:
                    if fea_type != 'fea_demo':
                        fake_col_name = [f'{lead}-{f_col_name}' for f_col_name in FEA_NAMES[fea_type]]
                        f = pd.DataFrame(np.asarray([np.nan]*GET_FEA_DIMS[fea_type]).reshape(1,-1), index=[sampleId], columns=fake_col_name)
                        fea_lead.append(f)
                fea_lead = pd.concat(fea_lead, axis=1)
    
            if 'fea_demo' in fea_types:
                fea_demo = data_hrv['fea_demo']
                fea_demo.index =[sampleId]
                fea_lead = pd.concat([fea_demo, fea_lead], axis=1)
            fea_lead_all.update({lead:fea_lead})

        # parse label
        labels = {sampleId:data_hrv['labels']}
    
    # # save pkl
    if config.param_feature.save_single_pkl:
        save_path = os.path.join(save_hrv_dir, f'{sampleId}.pkl')
        joblib.dump(data_hrv, save_path)
        save_path = os.path.join(save_time_dir, f'{sampleId}.pkl')
        joblib.dump(data_time, save_path)
    return fea_lead_all, labels, failed_list



#%% main function
def extract_nk2_single_site_mp(config, site, leads=None):
    logger.info('-'*60)
    start_time = time.time()
    failed_list_all = []
    
    # initialization
    logger.info(f'Now Extract Neurokit2 Features: {site} to {OUT_HRV_ROOT}')
    save_hrv_dir = OUT_HRV_ROOT/site
    os.makedirs(save_hrv_dir, exist_ok=True)
    save_time_dir = OUT_TIME_ROOT/site
    os.makedirs(save_time_dir, exist_ok=True)
    NUMCPU = multiprocessing.cpu_count()-1 if config.param_feature.num_cpu<0 else config.param_feature.num_cpu


    # load header_files / recording_files
    data_directory = RAW_DATA_ROOT/site
    header_files, recording_files = find_challenge_files(data_directory)
    num_recordings = len(recording_files)
    if num_recordings==0:
        raise Exception('No data was provided.')

    # Extract features and labels from dataset: parallel processing
    logger.info('Extracting features and labels...')
    partial_func = partial(extract_data, config=config, header_files=header_files, recording_files=recording_files, leads=leads)
    fea_lead_all_tmp, label_all, failed_list_all = [], {}, []
    with Pool(processes=NUMCPU) as pool:
        for fea_lead, label, failed_list in tqdm(pool.istarmap(partial_func, zip(range(num_recordings))), total=num_recordings, disable=not(config.param_feature.print_bar)):
            if len(failed_list)>0:
                failed_list = [[site]+fail_item for fail_item in failed_list]
                failed_list_all.extend(failed_list)
            fea_lead_all_tmp.append(fea_lead)
            label_all.update(label)

    fea_lead_all = {}
    for lead in leads:
        fea_lead_all[lead] = pd.concat([f[lead] for f in fea_lead_all_tmp], axis=0)
                
    # sve site feature & label as single pkl
    joblib.dump([fea_lead_all, label_all], save_hrv_dir+"hrv_all.pkl")
    
    # log failed samples            
    with open(OUT_HRV_ROOT/f'../failed-{site}.json', 'w') as json_file:
        json.dump(failed_list_all, json_file, indent=4, sort_keys=True)
    logger.info(f'Finish {site}. Time usage: {time.time()-start_time}\n')
    
    
