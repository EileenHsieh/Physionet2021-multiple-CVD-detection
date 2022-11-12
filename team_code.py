#!/usr/bin/env python

# Edit this script to add your team's training code.
# Some functions are *required*, but you can edit most parts of the required functions, remove non-required functions, and add your own functions.

################################################################################
#
# Imported functions and variables
#
################################################################################

#%%
# Import functions. These functions are not required. You can change or remove them.
import os

# from nevergrad.parametrization.core import P 
from helper_code import *
import numpy as np, os, joblib

from src.engine.utils import (set_device, init_mlflow, log_params_mlflow, 
                              to_numpy, SCORED_CLASSES, patch_logit_To_Scalar_Binary,
                              get_rb_feature, wfdb_process)

from src.engine.solver_raw import Solver
from src.engine.loaders.raw_testloader import TestDataset, my_collate_fn
from src.dataset_generator.extract_nk2_single_model import extract_data
from time import time, ctime

import coloredlogs, logging
coloredlogs.install()
logger = logging.getLogger(__name__)  

from omegaconf import OmegaConf
from shutil import rmtree
from pathlib import Path
import mlflow as mf

from src.engine.models import *
from torch.utils.data import DataLoader
from glob import glob
import torch


# Define the Challenge lead sets. These variables are not required. You can change or remove them.
twelve_leads = ('I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6')
six_leads = ('I', 'II', 'III', 'aVR', 'aVL', 'aVF')
four_leads = ('I', 'II', 'III', 'V2')
three_leads = ('I', 'II', 'V2')
two_leads = ('I', 'II')
lead_sets = (twelve_leads, six_leads, four_leads, three_leads, two_leads)

nb_entry = 'entry9'
################################################################################
#
# Training model function
#
################################################################################

# Train your model. This function is *required*. You should edit this function to add your code, but do *not* change the arguments of this function.
def training_code(data_directory, model_directory):
    """
    Training code that load config, build solver and log the results in mlflow. It will execute for every lead.

    Parameters
    ----------
    data_directory: str
        Path to the datasets
    model_directory: str
        Path to the model folder
    
    Returns
    -------
    
    """

    #%%
    # =============================================================================
    # initialize
    # =============================================================================
    for leads in lead_sets:
        config_path = f'./config/{nb_entry}/rsn_raw_{len(leads)}leads.yaml'

        time_start = time()
        config = OmegaConf.load(config_path)
        
        config.path.official_data_directory = data_directory
        config.path.model_directory = model_directory
        # config.path.mlflow_dir = model_directory+'/mlruns'
        logger.info(f"Data Path: {Path(config.path.data_directory).absolute()} | MLflow Path: {Path(config.path.mlflow_dir).absolute()}")
        logger.info(f"Training Site:{config.exp.train_sites} | Eval Sites:{config.exp.eval_sites}")
        
        logger.info("\n"+'='*120)
        logger.info(f'Training {len(leads)}-lead ECG model...')
        config.param_loader.leads = leads
        config.param_loader.num_leads = len(config.param_loader.leads)

        solver = Solver(config)
        self=solver
        #%%
        init_mlflow(config)
        with mf.start_run(run_name=f"{config.exp.N_fold}fold_CV_Results") as run:
            log_params_mlflow(config)
            cv_metrics = solver.evaluate()
            print(cv_metrics)
            mf.log_metrics(cv_metrics)
        time_now = time()
        logger.warning(f"{len(leads)}-Lead Time Used: {ctime(time_now-time_start)}")

        # =============================================================================
        # output
        # =============================================================================
        pytorch_lightning_ckpt_dir = Path("./lightning_logs/")
        if pytorch_lightning_ckpt_dir.exists(): rmtree(pytorch_lightning_ckpt_dir)


################################################################################
#
# Running trained model function
#
################################################################################

# Run your trained model. This function is *required*. You should edit this function to add your code, but do *not* change the arguments of this function.
@torch.no_grad()
def run_model(model, header, recording):
    """
    Run the trained model. This will be called in test_model.py

    Parameters
    ----------
    model: dict
        Keys include 'ml' and 'dl', model['ml'] contains the pretrained rule-based models,
        while model['dl'] is a list of classfier object models
    header: str
        Output of load_header function in helper_code.py
    recording: array
        Output of load_recording function in helper_code.py
        
    Returns
    -------
    classes: array
        Classes of interest
    labels: array
        Predicted binary labels for each classes
    probabilities:
        Predicted probability for each classes
    
    """
    # Dataloader
    config_path = f'./config/{nb_entry}/rsn_raw_{len(recording)}leads.yaml'
    config = OmegaConf.load(config_path)
    print("config", config)

    try:
        # Pre-process the recording as WFDB did
        
        recording = wfdb_process(header, recording, config.param_loader.leads)

        testSet = TestDataset(config, header, recording)
        testLoader = DataLoader(dataset=testSet, batch_size=len(testSet), shuffle=False, collate_fn=my_collate_fn) 
        print("lenght of testloader", len(testLoader))
        logger.info(f"Test:{len(testSet)}") 
        testData = next(iter(testLoader))[0]

        # Predict with top_N deep learning models:
        top_n_prob = []
        top_n_lab = []
        for n_model in model['dl']: 
            n_model.eval()
            # Prediction
            logits = n_model(testData) 
            # turn logits to -inf is is_garbage = True
            for i, is_garbage in enumerate(testData["is_garbage"]):
                if is_garbage:  logits[i] = torch.tensor(-np.inf)

            probabilities, labels = patch_logit_To_Scalar_Binary(logits, n_model.bin_thre_arr.cpu()) 
            probabilities = to_numpy(probabilities)
            labels = to_numpy(labels)
            top_n_prob.append(probabilities)
            top_n_lab.append(labels)
        
        # Predict specific classes with rule-based models:
        if len(model['ml'])!=0:
            for class_name, rb_model in model['ml'].items():
                class_idx = SCORED_CLASSES.index(class_name)
                if (class_name=='6374002') and labels[class_idx]==1: # replace when resnet predict 1

                    # Extract data
                    extract_lead = 'aVL'
                    sub_fea, _, _ = extract_data(header=header, recording=recording, 
                                                lead=extract_lead)
                    if sub_fea.shape==(1,115):
                        # Extract feature
                        fea_combination = ['q', 'q%', 'r', 'r%', 's', 's%', 't', 't%', 'qs', 'qs%']
                        feature = get_rb_feature(sub_fea, fea_combination)
                        # Predict
                        rb_pred = rb_model.predict(feature)[0]
                        print(f"{class_name} changes from {labels[class_idx]} to {rb_pred}")
                        labels[class_idx] = rb_pred
                    
                    else: 
                        # No feature could be extract, remain the resnet result
                        labels[class_idx] = labels[class_idx]
                        

                    
                elif class_name=='251146004' and labels[class_idx]==0: # replace when resnet predict 0
                    # Extract feature
                    extract_lead = 'II'
                    sub_fea, _, _ = extract_data(header=header, recording=recording, 
                                                lead=extract_lead)
                    if sub_fea.shape==(1,115):
                        # Extract feature
                        fea_combination = ['q','q%','r','r%','s','s%','t','t%']
                        feature = get_rb_feature(sub_fea, fea_combination)
                        # Predict
                        rb_pred = rb_model.predict(feature)[0]
                        print(f"{class_name} changes from {labels[class_idx]} to {rb_pred}")
                        labels[class_idx] = rb_pred

                    else: 
                        # No feature could be extract, remain the resnet result
                        labels[class_idx] = labels[class_idx]

        probabilities = np.mean(np.array(top_n_prob), 0)
        print("prob", probabilities)
        labels = np.any(np.array(top_n_lab), 0).astype(int)
        print("labels", labels)
        classes = np.array(SCORED_CLASSES)
        print("classes", classes)
    except:
        classes = np.array(SCORED_CLASSES)
        probabilities = np.tile(0.5,len(classes))
        labels = np.zeros(len(classes)).astype(int)
    return classes, labels, probabilities

################################################################################
#
# File I/O functions
#
################################################################################

# Load a trained model. This function is *required*. You should edit this function to add your code, but do *not* change the arguments of this function.
def load_model(model_directory, leads):
    """
    Run the trained model. This will be called in test_model.py

    Parameters
    ----------
    model_directory: str
        Path to the model folder
    leads: tuple
        Tuple of the leads of interest
        
    Returns
    -------
    model: dict
        Keys include 'ml' and 'dl', model['ml'] contains the pretrained rule-based models,
        while model['dl'] is a list of classfier object models
    
    
    """
    model = {}
    config_path = f'./config/{nb_entry}/rsn_raw_{len(leads)}leads.yaml'
    config = OmegaConf.load(config_path)
    # solver = Solver(config)
    top_N = config.exp.top_N
    model_path = glob(f'{model_directory}/{len(leads)}leads*.ckpt')
    if len(model_path)==0:
        print(f"Not found model for {len(leads)} leads")
    model_results = [float(m.split('=')[-1][:5]) for m in model_path]
    top_N_idx = sorted(range(len(model_results)), key=lambda i: model_results[i])[-top_N:]

    model['dl'] = [AttnRsnClassic.load_from_checkpoint(model_path[idx]) for idx in top_N_idx]
    # model = [solver._get_model(ckpt_path_abs=model_path[idx]) for idx in top_N_idx]
    print("length of deep learning models", len(model['dl']))

    # load rule-base model
    rb_model = {}
    if ("is_rule_base" in config.exp) and (config.exp.is_rule_base):
        print("...Loading 251146004 rule-based model")
        rb_model["251146004"] = joblib.load('./rb_251146004.pkl')
        if len(leads)==12 or len(leads)==6:
            print("...Loading 6374002 rule-based model")
            rb_model["6374002"] = joblib.load('./rb_6374002.pkl')
    model['ml'] = rb_model

    return model


################################################################################
#
# Feature extraction function
#
################################################################################

# Extract features from the header and recording. This function is not required. You can change or remove it.
def get_features(header, recording, leads):
    # Extract age.
    age = get_age(header)
    if age is None:
        age = float('nan')

    # Extract sex. Encode as 0 for female, 1 for male, and NaN for other.
    sex = get_sex(header)
    if sex in ('Female', 'female', 'F', 'f'):
        sex = 0
    elif sex in ('Male', 'male', 'M', 'm'):
        sex = 1
    else:
        sex = float('nan')

    # Reorder/reselect leads in recordings.
    recording = choose_leads(recording, header, leads)

    # Pre-process recordings.
    adc_gains = get_adc_gains(header, leads)
    baselines = get_baselines(header, leads)
    num_leads = len(leads)
    for i in range(num_leads):
        recording[i, :] = (recording[i, :] - baselines[i]) / adc_gains[i]

    # Compute the root mean square of each ECG lead signal.
    rms = np.zeros(num_leads)
    for i in range(num_leads):
        x = recording[i, :]
        rms[i] = np.sqrt(np.sum(x**2) / np.size(x))

    return age, sex, rms

