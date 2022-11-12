#%%
from helper_code import *
import os, sys, json
import pandas as pd

from src.engine.utils import log_params_mlflow, log_hydra_mlflow
from src.engine.solver_raw import Solver
from time import time, ctime

import coloredlogs, logging
coloredlogs.install()
logger = logging.getLogger(__name__)  

from shutil import rmtree
from pathlib import Path

import hydra
from hydra import utils
import omegaconf
import mlflow as mf
from mlflow.tracking.client import MlflowClient

twelve_leads = ('I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6')
six_leads = ('I', 'II', 'III', 'aVR', 'aVL', 'aVF')
four_leads = ('I', 'II', 'III', 'V2')
three_leads = ('I', 'II', 'V2')
two_leads = ('I', 'II')


#%%
@hydra.main(config_path='./config/hydra', config_name="rsn_raw_6leads")
def main(config):
    if (config.param_model.batch_size>=96): config.param_model.batch_size=96

    # =============================================================================
    # check config have been run
    # =============================================================================
    # generate a fake run to get the to be stored parameter format of this run
    MLRUNS_DIR = utils.get_original_cwd()+'/mlruns'
    client = MlflowClient(tracking_uri=MLRUNS_DIR)
    exp = client.get_experiment_by_name(config.exp.exp_name)
    if exp is not None:
        exp_id = exp.experiment_id

        fake_run_name = "fake_run"
        mf.set_tracking_uri(MLRUNS_DIR)
        mf.set_experiment(config.exp.exp_name)
        with mf.start_run(run_name=fake_run_name) as run:
            log_params_mlflow(config)
            fake_run_id = run.info.run_id
        fake_run = client.get_run(fake_run_id)
        fake_run_params = fake_run.data.params

        # delete all failed
        all_run_infos = client.list_run_infos(experiment_id=exp_id)
        for run_info in all_run_infos:
            if run_info.status=="FAILED":
                client.delete_run(run_info.run_id)

        # check all existed run
        all_run_infos = client.list_run_infos(experiment_id=exp_id)
        for run_info in all_run_infos:
            if run_info.run_id==fake_run_id: continue # skip fakerun itself
            run = client.get_run(run_info.run_id)
            params = run.data.params
            metrics = run.data.metrics
            if params==fake_run_params and run_info.status=="FINISHED":
                logger.warning(f'Find Exist Run with cv_test_cm: {round(metrics["cv_test_cm"],3)}')
                client.delete_run(fake_run_id)
                return metrics["cv_test_cm"]
        client.delete_run(fake_run_id)


    # =============================================================================
    # initialize
    # =============================================================================
    # print(config)
    time_start = time()
    logger.info(f"Data Path: {Path(config.path.data_directory).absolute()} | MLflow Path: {Path(config.path.mlflow_dir).absolute()}")
    logger.info(f"Training Site:{config.exp.train_sites} | Eval Sites:{config.exp.eval_sites}")


    # =============================================================================
    # Setup Solver
    # =============================================================================
    logger.info("\n"+'='*120)
    logger.info('Training 12-lead ECG model...')
    if config.param_loader.num_leads==12:
        config.param_loader.leads = twelve_leads
    elif config.param_loader.num_leads==6:
        config.param_loader.leads = six_leads
    elif config.param_loader.num_leads==4:
        config.param_loader.leads = four_leads
    elif config.param_loader.num_leads==3:
        config.param_loader.leads = three_leads
    elif config.param_loader.num_leads==2:
        config.param_loader.leads = two_leads
    solver = Solver(config)

#%%
    # =============================================================================
    # Run
    # =============================================================================
    with omegaconf.open_dict(config):
        config.path.data_directory = str((Path(utils.get_original_cwd())/config.path.data_directory).absolute())
        config.path.official_data_directory = str((Path(utils.get_original_cwd())/config.path.official_data_directory).absolute())
    
    mf.set_tracking_uri(MLRUNS_DIR)
    mf.set_experiment(config.exp.exp_name)
    with mf.start_run(run_name=f"{config.exp.N_fold}fold_CV_Results") as run:
        log_params_mlflow(config)
        cv_metrics = solver.evaluate()
        mf.log_metrics(cv_metrics)
        log_hydra_mlflow(name="tune_raw")
        
    time_now = time()
    logger.warning(f"{config.param_loader.num_leads}-Lead Time Used: {ctime(time_now-time_start)}")
    return cv_metrics["cv_test_cm"]


#%%
if __name__ == "__main__":
    main()
    # rmtree("./multirun")
