#%%
import os
from shutil import rmtree
import random
import pandas as pd
import numpy as np 
# import seaborn as sn
import matplotlib.pyplot as plt
from matplotlib import gridspec

from src.engine.loaders.raw_loader import RawDataModule, RawDataset, my_collate_fn
from src.engine.utils import get_ckpt, get_cv_logits_metrics, get_nested_fold_idx, TuneBinaryThreshold
from src.engine.utils import NORMAL_CLASS, CLASSES, WEIGHTS, SCORED_CLASSES, EQ_CLASSES, logit_To_Scalar_Binary
from lib.evaluation_2021.evaluate_model import *

from src.engine.models import *
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint, StochasticWeightAveraging
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.core import saving 
from pytorch_lightning import seed_everything

import mlflow as mf
import omegaconf
import sys, tempfile, json

import coloredlogs, logging
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

coloredlogs.install()
logger = logging.getLogger(__name__)  
# logger.setLevel(logging.DEBUG)


#%%
class Solver:
    DEFAULTS = {}   

    def __init__(self, config):
        self.config = config
        self.leads = config.param_loader.leads
        self.NORMAL_CLASS = NORMAL_CLASS
        self.SCORED_CLASSES = SCORED_CLASSES # all target classes without merge (24)
        with omegaconf.open_dict(self.config):
            self.config.param_model.output_size = len(self.SCORED_CLASSES)
            
            
        SEED = self.config.exp.random_state
        random.seed(SEED)
        np.random.seed(SEED)
        torch.manual_seed(SEED)
        seed_everything(SEED)

        
            
    # other functions
    def _get_model(self, pos_weight=None, ckpt_path_abs=None):
        model = None
        if not ckpt_path_abs:
            if self.config.exp.model_type == "cnn_vanilla":
                model = CnnVanilla(self.config.param_model, random_state=self.config.exp.random_state, pos_weight=pos_weight)
            elif self.config.exp.model_type == "rsn_vanilla":
                model = RsnVanilla(self.config.param_model, random_state=self.config.exp.random_state, pos_weight=pos_weight)
            elif self.config.exp.model_type == "attnrsn":
                model = AttnRsn(self.config.param_model, random_state=self.config.exp.random_state, pos_weight=pos_weight)
            elif self.config.exp.model_type == "rsn_classic":
                model = RsnClassic(self.config.param_model, random_state=self.config.exp.random_state, pos_weight=pos_weight)
            elif self.config.exp.model_type == "attn_rsn_classic":
                model = AttnRsnClassic(self.config.param_model, random_state=self.config.exp.random_state, pos_weight=pos_weight)
            return model
        else:
            if self.config.exp.model_type == "cnn_vanilla":
                model = CnnVanilla.load_from_checkpoint(ckpt_path_abs)
            elif self.config.exp.model_type == "rsn_vanilla":
                model = RsnVanilla.load_from_checkpoint(ckpt_path_abs)
            elif self.config.exp.model_type == "attnrsn":
                model = AttnRsn.load_from_checkpoint(ckpt_path_abs)
            elif self.config.exp.model_type == "rsn_classic":
                model = RsnClassic.load_from_checkpoint(ckpt_path_abs)
            elif self.config.exp.model_type == "attn_rsn_classic":
                model = AttnRsnClassic.load_from_checkpoint(ckpt_path_abs)
            return model

    
    def evaluate(self):
        """
        Training procedure
        """
#%%
        fold_errors_template = {"binary_outputs":[],
                       "scalar_outputs":[],
                       "label":[],
                       "auroc":[],
                       "auprc":[],
                       "cm":[]}
        fold_errors = {f"{mode}_{k}":[] for k,v in fold_errors_template.items() for mode in ["val","test"]}
        # =============================================================================
        # data module
        # =============================================================================
        dm = RawDataModule(self.config)
        dm.setup()
        
#%%
        # for foldIdx in range(self.config.exp.N_fold):
        for foldIdx, (folds_train, folds_val, folds_test) in enumerate(get_nested_fold_idx(self.config.exp.N_fold)):
            if foldIdx>(self.config.exp.top_N-1):   break
            # init data module
            logger.info("== CROSS-SUBJECT FOLD [{}/{}] ==".format(foldIdx+1, self.config.exp.N_fold))
            dm.setup_kfold(folds_train, folds_val, folds_test)
            pos_weight = dm._cal_train_class_weight() if self.config.param_model.is_class_weight else None
            logger.info(f"Train:{len(dm.train_dataloader().dataset)} | Val:{len(dm.val_dataloader().dataset)} | Test:{len(dm.test_dataloader().dataset)}")
#%%
            # init model
            model = self._get_model(pos_weight=pos_weight)
            early_stop_callback = EarlyStopping(**dict(self.config.param_early_stop))
            checkpoint_callback = ModelCheckpoint(**dict(self.config.logger.param_ckpt))
            lr_logger = LearningRateMonitor()
            if self.config.get("param_swa"):
                swa_callback = StochasticWeightAveraging(**dict(self.config.param_swa))
                trainer = MyTrainer(**dict(self.config.param_trainer), callbacks=[early_stop_callback, checkpoint_callback, lr_logger ], deterministic=True)
            else:
                trainer = MyTrainer(**dict(self.config.param_trainer), callbacks=[early_stop_callback, checkpoint_callback, lr_logger ], deterministic=True)

            # trainer main loop
            mf.pytorch.autolog()
            with mf.start_run(run_name=f"cv{foldIdx}", nested=True) as run:
                # init

                # train
                trainer.fit(model, dm)
                print("run_id", run.info.run_id)
                artifact_uri, ckpt_path = get_ckpt(mf.get_run(run_id=run.info.run_id))

                # load best ckpt
                ckpt_path_abs = str(Path(artifact_uri)/ckpt_path[0])
                # if ":" in ckpt_path_abs:
                #     ckpt_path_abs = ckpt_path_abs.split(":",1)[1]
                model = self._get_model(ckpt_path_abs=ckpt_path_abs)

                # update bianry thresholding
                with TuneBinaryThreshold(model) as model:
                    # model.update_bin_thre_arr = False
                    val_outputs = trainer.validate(model=model, val_dataloaders=dm.val_dataloader(), verbose=False)
                    test_outputs = trainer.test(model=model, test_dataloaders=dm.test_dataloader(), verbose=True)

                # save updated model
                trainer.model = model
                trainer.save_checkpoint(ckpt_path_abs)

#%%
                # reinference to calculate site-wise metric
                for eval_site in self.config.exp.eval_sites:
                    fold_data_scored_all = {mode:[[] for _ in range(self.config.exp.N_fold)] for mode in ["train","eval"]}
                    fold_data_unscored_all = {mode:[[] for _ in range(self.config.exp.N_fold)] for mode in ["train","eval"]}
                    for site_foldIdx in range(self.config.exp.N_fold):
                        # add scored samples
                        fold_data_json = Path(self.config.path.data_directory)/"CVFolds"/f"cv-{self.config.exp.N_fold}_order-{self.config.exp.N_fold_Order}"/eval_site/f"fold_{site_foldIdx}_scored.json"
                        with open(fold_data_json, "r") as f:
                            sampleIds = json.load(f)
                            fold_data = [f"{eval_site}/{sampleId}" for sampleId in sampleIds]
                            fold_data_scored_all["eval"][site_foldIdx]+=fold_data
                    
                    dataset = RawDataset(self.config,
                                                fold_data_scored_all.copy(),
                                                fold_data_unscored_all,
                                                folds = folds_test,
                                                transform=None,
                                                mode="test")
                    test_dataloader = DataLoader(dataset=dataset, batch_size=self.config.param_model.batch_size, num_workers=self.config.param_loader.num_workers, shuffle=False, pin_memory=False, collate_fn=my_collate_fn) 
                    site_test_outputs = trainer.test(model=model, test_dataloaders=test_dataloader, verbose=False)
                    
                    scalar_outputs, binary_outputs = logit_To_Scalar_Binary(site_test_outputs["logit"], model.bin_thre_arr)
                    metrics = model._cal_metric(torch.tensor(site_test_outputs["logit"]), torch.tensor(site_test_outputs["label"]))
                    metrics = {f'test/{k}/{eval_site}':round(v,3) for k,v in metrics.items()}
                    logger.info(f"\t Test site {eval_site}: {metrics}")
                    mf.log_metrics(metrics)

                # clear redundant mlflow models (save disk space)
                redundant_model_path = Path(artifact_uri)/'model'
                if redundant_model_path.exists(): rmtree(redundant_model_path)

            logger.info(f"bin_thre_arr:{model.bin_thre_arr}")


#%%            
            # =============================================================================
            # check validation/test metrics and logits
            # =============================================================================
            get_cv_logits_metrics(fold_errors, model, val_outputs, mode="val")
            get_cv_logits_metrics(fold_errors, model, test_outputs, mode="test")

            # Save to model directory
            cur_test_cm = fold_errors["test_cm"][foldIdx]
            print(cur_test_cm)
            os.makedirs(self.config.path.model_directory, exist_ok=True)
            trainer.save_checkpoint("{}/{}leads-fold{}-test_cm={:.3f}.ckpt".format(
                                                                           self.config.path.model_directory,
                                                                           self.config.param_loader.num_leads,
                                                                           foldIdx, 
                                                                           cur_test_cm[0]))
                
        # Cross Validation Summary
        fold_errors = {k:np.concatenate(v, axis=0) for k,v in fold_errors.items()}
        val_auroc, val_auprc, val_auroc_classes, val_auprc_classes = compute_auc(fold_errors["val_label"], fold_errors["val_scalar_outputs"])
        val_cm = compute_challenge_metric(WEIGHTS, fold_errors["val_label"], fold_errors["val_binary_outputs"], CLASSES, self.NORMAL_CLASS)
        test_auroc, test_auprc, test_auroc_classes, test_auprc_classes = compute_auc(fold_errors["test_label"], fold_errors["test_scalar_outputs"])
        test_cm = compute_challenge_metric(WEIGHTS, fold_errors["test_label"], fold_errors["test_binary_outputs"], CLASSES, self.NORMAL_CLASS)
        
        return {"cv_test_cm":test_cm, "cv_test_auroc":test_auroc, "cv_test_auprc":test_auprc, "cv_val_cm":val_cm, "cv_val_auroc":val_auroc, "cv_val_auprc":val_auprc}
