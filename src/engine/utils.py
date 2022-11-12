import os, sys
import torch
import torch.nn.functional as F
import json
import mlflow as mf
from mlflow.tracking import MlflowClient
import numpy as np
from pathlib import Path
from lib.evaluation_2021.evaluate_model import *
from mlflow.utils.autologging_utils.safety import try_mlflow_log
from shutil import rmtree
import coloredlogs, logging
coloredlogs.install()
logger = logging.getLogger(__name__)  


#%%
SITE_NAME_MAP = {'CPSC2018':'CPSC', 'CPSC2018_2':'CPSC_Extra', 'StPetersburg':'StPetersburg', 'PTB':'PTB', 'PTBXL':'PTB_XL', 'Ga':'Georgia', 'Chapman_Shaoxing':'Chapman_Shaoxing', 'Ningbo':'Ningbo'}

#%%
TOUGH_CLASSES = ['6374002', '251146004', '365413008', '111975006', '164917005']
EQ_CLASSES = [[['733534002','164909002'],'733534002|164909002'], [['713427006','59118001'],'713427006|59118001'], [['284470004', '63593006'], '284470004|63593006'], [['427172004', '17338001'],'427172004|17338001']]


#%%
def to_numpy(x):
    if x.is_cuda: return x.detach().cpu().data.numpy()
    return x.detach().data.numpy()

# =============================================================================
# data
# =============================================================================
ALL_TRAIN_SITES = ["WFDB_CPSC2018", "WFDB_CPSC2018_2", "WFDB_StPetersburg", "WFDB_PTB", "WFDB_PTBXL", "WFDB_Ga", "WFDB_ChapmanShaoxing", "WFDB_Ningbo"]


# =============================================================================
# wfdb
# =============================================================================
def get_wafdb_age(header_info, impute_nan=50):
    try:
        age = np.asarray(int(header_info['Age'])).astype('float32')
    except:
        if impute_nan:
            age = np.asarray(impute_nan)
        else:
            age = np.asarray(np.nan)
    return age


# =============================================================================
# feature
# =============================================================================
def map_sex(sex, impute_nan=1):
    if sex in ('Female', 'female', 'F', 'f'):
        sex = np.asarray([0])
    elif sex in ('Male', 'male', 'M', 'm'):
        sex = np.asarray([1])
    else:
        if impute_nan:
            sex = np.asarray([impute_nan])
        else:
            sex = np.asarray([np.nan])
    return sex.astype("float32")

# =============================================================================
# label
# =============================================================================

def replace_equivalent_classes(classes, equivalent_classes):
    """
    Replace the labels in predefined equivalent pairs with its equivalent class. 

    Parameters
    ----------
    classes: list
        List of the labels
    equivalent_classes: list of list
        [[['733534002','164909002'],'733534002|164909002'], [['713427006','59118001'],'713427006|59118001'], [['284470004', '63593006'], '284470004|63593006'], [['427172004', '17338001'],'427172004|17338001']]

        
    Returns
    -------
    classes: list
        List of the replaced labels
    
    
    """
    for j, x in enumerate(classes):
        for multiple_classes in equivalent_classes:
            if x in multiple_classes[0]:
                classes[j] = multiple_classes[1] # Use the first class as the representative class.
    return classes


# =============================================================================
# loader
# =============================================================================
def get_nested_fold_idx(kfold):
    for fold_test_idx in range(kfold):
        fold_val_idx = (fold_test_idx+1)%kfold
        fold_train_idx = [fold for fold in range(kfold) if fold not in [fold_test_idx, fold_val_idx]]
        yield fold_train_idx, [fold_val_idx], [fold_test_idx]

# =============================================================================
# metric
# =============================================================================
def logit_To_Scalar_Binary(logit:torch.tensor, bin_thre_arr=None):
    """
    Convert the predicted logit to predict probabilty, and apply the binary threshold to obtain binary output. 

    Parameters
    ----------
    logit: tensor, size: (N patients, n_classes)
        Predicted logit before sigmoid.
    bin_thre_arr: tensor
        Binary threshold for each classes 
        
    Returns
    -------
    scalar_output: tensor
        Predict probability after sigmoid
    binary_output: tensor
        Predicted binary outputs  
    
    """
    scalar_output = torch.sigmoid(logit)
    binary_output = torch.zeros_like(scalar_output)
    if bin_thre_arr is not None:
        binary_output[scalar_output>=bin_thre_arr] = 1
        binary_output[scalar_output<bin_thre_arr] = 0
    else:
        binary_output[scalar_output>=0.5] = 1
        binary_output[scalar_output<0.5] = 0
    return scalar_output, binary_output

def patch_logit_To_Scalar_Binary(logit:torch.tensor, bin_thre_arr=None):
    """
    Convert the predicted logit to predict probabilty, and apply the binary threshold to obtain binary output. 
    This is applied in inference, when each patient has different length of recording,
    we will segment the recording into patches and ensemble the prediction of every patch.

    Parameters
    ----------
    logit: tensor, size: (N patches for the same patient, n_classes)
        Predicted logit before sigmoid.
    bin_thre_arr: tensor
        Binary threshold for each classes 
        
    Returns
    -------
    scalar_output: tensor
        Predict probability after sigmoid
    binary_output: tensor
        Predicted binary outputs  
    
    """
    scalar_output = torch.sigmoid(logit)
    scalar_output = scalar_output.mean(dim=0) # mean the probabilities(sigmoid logits) of patches
    binary_output = torch.zeros_like(scalar_output)
    if bin_thre_arr is not None:
        binary_output[scalar_output>=bin_thre_arr] = 1
        binary_output[scalar_output<bin_thre_arr] = 0
    else:
        binary_output[scalar_output>=0.5] = 1
        binary_output[scalar_output<0.5] = 0
    return scalar_output, binary_output

def get_cv_logits_metrics(fold_errors, model, outputs, mode="val"):
    scalar_outputs, binary_outputs = logit_To_Scalar_Binary(outputs["logit"], model.bin_thre_arr)
    metrics = model._cal_metric(torch.tensor(outputs["logit"]), torch.tensor(outputs["label"]))
    fold_errors[f"{mode}_binary_outputs"].append(binary_outputs)
    fold_errors[f"{mode}_scalar_outputs"].append(scalar_outputs)
    fold_errors[f"{mode}_label"].append(outputs["label"])
    fold_errors[f"{mode}_auroc"].append([metrics["auroc"]])
    fold_errors[f"{mode}_auprc"].append([metrics["auprc"]])
    fold_errors[f"{mode}_cm"].append([metrics["cm"]])


# =============================================================================
# evaluation

# =============================================================================
NORMAL_CLASS = set(['426783006'])
weights_file = Path('lib/evaluation_2021/weights.csv').absolute()
CLASSES, WEIGHTS = load_weights(weights_file)
SCORED_CLASSES = load_table(Path('lib/evaluation_2021/weights.csv'))[0]


#%%
def set_device(gpu_id):
    # Manage GPU availability
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    if gpu_id != "": 
        torch.cuda.set_device(0)
        
    else:
        n_threads = torch.get_num_threads()
        n_threads = min(n_threads, 8)
        torch.set_num_threads(n_threads)
        print("Using {} CPU Core".format(n_threads))
        
        
#%%
import multiprocessing.pool as mpp
# istarmap, hgy
def istarmap(self, func, iterable, chunksize=1):
    """starmap-version of imap
    """
    if self._state != mpp.RUN:
        raise ValueError("Pool not running")

    if chunksize < 1:
        raise ValueError(
            "Chunksize must be 1+, not {0:n}".format(
                chunksize))

    task_batches = mpp.Pool._get_tasks(func, iterable, chunksize)
    result = mpp.IMapIterator(self)
    self._taskqueue.put(
        (
            self._guarded_task_generation(result._job,
                                          mpp.starmapstar,
                                          task_batches),
            result._set_length
        ))
    return (item for chunk in result for item in chunk)
mpp.Pool.istarmap = istarmap

def init_mlflow(config):
    mf.set_tracking_uri(str(Path(config.path.mlflow_dir).absolute()))  # set up connection
    mf.set_experiment(config.exp.exp_name)          # set the experiment

def log_params_mlflow(config):
    mf.log_params(config.get("exp"))
    mf.log_params(config.get("param_feature"))
    try_mlflow_log(mf.log_params, config.get("param_preprocess"))
    mf.log_params(config.get("param_loader"))
    mf.log_params(config.get("param_trainer"))
    mf.log_params(config.get("param_early_stop"))
    if config.get("param_aug"):
        if config.param_aug.get("filter"):
            for k,v in dict(config.param_aug.filter).items():
                mf.log_params({k:v})
    # mf.log_params(config.get("param_aug"))
    mf.log_params(config.get("param_model"))

def log_hydra_mlflow(name):
    mf.log_artifact(os.path.join(os.getcwd(), '.hydra/config.yaml'))
    mf.log_artifact(os.path.join(os.getcwd(), '.hydra/hydra.yaml'))
    mf.log_artifact(os.path.join(os.getcwd(), '.hydra/overrides.yaml'))
    mf.log_artifact(os.path.join(os.getcwd(), f'{name}.log'))
    rmtree(os.path.join(os.getcwd()))

def get_ckpt(r):
    ckpts = [f.path for f in MlflowClient().list_artifacts(r.info.run_id, "restored_model_checkpoint")]
    return r.info.artifact_uri, ckpts
    
class TuneBinaryThreshold(object):
    def __init__(self, model):
        self.model = model
    def __enter__(self):
        self.model.update_bin_thre_arr = True
        logger.info("Tuning Bianry Thresholding !")
        return self.model
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.model.update_bin_thre_arr = False


#%% Rule base feature extractor
from scipy import signal
from neurokit2.signal import signal_filter
from helper_code import *
import neurokit2 as nk2

def butterworth(data, lowcut, highcut, fs, order):             
    nyq = fs * 0.5  # https://en.wikipedia.org/wiki/Nyquist_frequency
    if (lowcut is not None) & (highcut is not None):
        # Band Pass filter
        lowcut = lowcut / nyq
        highcut = highcut / nyq
        b, a = signal.butter(order, [lowcut, highcut], btype='band', analog=False)
        
    elif lowcut:
        # High Pass filter
        lowcut = lowcut / nyq  
        b, a = signal.butter(order, lowcut, btype='high', analog=False)
        
    elif highcut:
        # Low Pass filter
        highcut = highcut / nyq
        b, a = signal.butter(order, highcut, btype='low', analog=False)
    
    return signal.filtfilt(b, a, data)

def functional(arr):
    arr = np.array(arr)
    return np.stack([np.min(arr),np.mean(arr),np.max(arr),np.std(arr),
            np.percentile(arr, 25, axis = 0),
            np.percentile(arr, 75, axis = 0),
            np.median(arr, axis = 0)
            ])

def get_rb_feature(fea_df, fea_combination):
    """
    To generate the features for pre-trained rule-based model.

    Parameters
    ----------
    fea_df: df
        All the handcrafted features extract by src/dataset_generator/extract_nk2_single_model.py: extract_fea
    fea_combination: list
        Features of interest, ['q','q%','r','r%','s','s%','t','t%'], value and percentage of q, r, s, t amplitude 
        
    Returns
    -------
    feature: array
        Features of interest
        
    """
    fea_cols = ['top5_rmeans']
    fea_cols += [obj+stc for obj in ['p','p%','q','q%','r','r%','s','s%','t','t%','qt','qt%','qs','qs%','qq','qq%'] 
                for stc in ['min', 'mean', 'max', 'std', '25', '75', '50']]
    fea_cols += ['age', 'sex']
    fea_df.columns = fea_cols
    cols = [obj+stc for obj in fea_combination 
        for stc in ['min', 'mean', 'max', 'std', '25', '75', '50']]
    fea_mask = fea_df[cols]
    feature = np.nan_to_num(fea_mask.to_numpy())
    return feature

# Get fmt from header.
def get_fmt(header):
    fmt = '16'#None
    for i, l in enumerate(header.split('\n')):
        if i==1:
            try:
                fmt = l.split(' ')[3]
                break
            except:
                pass
        else:
            continue
    return fmt

# Process the recording loaded from load_recording as WFDB does
def wfdb_process(header, recording, leads):
    from wfdb.io._signal import _digi_nan
    fmt = get_fmt(header) # default: '16' (signal written in 16 bits or 8 bits are most common)
    baseline = get_baselines(header, leads).reshape(-1, 1) # default: zero array
    adc_gain = get_adc_gains(header, leads).reshape(-1, 1) # default: zero array
    d_nans = _digi_nan(fmt) 
    nanlocs = recording == d_nans
    # Do float conversion immediately to avoid potential under/overflow
    # of efficient int dtype
    recording = recording.astype('float64', copy=False)
    np.subtract(recording, baseline, recording)
    np.divide(recording, adc_gain, recording)
    recording[nanlocs] = np.nan
    return recording

def remove_nan(recording):
    nans = np.where(np.isnan(recording))
    if len(nans[0])==0: return recording
    else:
        nan_recording = np.zeros_like(recording[0])
        for i in range(len(nans[0])):
            start = nans[1][i]-50 if nans[1][i]-50>=0 else 0
            end = nans[1][i]+50 if nans[1][i]+50<len(nan_recording) else len(nan_recording)-1
            nan_recording[start:end] = 1
        
        non_nan_ranges = zero_runs(nan_recording)
        kept_seg = []
        kept_seg_len = []
        for r in non_nan_ranges:
            start, end = r
            seg = recording[:, start:end]
            kept_seg.append(seg)
            kept_seg_len.append(seg.shape[1])
        recording = kept_seg[np.argmax(kept_seg_len)]

        return recording

def zero_runs(a):
    # Create an array that is 1 where a is 0, and pad each end with an extra 0.
    iszero = np.concatenate(([0], np.equal(a, 0).view(np.int8), [0]))
    absdiff = np.abs(np.diff(iszero))
    # Runs start and end where absdiff is 1.
    ranges = np.where(absdiff == 1)[0].reshape(-1, 2)
    return ranges