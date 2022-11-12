#%%
import coloredlogs, logging
coloredlogs.install()
logger = logging.getLogger(__name__)

import neurokit2 as nk
from scipy.io import loadmat
import numpy as np
import pandas as pd
import time

from typing import Dict, Tuple, List, Union
import warnings
from neurokit2.misc import NeuroKitWarning
warnings.filterwarnings("ignore", category=NeuroKitWarning)

from .helper_code import find_challenge_files, get_labels
from .helper_code import get_age, get_sex, get_leads, get_frequency
from .helper_code import get_adcgains, get_baselines
from .utils_neurokit2 import getFunctional10Dict



#%% extract the hrv features
def extract_ECG_hrv_single_lead(
        ecg_single_lead: np.array,
        sr: int,
        name: str,
        fea_types: list=['time', 'freq', 'rsa', 'nonlin', 'ECG_Rate', 'ECG_Quality', 'ECG_Phase_Completion_Atrial', 'ECG_Phase_Completion_Ventricular']) -> Union[Dict, pd.DataFrame]:
    '''
    Parameters
    ----------
    ecg_single_lead : np.array
        DESCRIPTION.
    sr : int
        DESCRIPTION.
    name : str
        DESCRIPTION.
    fea_types : list, optional
        DESCRIPTION. The default is ['time', 'freq', 'rsa', 'nonlin', 'ECG_Rate', 'ECG_Quality', 'ECG_Phase_Completion_Atrial', 'ECG_Phase_Completion_Ventricular'].
    Returns
    -------
    ecgFeaDict : Dict
        ECG features.
    ecg_signals : pd.DataFrame
        Processed ECG time series signals.
    '''
    
    assert (~np.isnan(ecg_single_lead).any()), '\n\nExist nan in ecg data, skip the data !\n\n'
    ecg_signals, rpeaks = nk.ecg_process(ecg_signal=ecg_single_lead, sampling_rate=sr)
    
    # hrv related features
    ecgFeaDict = {}
    if 'hrv_time' in fea_types: # https://neurokit2.readthedocs.io/en/latest/_modules/neurokit2/hrv/hrv_time.html
        hrv_time = nk.hrv_time(rpeaks, sampling_rate=sr) # d=14
        hrv_time.index = [name]
        ecgFeaDict.update({'hrv_time':hrv_time})
        
    if 'hrv_freq' in fea_types: # https://neurokit2.readthedocs.io/en/latest/_modules/neurokit2/hrv/hrv_frequency.html
        hrv_freq = nk.hrv_frequency(rpeaks, sampling_rate=sr) # d=9
        hrv_freq.index = [name]
        ecgFeaDict.update({'hrv_freq':hrv_freq})
        
    if 'hrv_rsa' in fea_types: # https://neurokit2.readthedocs.io/en/latest/_modules/neurokit2/hrv/hrv_rsa.html
        hrv_rsa = nk.hrv_rsa(ecg_signals=ecg_signals, rsp_signals=None, rpeaks=rpeaks, sampling_rate=sr, continuous=False)
        hrv_rsa = pd.DataFrame.from_dict(hrv_rsa, orient='index').T # d=5
        hrv_rsa.index = [name]
        ecgFeaDict.update({'hrv_rsa':hrv_rsa})
        
    if 'hrv_nonlin' in fea_types: # https://neurokit2.readthedocs.io/en/latest/_modules/neurokit2/hrv/hrv_nonlinear.html
        hrv_nonlin = nk.hrv_nonlinear(rpeaks, sampling_rate=sr) # d=29
        hrv_nonlin.index = [name]
        ecgFeaDict.update({'hrv_nonlin':hrv_nonlin})


    # signal functional related features
    if 'ecg_rate_func10' in fea_types:
        ecg_rate_func10 = getFunctional10Dict(ecg_signals['ECG_Rate'].values,'ECG_Rate') # d=10
        ecg_rate_func10 = pd.DataFrame(ecg_rate_func10, index=[name])
        ecgFeaDict.update({'ecg_rate_func10':ecg_rate_func10})
                                            
    if 'ecg_quality_func10' in fea_types:
        ecg_quality_func10 = getFunctional10Dict(ecg_signals['ECG_Quality'].values,'ECG_Quality') # d=10
        ecg_quality_func10 = pd.DataFrame(ecg_quality_func10, index=[name])
        ecgFeaDict.update({'ecg_quality_func10':ecg_quality_func10})
        
    if 'ecg_phase_comp_atrial_func10' in fea_types:
        ecg_phase_comp_atrial_func10 = getFunctional10Dict(ecg_signals['ECG_Phase_Completion_Atrial'].values,'ECG_Phase_Completion_Atrial') # d=10 
        ecg_phase_comp_atrial_func10 = pd.DataFrame(ecg_phase_comp_atrial_func10, index=[name])
        ecgFeaDict.update({'ecg_phase_comp_atrial_func10':ecg_phase_comp_atrial_func10})
    
    if 'ecg_phase_comp_vent_func10' in fea_types:
        ecg_phase_comp_vent_func10 = getFunctional10Dict(ecg_signals['ECG_Phase_Completion_Ventricular'].values,'ECG_Phase_Completion_Ventricular') # d=10
        ecg_phase_comp_vent_func10 = pd.DataFrame(ecg_phase_comp_vent_func10, index=[name])
        ecgFeaDict.update({'ecg_phase_comp_vent_func10':ecg_phase_comp_vent_func10})
    return ecgFeaDict, ecg_signals
        


#%%
def extract_ECG_hrv_dict_multilead(
        ecg_multi_lead: List[Tuple[int, np.array]],
        sr: int,
        sampleId: str,
        fea_types: List=['time', 'freq', 'rsa', 'nonlin', 'ECG_Rate', 'ECG_Quality', 'ECG_Phase_Completion_Atrial', 'ECG_Phase_Completion_Ventricular']) -> Union[Dict, Dict, List]:

    ecg_fea_multi_lead, time_seris_multi_lead, failed_list = {}, {}, []
    for leadId, ecg_single_lead in ecg_multi_lead:
        try:
            fea, time_seris = extract_ECG_hrv_single_lead(ecg_single_lead, sr, name=f'{sampleId}-{leadId}', fea_types=fea_types)
        except (ValueError,ZeroDivisionError,IndexError) as err:
            logger.error(f'{sampleId} / {leadId}: {err}')
            fea = None
            time_seris = None
            failed_list.append([sampleId, leadId, str(err)])

        ecg_fea_multi_lead.update({f'{sampleId}-{leadId}':fea})
        time_seris_multi_lead.update({f'{sampleId}-{leadId}':time_seris})
    return ecg_fea_multi_lead, time_seris_multi_lead, failed_list


#%%
'''
import matplotlib.pyplot as plt
for leadId, ecg_single_lead in ecg_multi_lead:
    figure = plt.figure()
    plt.plot(ecg_single_lead)
    plt.title(leadId)
'''

#%%
def extract_neurokit2_data(header, recording, sampleId, leads, fea_types) -> Union[Dict, Dict, List]:
    # label
    labels = get_labels(header)
    
    # demographic features: age
    age = get_age(header)
    if age is None:
        age = float('nan')
    
    # demographic features: sex. Encode as 0 for female, 1 for male, and NaN for other.
    sex = get_sex(header)
    if sex in ('Female', 'female', 'F', 'f'):
        sex = 0
    elif sex in ('Male', 'male', 'M', 'm'):
        sex = 1
    else:
        sex = float('nan')
    fea_demo = pd.DataFrame([[age, sex]], index=[sampleId], columns=['age','sex'])

    
    # Reorder/reselect leads in recordings.
    available_leads = get_leads(header)
    indices = list()
    for lead in leads:
        i = available_leads.index(lead)
        indices.append(i)
    recording = recording[indices, :]
    
    # Pre-process recordings.
    adc_gains = get_adcgains(header, leads)
    baselines = get_baselines(header, leads)
    num_leads = len(leads)
    for i in range(num_leads):
        recording[i, :] = (recording[i, :] - baselines[i]) / adc_gains[i]
    
    ecg_multi_lead = []
    for idx, leadId in enumerate(leads):
        ecg_single_lead = recording[idx,:]
        ecg_multi_lead.append((leadId, ecg_single_lead))
    
    
    # ecg features
    sr = int(get_frequency(header))
    fea_ECG_multi_lead, time_seris_multi_lead, failed_list = extract_ECG_hrv_dict_multilead(ecg_multi_lead, sr, sampleId, fea_types)

    # return
    data_hrv = {'labels':labels, 'fea_demo':fea_demo, 'fea_ECG_multi_lead':fea_ECG_multi_lead}
    data_time = {'labels':labels, 'fea_demo':fea_demo, 'time_seris_multi_lead':time_seris_multi_lead}
    return data_hrv, data_time, failed_list




#%% usage examle
if __name__ == '__main__':
    start_time = time.time()

    # sample ecg input
    num_ecg_lead = 3
    length_ecg = 500000 # number of points sampled from ecg 
    sr = 500 #sampling rate
    noise = 0.05 # add random noise to simulated ecg data
    heart_rate = 60
    sampleId = 'ID0001'
    
    # fea_types = ['time', 'freq', 'rsa', 'nonlin', 'ECG_Rate', 'ECG_Quality', 'ECG_Phase_Completion_Atrial', 'ECG_Phase_Completion_Ventricular']
    fea_types = ['time', 'freq', 'nonlin', 'ECG_Rate', 'ECG_Quality', 'ECG_Phase_Completion_Atrial', 'ECG_Phase_Completion_Ventricular'] # omit the rsa features because the neurokit2 problems
    
    
    # generate simulated ecg data
    duration = (length_ecg/sr)
    print(f'Generate {num_ecg_lead} lead ECG with duration {duration} seconds')
    ecg_multi_lead = [nk.ecg_simulate(length=length_ecg, noise=noise, heart_rate=heart_rate) for leadId in range(1, num_ecg_lead+1)]
    min_len = min([len(ecg_single_lead) for ecg_single_lead in ecg_multi_lead])
    ecg_multi_lead = [(leadId, ecg_single_lead[:min_len]) for leadId, ecg_single_lead in enumerate(ecg_multi_lead)]
    
    
    # plot ecg
    ecg_multi_lead_df = pd.DataFrame({f'Lead-{leadId}':ecg_single_lead[1] for leadId, ecg_single_lead in enumerate(ecg_multi_lead)})
    nk.signal_plot(ecg_multi_lead_df, subplots=True)
    
    # featrue extraction
    ecg_fea_multi_lead, time_seris_multi_lead, failed_list = extract_ECG_hrv_dict_multilead(ecg_multi_lead, sr, sampleId, fea_types)
    
    print(f'Finish feature extraction, time usage: {time.time()-start_time}')