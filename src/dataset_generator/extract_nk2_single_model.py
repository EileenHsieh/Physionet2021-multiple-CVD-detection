#%%
from helper_code import *
import neurokit2 as nk2
from neurokit2.signal import signal_filter
from skimage.restoration import denoise_wavelet
from src.engine.utils import butterworth, functional
import pandas as pd
import numpy as np

from pathlib import Path

import coloredlogs, logging
coloredlogs.install()
logger = logging.getLogger(__name__)  


twelve_leads = ('I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6')


#%%
def peaks_ratio_fun(peaks, clean_signal, total_amps):
    vals = []
    ratios = []
    for i, p in enumerate(peaks):
        if not np.isnan(p):
            peak_val = clean_signal[p]
            vals.append(peak_val)
            if not total_amps[i]==-100: # total_amp is -100 if there is nan
                ratios.append(peak_val / total_amps[i])
    return functional(vals), functional(ratios)

def extract_fea(clean_signal, sr, sampleId, age, sex, label):
    _, rpeaks = nk2.ecg_peaks(clean_signal, sampling_rate=sr)
    signals, waves = nk2.ecg_delineate(clean_signal, rpeaks, sampling_rate=sr, method="peak")
    _, waves2 = nk2.ecg_delineate(clean_signal, rpeaks, sampling_rate=sr, method="cwt")
    
    signals['clean'] = clean_signal
    qrs_onset = waves2["ECG_R_Onsets"]
    qrs_offset = waves2["ECG_R_Offsets"]
    q_peaks = waves["ECG_Q_Peaks"]
    s_peaks = waves["ECG_S_Peaks"]
    t_peaks = waves["ECG_T_Peaks"]
    p_peaks = waves["ECG_P_Peaks"]
    r_peaks = rpeaks["ECG_R_Peaks"]
    total_amps = []
    for i in range(len(q_peaks)):
        if ((not np.isnan(p_peaks[i])) and (not np.isnan(q_peaks[i])) and 
            (not np.isnan(r_peaks[i])) and (not np.isnan(s_peaks[i])) and
            (not np.isnan(t_peaks[i]))):
            p,q,r,s,t = clean_signal[p_peaks[i]],clean_signal[q_peaks[i]],clean_signal[r_peaks[i]],clean_signal[s_peaks[i]],clean_signal[t_peaks[i]]
            maxval = max(p,q,r,s,t)
            minval = min(p,q,r,s,t)
            if minval<0:   total_amps.append(maxval-minval) 
            else:   total_amps.append(maxval)
        else: total_amps.append(-100)

    #-- rule for low QRS voltage: 300 microvolts as cutoff
    top5_rmeans = np.mean(clean_signal[rpeaks["ECG_R_Peaks"][:5]])
    
    #-- rule for prolong QT
    try:
        qt_durations = []
        qt_dur_ratio = []
        for i in range(len(q_peaks)):
            if (not np.isnan(t_peaks[i])) and (not np.isnan(q_peaks[i])):
                qt = t_peaks[i]-q_peaks[i]
                qt_durations.append(qt)
                if (not np.isnan(p_peaks[i])):
                    pt = t_peaks[i]-p_peaks[i]
                    qt_dur_ratio.append(qt/pt)
        qt_dur_fun = functional(qt_durations)
        qt_ratio_funs = functional(qt_dur_ratio)
    except:
        qt_dur_fun, qt_ratio_funs = np.zeros(7),np.zeros(7)

    #-- rule for bundle branch block
    try:
        qs_durations = []
        qs_dur_ratio = []
        for i in range(len(q_peaks)):
            if (not np.isnan(s_peaks[i])) and (not np.isnan(q_peaks[i])):
                qs = s_peaks[i]-q_peaks[i]
                qs_durations.append(qs)
                if (not np.isnan(p_peaks[i])) and (not np.isnan(t_peaks[i])):
                    pt = t_peaks[i]-p_peaks[i]
                    qs_dur_ratio.append(qs/pt)

        qs_dur_fun = functional(qs_durations)
        qs_ratio_funs = functional(qs_dur_ratio)
    except:
        qs_dur_fun, qs_ratio_funs = np.zeros(7),np.zeros(7)

    #-- rule for qwaves abnormal: qwave duration>0.2s
    try:
        q_durations = []
        q_dur_ratio = []
        for i in range(len(q_peaks)):
            if (not np.isnan(qrs_onset[i])) and (not np.isnan(q_peaks[i])):
                q = q_peaks[i] - qrs_onset[i]
                q_durations.append(q)
                if not(np.isnan(qrs_offset[i])):
                    qrs = qrs_offset[i] - qrs_onset[i]
                    q_dur_ratio.append(q/qrs)
        q_dur_fun = functional(q_durations)
        q_ratio_fun = functional(q_dur_ratio)
    except:
        q_dur_fun, q_ratio_fun = np.zeros(7),np.zeros(7)

    #-- extract features from waves
    objects = ['ECG_P_Peaks','ECG_Q_Peaks','ECG_S_Peaks','ECG_T_Peaks']
    for obj in objects:
        signals[obj] = np.where(signals[obj] == 1, signals['clean'], signals[obj])

    p_fun, pamp_ratio_fun = peaks_ratio_fun(p_peaks, clean_signal, total_amps)
    q_fun, qamp_ratio_fun = peaks_ratio_fun(q_peaks, clean_signal, total_amps)
    r_fun, ramp_ratio_fun = peaks_ratio_fun(r_peaks, clean_signal, total_amps)
    s_fun, samp_ratio_fun = peaks_ratio_fun(s_peaks, clean_signal, total_amps)
    t_fun, tamp_ratio_fun = peaks_ratio_fun(t_peaks, clean_signal, total_amps)
    fea = np.hstack([top5_rmeans, p_fun, pamp_ratio_fun, 
                    q_fun, qamp_ratio_fun, 
                    r_fun, ramp_ratio_fun, 
                    s_fun, samp_ratio_fun,
                    t_fun, tamp_ratio_fun,
                    qt_dur_fun, qt_ratio_funs, 
                    qs_dur_fun, qs_ratio_funs,
                    q_dur_fun, q_ratio_fun, age, sex])
    fea = np.nan_to_num(fea).astype('float32')
    sub_fea = {sampleId: fea}
    sub_label = {sampleId: label}
    return sub_fea, sub_label

def extract_data(header=None, recording=None, lead=None):
    # fea_types = config.param_feature.hrv_nk2.fea_types
    failed_list = []
    recording_lead = recording[twelve_leads.index(lead)]

    age = get_age(header)
    sex = 1 if get_sex(header)=='Male' else 0
    sr = get_frequency(header)

    #---------------------------------------------------
    # we don't need it as final test
    # sampleId = get_recording_id(header) 
    # sampleId = sampleId[:-4] if '.mat' in sampleId else sampleId 
    sampleId = 'test'
    
    # Remove the starting part and ending part: 0.02s
    window = int(0.2 * sr)
    recording_lead = recording_lead[window:-window]

    #---------------------------------------------------
    # we don't need it as final test
    # label = [1 if t in get_labels(header) else 0 for t in target_labels]
    label = []

    butter_signal = butterworth(recording_lead, 3, None, sr, 4) # lead II
    try:
        clean_signal = denoise_wavelet(butter_signal, sigma = 1, method='VisuShrink', mode='soft', rescale_sigma=True)
        sub_fea, sub_label = extract_fea(clean_signal, sr, sampleId, age, sex, label)
    except:
        try:
            clean_signal = signal_filter(signal=butter_signal, sampling_rate=sr, method="powerline", powerline=50)
            sub_fea, sub_label = extract_fea(clean_signal, sr, sampleId, age, sex, label)
        except:
            sub_fea = {sampleId: []}
            sub_label = {sampleId: label}
            failed_list.extend(sampleId)
    
    return pd.DataFrame.from_dict(sub_fea, orient='index'), sub_label, failed_list


