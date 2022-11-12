
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thur June 17 17:35:05 2021

@author: chadyang
"""
import random
import numpy as np
import torch
from torchvision import transforms
from torchvision.transforms import functional as TF
from heartpy.filtering import filter_signal, remove_baseline_wander
# import librosa

from neurokit2 import signal_sanitize, ecg_clean, ecg_peaks, signal_rate, ecg_quality, ecg_delineate, ecg_phase
import pandas as pd


class NK2_ECG_Process(object):
    def __init__(self, sr=500, method="neurokit", concat_lead=False):
        self.sr = sr
        self.method = method
        self.concat_lead =  concat_lead

    def __call__(self, recording:list):
        ecg_signal_tfm = []
        for ecg_signal in recording:
            ecg_signal_tfm.append(self.__nk2_process__(ecg_signal))
        if self.concat_lead:
            ecg_signal_tfm = np.concatenate(ecg_signal_tfm)
        return ecg_signal_tfm

    def __repr__(self):
        return self.__class__.__name__

    def __nk2_process__(self, ecg_signal):
        ecg_signal = signal_sanitize(ecg_signal)
        ecg_cleaned = ecg_clean(ecg_signal, sampling_rate=self.sr, method=self.method)

        # instant_peaks, rpeaks, = ecg_peaks(ecg_cleaned=ecg_cleaned, sampling_rate=self.sr, method=self.method, correct_artifacts=True)
        # rate = signal_rate(rpeaks, sampling_rate=self.sr, desired_length=len(ecg_cleaned))/70
        # quality = ecg_quality(ecg_cleaned, rpeaks=None, sampling_rate=self.sr)
        # signals = pd.DataFrame({"ECG_Clean": ecg_cleaned, "ECG_Rate": rate, "ECG_Quality": quality})

        # delineate_signal, delineate_info = ecg_delineate(ecg_cleaned=ecg_cleaned, rpeaks=rpeaks, sampling_rate=self.sr)
        # cardiac_phase = ecg_phase(ecg_cleaned=ecg_cleaned, rpeaks=rpeaks, delineate_info=delineate_info)
        # cardiac_phase = cardiac_phase[["ECG_Phase_Completion_Atrial", "ECG_Phase_Completion_Ventricular"]]
        # cardiac_phase = cardiac_phase.fillna(-1)
        # signals = pd.concat([signals, cardiac_phase], axis=1)
        # return signals.values.T # 5*T
        return ecg_cleaned.reshape(1,-1) # (T,)

class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

class Zscore(object):
    def __call__(self, array):
        array = (array-array.mean(axis=0))/array.std(axis=0)
        array[np.isnan(array)] = 0
        return array
    def __repr__(self):
        return self.__class__.__name__

class FilterSignal(object):
    def __init__(self, cutoff=0.05, sr=500, order=2, filtertype="bandpass",return_top = False):
        self.sr = sr
        self.cutoff = cutoff
        self.order = order
        self.filtertype = filtertype
        self.return_top =  return_top
    def __call__(self, data:list):
        return filter_signal(data, sample_rate=self.sr, cutoff=self.cutoff, order=self.order, filtertype=self.filtertype).copy()
    def __repr__(self):
        return self.__class__.__name__


class RemoveBaselineWander(object):
    def __init__(self, sr=500, cutoff=0.05):
        self.sr = sr
        self.cutoff = cutoff
    def __call__(self, data:list):
        return remove_baseline_wander(data, sample_rate=self.sr, cutoff=self.cutoff).copy()
    def __repr__(self):
        return self.__class__.__name__

class BandPass(object):
    def __init__(self, sr=500, cutoff=0.05, filtertype="bandpass"):
        self.sr = sr
        self.cutoff = cutoff
        self.filtertype = filtertype
    def __call__(self, data:list):
        return filter_signal(data, sample_rate=self.sr, cutoff=self.cutoff, filtertype=self.filtertype).copy()
    def __repr__(self):
        return self.__class__.__name__

class MinMaxScaler(object):
    def __call__(self, array):
        array = array.T
        array = ((array-array.min(axis=0))/(array.max(axis=0)-array.min(axis=0))).T
        array[~np.isfinite(array)] = 0
        return array
    def __repr__(self):
        return self.__class__.__name__

class RandomLeadMask(object):
    def __init__(self, p=0.5):
        self.p = p 
    def __call__(self, tensor):
        rand_number = np.random.uniform(low=0, high=1, size=tensor.shape[1])
        tmp = torch.zeros_like(tensor)
        tmp[:,rand_number>self.p,:] = tensor[:,rand_number>self.p,:]
        return tmp
    def __repr__(self):
        return self.__class__.__name__

class RandomShuflleLead(object):
    def __init__(self, p=0.5):
        self.p = p
    def __call__(self, tensor):
        rand_number = np.random.uniform(low=0, high=1, size=1)[0]
        if rand_number<self.p:
            lead_idx = list(range(tensor.shape[1]))
            random.shuffle(lead_idx)
            tensor = tensor[:,lead_idx,:]
        return tensor
    def __repr__(self):
        return self.__class__.__name__


# class GetSpecTralGram(object):
#     def __init__(self, nfft, hop_len):
#         self.nfft = nfft
#         self.hop_len = hop_len
#     def __call__(self, array):
#         # array: lead*len
#         array_stft = []
#         for lead in array:
#             specgm = librosa.stft(lead, n_fft=self.nfft, hop_length=self.hop_len)
#             specgm = np.abs(specgm) 
#             array_stft.append(specgm)
#         array_stft = np.asarray(array_stft) # Lead*D*T
#         # array_stft = np.transpose(array_stft, (2,1,0)) # T*D*Lead
#         return array_stft.astype("float")
#     def __repr__(self):
#         return self.__class__.__name__

class AsTensor(object):
    def __call__(self, array):
        return torch.tensor(array)
    def __repr__(self):
        return self.__class__.__name__

