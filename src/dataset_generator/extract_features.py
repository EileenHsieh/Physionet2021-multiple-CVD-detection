import os
import joblib
from helper_code import *
from lib.signal_process.neurokit2.extract_neurokit2 import extract_neurokit2_data

def _extract_hrv(data_directory, ):
    FEA_TYPES = ['time', 'freq', 'nonlin', 'ECG_Rate', 'ECG_Quality', 
                    'ECG_Phase_Completion_Atrial', 'ECG_Phase_Completion_Ventricular']

    header_files, recording_files = find_challenge_files(data_directory)
    num_recordings = len(recording_files)
    for i in range(num_recordings):
        header = load_header(header_files[i])
        recording = load_recording(recording_files[i])
        sampleId = header.split('.')[0]
        data_hrv, _, _ = extract_neurokit2_data(header, recording, sampleId, twelve_leads, FEA_TYPES)
        if os.path.exists(header_files.replace("raw", "processed")):    continue
        joblib.dump(data_hrv, header_files.replace("raw", "processed"))

def _extract_one_sample_hrv(header, recording):
    FEA_TYPES = ['time', 'freq', 'nonlin', 'ECG_Rate', 'ECG_Quality', 
                    'ECG_Phase_Completion_Atrial', 'ECG_Phase_Completion_Ventricular']

    sample_id = header.split('.')[0]
    data_hrv, _, _ = extract_neurokit2_data(header, recording, sample_id, twelve_leads, FEA_TYPES)

    return data_hrv, sample_id


