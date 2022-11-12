import sys, json
sys.path.append("../../")
from helper_code import *
from collections import defaultdict
from glob import glob
import pandas as pd
import json
import os 

def hea2dict(header):
    lines = header.split('\n')
    hea_dict = {}
    label_sub = defaultdict(list) # dict-> label: subject
    
    # hardware related
    demographic = lines[0].split(' ') 
    hea_dict['ID'] = demographic[0].split(".")[0]
    hea_dict['num_lead'] = demographic[1]
    hea_dict['sample_freq'] = demographic[2]
    hea_dict['num_samples'] = demographic[3]
    hea_dict['date'] = demographic[4]
    hea_dict['time'] = demographic[5]
    
    # demographic related
    for line in lines:
        if line.startswith("#Dx"): # disease
            d_seq = line.split(" ")[-1].split(",")
            for d in d_seq: 
                label_sub[d].append(hea_dict['ID'])
        elif line.startswith("#"):
            cat = line.split("#")[-1].split(":")[0]
            hea_dict[cat] = line.split(" ")[-1]
    return hea_dict, label_sub

def parse_meta(data_directory):
    header_files, recording_files = find_challenge_files(data_directory)
    num_recordings = len(recording_files)
    
    all_meta_df = pd.DataFrame()
    for i in range(num_recordings):
        print(f'    {i+1}/{num_recordings}...', end="\r")

        # Load header and recording.
        header = load_header(header_files[i])
        hea_dict, label_sub = hea2dict(header)
        meta_df = pd.DataFrame(hea_dict, index=[i])
        all_meta_df = pd.concat([all_meta_df,meta_df], axis=0)
    
    return all_meta_df

if __name__ == '__main__':
    data_directory = '../../datasets/raw/WFDB_StPetersburg'
    all_meta_df = parse_meta(data_directory)