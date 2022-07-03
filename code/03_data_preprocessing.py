#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: Christos
"""

# =============================================================================
# IMPORT MODULES
# =============================================================================

import os
from sklearn.preprocessing import StandardScaler
from sklearnex import patch_sklearn
patch_sklearn()
import numpy as np
import pandas as pd
import itertools
from scipy.ndimage import gaussian_filter1d
from mne.parallel import parallel_func
import wfdb
import config as c


def collect_recordings(patient, path):
    '''
    Return the recording for a given patient and all electrodes. 

    Parameters
    ----------
    patient : string
        e.g 'patient001'
    path : Class
        The path constructor

    Returns
    ----------
    collector: Dict
        Keys are records, values are the signal as numpy array for all elecs
    '''
    
    # get the available records per patient
    records = [f for f in os.listdir(c.join(path.to_info(),patient)) if not f.startswith('.')]
    curr_patient = c.join(path.to_data_raw(), patient)
    
    collector = {}
    for record in records:
        # read the record
        info = wfdb.rdrecord(c.join(curr_patient, record))
        # get the data from all leads
        data = info.p_signal
        # gather data for all records
        collector[record] = data        
    
    return collector


def preprocess_signal(patient, path, collector):
    '''
    The following steps are applied to the signal coming from a 
    given recording:
        1. Smoothing with a Gaussian kernel (width=10ms)
        2. Scaling (z-tranformation) of the time series

    '''
    
    scale= StandardScaler()    
    for record in collector.keys():
        data = collector[record] 

        width_sec = 0.01 # Gaussian-kernal width in [sec]
        sr = 1000
        for ch in range(data.shape[1]): # Loop over channels
            time_series = data[:, ch]
            data[:, ch] = gaussian_filter1d(time_series, width_sec*sr)
  
        
        # standardize the data (z-tranform)
        scaled_data = scale.fit_transform(data)
        
        # save the scaled reording per segment in a separate directory 
        # in the preprocessed folder
        path2data=c.join(path.to_data_preprocessed(), patient, record)
        if not c.exists(path2data):
            c.make(path2data)
        fname =  c.join(path2data, f'{patient}_{record}.npy')
        np.save(fname, scaled_data)
        
def main(patient):
    '''
    The main function that loads and preprocesses the 
    data for all recordings of a given patient.
    '''
    # return the data for all records     
    collector = collect_recordings(patient, path) 
    # preprocess and save the data
    preprocess_signal(patient, path, collector)
    

# %%        
# =============================================================================
# EXECUTE IN PARALLEL (For all patients)
# =============================================================================

if __name__ == "__main__":
    # call the path constructor
    path = c.FetchPaths(c.PROJECTS_PATH, c.PROJECT_NAME)
    # available patients
    patient_list = pd.read_csv(
        c.join(
            path.to_info(),
            'patients.tsv'),
        header=None).values.tolist()
    # unpack the list of lists
    patients = list(itertools.chain(*patient_list))
    # parallelize the main function
    parallel, run_func, _ = parallel_func(main,n_jobs=c.n_jobs)
    # run for all patients
    parallel(run_func(patient) for patient in patients)
    
    
        
        
