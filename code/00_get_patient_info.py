#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Scope: This script reads the patient info and logs information in the log file
(i.e: number of patients). Additionally, it loads the metadata for each patient
and stores this info in different directories.

This allows us to build a processing pipeline that runs at the patient level
and can be fully parallelized in the future.

!!!! This function runs in parallel and uses all threads. To change the 
number of threads, see the variable "n_jobs" @config.py

@author: Christos
"""

# =============================================================================
# IMPORT MODULES
# =============================================================================
import os
import pandas as pd
import numpy as np
from scipy import signal
from mne.parallel import parallel_func
import wfdb
import config as c


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================
def list_patients():
    '''

    Based on the 'raw' directory, list the number of patients
    and return their code names as a list.

    Parameters
    ----------
    None

    Returns
    -------
    patients : List
        The sorted list of patients (e.g: patient001,...).

    '''

    # call the path constructor
    path = c.FetchPaths(c.PROJECTS_PATH, c.PROJECT_NAME)
    # list and log the number of files in the raw dir
    files = os.listdir(path.to_data_raw())
    # keep only the directories
    patients = sorted([file for file in files if os.path.isdir(
        c.join(path.to_data_raw(), file))])
    c.logging.info(
        f'Data from {len(patients)} patients available in this dataset')
    # save the patients list as a .csv
    fname = c.join(path.to_info(),'patients.tsv')
    np.savetxt(fname, patients, delimiter=",", fmt='%s')

    

    return patients


def tranform_metadata_to_dataframe(metadata, patient, record, path):
    '''
    Transform the metadata extracted from the header into a pandas dataframe
    and store this information in the info derivative. Each metadata file
    corresponds to a record of a given patient.

    Additionally, strip the whitespace of each entry and set the empty
    to the string "empty_value"

    Parameters
    ----------
    metadata : List
        The original metadata file extracted from the header for a given
        patient and record.
    record : String
        The corresponfing record of the current patient.
    patient: String
        The current patient.
    path : Class
        The path constructor.

    Returns
    -------
    None
    '''
    # get the features (e.g age)
    columns = [metadata[i].split(':')[0].strip() for i in range(len(metadata))]
    # get and preprocess the values
    values = ["no_entry" if metadata[i].split(':')[1].strip(
    ) == '' else metadata[i].split(':')[1].strip() for i in range(len(metadata))]
    # transform into a dataframe
    metadata_df = pd.DataFrame(list(zip(columns, values)), columns=['feature', 'value'])
    # store into the info derivative dir
    path2metadata = c.join(path.to_info(), patient, record, 'patient_metadata')
    if not c.exists(path2metadata):
        c.make(path2metadata)
    fname = c.join(path2metadata, f'{patient}_{record}_header_metadata.csv')
    metadata_df.to_csv(fname)


def extract_signal_metadata(data, patient, record, info, path):
    '''
    Extract metadata from the recorded data and for all 15 leads for a
    given patient and record. The metadata correspond to one value per channel
    for the whole duration of the record.

    These include descriptive metrics such as the following:
        1. The variance of each channel.
        2. The mean amplitude of each channel.
        3. The median amplirude of each channel.
        4. The mean value of the 1st derivative (how fast, on average
                                                 the data changes)
        5. The median value of the 1st derivative
        6. The peak of the power-spectral density for each lead

    Parameters
    ----------
    data : Numpy Array (duration X #channels)
        The recorded data for all leads.
    record : String
        The corresponfing record of the current patient.
    patient: String
        The current patient.
    info :

        DESCRIPTION.
    path : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    '''

    # get the channel names
    channel_names = info.sig_name
    # EXTRACT METRICS
    # 1. Variance
    channel_variance = np.var(data, axis=0)
    # 2. Mean amplitude
    mean_amplitude = np.mean(data, axis=0)
    # 3. Median amplitude
    median_amplitude = np.median(data, axis=0)
    # 4. Mean value of the derivative (1st)
    mean_der_value = np.mean(np.gradient(data, 1, axis=0), axis=0)
    # 5. Median value of the derivative
    median_der_value = np.median(np.gradient(data, 1, axis=0), axis=0)
    # 6. Peak of the power-spectrum (in Hz)
    peaks = [np.max(signal.welch(data[:, sig], 1000, 'flattop', 1024, scaling='spectrum')[
                    1] * 1e3) for sig in range(0, data.shape[1])]

    columns = ['channel_variance', 'mean_amplitude', 'median_amplitude',
               'mean_derivative_value', 'median_derivative_value',
               'power_spectral_density_max']

    signal_metadata = pd.DataFrame(list(zip(channel_variance, mean_amplitude,
                               median_amplitude, mean_der_value,
                               median_der_value, peaks)), index=channel_names,
                      columns=columns)

    # store into the info derivative dir
    path2metadata = c.join(path.to_info(), patient, record, 'signal_metadata')

    if not c.exists(path2metadata):
        c.make(path2metadata)
    fname = c.join(path2metadata, f'{patient}_{record}_signal_metadata.csv')
    signal_metadata.to_csv(fname)


# =============================================================================
# MAIN FUNCTION (WRAPPER))
# =============================================================================

def extract_patient_and_signal_info(patient):
    '''
    The main function of this analysis stage. This function calls all the utility
    functions defined above and performs the following steps:
        1. Identify the number of different records per patient
        2. Load the data from all leads and the header metadata
        3. Store the header metadata in the info directory
        4. Extract descriptive measures for each lead and store this info into
        the info directory for post-processing (these include the power spectral
                                                density peak for each lead,
                                                the variance of each channel and
                                                others.)

    Parameters
    ----------
    patient : string
        The current patient (e.g: patient001, constructed with @list_patients)

    Returns
    -------
    None.

    '''

    # call the path constructor
    path = c.FetchPaths(c.PROJECTS_PATH, c.PROJECT_NAME)
    # construct the path for the given
    curr_patient = c.join(path.to_data_raw(), patient)
    # now, see how many segments (recordings) each patient has
    records = os.listdir(curr_patient)
    # each recording is identified by a prefix, to get the number
    # of records, we need the number of distinct prefices
    # isolate the unique .dat files
    unq_dat_files = list(filter(lambda x: '.dat' in x, records))
    # from this, extract the unq record names
    record_names = [
        unq_dat_files[i].split('.dat')[0] for i in range(
            len(unq_dat_files))]
    # now, loop over the records and read the data and the metadata
    for record in record_names:
        # read the record
        info = wfdb.rdrecord(c.join(curr_patient, record))
        # get the metadata
        metadata = info.comments
        # convert metadata to df and store in the info directory
        tranform_metadata_to_dataframe(metadata, patient, record, path)

        # get the data from all leads
        data = info.p_signal
        # extract descriptive metrics for all leads and store into a dataframe
        extract_signal_metadata(data, patient, record, info, path)


# %%

# =============================================================================
# EXECUTE IN PARALLEL (For all patients)
# =============================================================================

if __name__ == "__main__":
    # get the number of patients 
    patient_list = list_patients()
    # parallelize the main function
    parallel, run_func, _ = parallel_func(extract_patient_and_signal_info,
                                          n_jobs=c.n_jobs)
    # run for all patients
    parallel(run_func(patient) for patient in patient_list)
