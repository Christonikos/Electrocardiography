#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Scope: Given the metadata from each patient (created with @00_get_patient_info)
extract statistical properties at the cohort level (age, different diagnosis,
                                                    etc.)

@author: Christos
"""
# =============================================================================
# IMPORT MODULES
# =============================================================================
import os
import itertools
import pandas as pd
import pickle
import config as c


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================
# convert dates from object to pandas.datetime
def dt_inplace(df):
    '''
    Automatically detect and convert (in place!) each
    dataframe column of datatype 'object' to a datetime just
    when ALL of its non-NaN values can be successfully parsed
    by pd.to_datetime().  Also returns a ref. to df for
    convenient use in an expression.
    '''
    from pandas.errors import ParserError
    for col in df.columns[df.dtypes=='object']: #don't cnvt num
        try:
            df[col]=pd.to_datetime(df[col])
        except (ParserError,ValueError): #Can't cnvrt some
            pass # ...so leave whole column as-is unconverted
    return df

# read patient metadata and build the cohort dataframe
def build_cohort_dataframe(path, patients):
    '''
    Collect the header metadata from all patients, and construct 
    a cohort dataframe. This dataframe is then stored into the info 

    Parameters
    ----------
    path : Class
        The path constructor.
    patients : List
        The sorted list of patients (e.g: patient001,...).

    Returns
    -------
    cohort_dataframe : Pandas Dataframe
        The collective dataframe build from the header metadata of each 
        patient.

    * ----------------------
    !!! **Important** !!! 
    * ----------------------
    In this version, only the first of the available records is utilized. 
    I will revisit this issue and make a final decision in the future. 
    I chose this to continue with the analysis.     
    
    '''
    collector = []
    for idx, patient in enumerate(patients):
        # get the available records per patient
        records = [f for f in os.listdir(c.join(path.to_info(),patient)) if not f.startswith('.')]
        # for this stage of the analysis, use only the first record
        record =records[0]
    
        
        header_metadata = pd.read_csv(c.join(path.to_info(),patient,
                                             record, 'patient_metadata',
                                             f'{patient}_{record}_header_metadata.csv'),
                                      index_col=0)  
        if idx ==0:
            features = header_metadata.feature.values.tolist()
        
        collector.append(header_metadata.value.values.tolist())
        
    # construct the dataframe and index it by the patient name
    cohort_dataframe = pd.DataFrame(collector, columns=features)
    cohort_dataframe=cohort_dataframe.set_axis(patients)
    
    
    #* ----------------------#
    # PREPROCESS THE DATAFRAME
    #* ----------------------#
    # convert automatically all available dates from object to string
    cohort_dataframe = dt_inplace(cohort_dataframe)
    # covert features to numeric
    num_features = ['age','Number of coronary vessels involved']
    for f in num_features:
        cohort_dataframe[f] = pd.to_numeric(cohort_dataframe[f], errors='coerce')
    
    # drop features with empty values
    empty_features = ['Hemodynamics','Diagnose']
    cohort_dataframe.drop(columns = empty_features, inplace = True)    

    
    #* ----------------------#
    # OUTPUT
    #* ----------------------#    
    # save the dataframe in the info directory
    fname = c.join(path.to_info(),'cohort_metadata.tsv')
    cohort_dataframe.to_csv(fname)
    
    
    cohort_dataframe.rename(columns={'Reason for admission': 'admission'}, inplace=True)

    # see the different reasons for admission
    cohort_classes = cohort_dataframe['admission'].unique().tolist()
    # save a dictionary indexed by the different classes that provides 
    # a list of the patients that correspond to each one
    class_collector = {}
    for class_ in cohort_classes:
        if str(class_)=='nan':
            continue
        class_collector[class_]=\
            cohort_dataframe[cohort_dataframe.admission==class_].index.tolist()
    # store as a pickle file        
    class_fname = c.join(path.to_info(),'cohort_classes.pickle')   
    with open(class_fname, 'wb') as handle:
        pickle.dump(class_collector, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    

# =============================================================================
# EXECUTE
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
    
    # run the function
    build_cohort_dataframe(path, patients)

