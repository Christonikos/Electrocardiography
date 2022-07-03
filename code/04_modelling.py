#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: Christos
"""

# =============================================================================
# IMPORT MODULES
# =============================================================================

import os
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.model_selection import cross_val_score
from sklearnex import patch_sklearn
patch_sklearn()
from lightgbm import LGBMClassifier
import numpy as np
import matplotlib.pyplot as plt
import time
import seaborn as sns
import pickle
import config as c
from utils import snake_case, load_the_cohort_class_info


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def collect_data(patient_list, path, electrode):
    '''
    Given a patient list (e.g: patients that belong in the 'healthy_control')
    population, and for a given electrode (e.g: "avl"), load the PREPROCESSED
    data to be used for time-series classification. 
    
    !!! At this stage, the function calls only data from the first record 
    of each patient. 

    Parameters
    ----------
    patient_list : List
        A list of patients belonging to a given population. You can get this 
        list by using the "load_the_cohort_class_info" function located 
        @ utils.py
    path : Class
        The path constructor.
    electrode : String
        A selected electrode.

    Returns
    -------
    collector : List
        Contains preprocessed data of the first recording of each patient.

    '''
    
    elec_index = c.electrodes.index(electrode)
    
    collector = []
    for patient in patient_list:
        # get the available records per patient
        records = [f for f in os.listdir(c.join(path.to_info(),patient)) if not f.startswith('.')]
        record = records[0]
        fname=c.join(path.to_data_preprocessed(), patient, record, f'{patient}_{record}.npy')
        data = np.load(fname)[:,elec_index]    
        collector.append(data)
        
    return collector

def make_sklearn_compatible(class_1, class_2):
    '''
    Given two arrays that correspond to two selected classes, transform the 
    data into the standard SKLEARN format. 
    
    The lists are created @collect_data

    Parameters
    ----------
    class_1 : Array
        E.g: Data from the "healthy control" population.
    class_2 : Array
        E.g: Data from the "heart failure" population.

    Returns
    -------
    X : Array
        The feature matrix (in this case, all the samples corresponding to both 
                            classes of interest).
    y : 1D Array (len = LEN(SAMPLES(class 1 & class2)))
        The target array is the concatenation of the samples belonging to each
        class. Since we framed this as a univariate, binary classification problem,
        this is a vector containg ones and zeros

    '''
    # Keep the seed constant (42)
    np.random.seed(42)
    # Make the data compatible with sklearn
    X = np.concatenate((class_1, class_2))
    y = np.concatenate((np.ones(class_1.shape[0]), np.zeros(class_2.shape[0])))  

    return X,y

def plot_target_distribution(y, path, class_1, class_2):
    '''
    Plots the distribution of the target values. This is used to select the
    appropriate evaluation metric.

    '''
    fig=plt.figure(dpi=100, facecolor='w', edgecolor='w')
    #fig.set_size_inches(20, 10)
    sns.displot(y)
    plt.ylabel('# counts ')
    plt.xlabel('target')
    plt.xticks([0,1])
    plt.title(f'Distribution of target values. \n {class_1} VS {class_2}',
              style='oblique', fontweight='bold', y=1.05)

    
    
    fig.savefig(fname=os.path.join(path.to_images(),
                                   f'target_distribution_{snake_case(class_1)}_vs_{snake_case(class_2)}.png'),
                bbox_inches='tight')
    plt.show()
    


def cross_val_hyperparam_tuning(RUN_RANDOMSEARCH,model, X,y, path):
    '''
    Perform Randomized search on hyper parameters. 

    Parameters
    ----------
    RUN_RANDOMSEARCH : Bool
        Whether to run the randomized search or load previous best params.
    X : DF
        Feature Matrix.
    y : Array
        Target.
    path : Class
        Used to save the best params if RUN_RANDOMSEARCH=False.

    Returns
    -------
    best_params : Dict
        Contains the best parameters of the Randomized search on hyper parameters.
        If RUN_RANDOMSEARCH == True, the function returns params calculated online, 
        otherwise, it loads them from a previous search. 

    '''

  

    if RUN_RANDOMSEARCH:
        print('Hyper-param tuning')        
        start_time = time.time()
        gkf = StratifiedKFold(n_splits=5, shuffle=True,
                              random_state=c.random_state).split(X=X, y=y)
        rsearch = RandomizedSearchCV(model, 
                                     param_distributions=c.param_test,
                                     cv=gkf, n_jobs=-1)
        lgb_model_random = rsearch.fit(X=X, y=np.ravel(y,order='C'))
        
        best_params = lgb_model_random.best_params_
        best_params["objective"] = "binary"
        
        print(lgb_model_random.best_params_, lgb_model_random.best_score_)
        print("--- %s seconds ---" % (time.time() - start_time))
        # save the parameters
        if not c.exists(path.to_params()):
            c.make(path.to_params())
        with open(c.join(path.to_params() ,'best_params.pkl'), 'wb') as f:
            pickle.dump(lgb_model_random.best_params_, f)
        
        best_params = lgb_model_random.best_params_ 
    else:
        with open(c.join(path.to_params() ,'best_params.pkl'), 'rb') as f:
            best_params = pickle.load(f)

        
    
    return best_params

############################
# MAIN MODELLING FUNCTION ##
############################

def modeling(class_1, class_2, electrode, path):
    '''
    1. Given a set of two classes and a selected electrode, perform 
    classification using the LightGBM classifier. 
    
    2. Save the results in the results derivative
    

    Parameters
    ----------
    class_1 : String
        e.g: Healthy control'.
    class_2 : String
        e.g: Myocardial infarction'.
    electrode : String
        e.g: 'ii'.
    path : Class
        The path constructor.

    Returns
    -------
    Numpy Array
        The mean AUC across 5-stratified folds.

    '''

    # load data for each class
    class_1_data = collect_data(cohort_classes[class_1], path, electrode)
    # concatenate all data per channel type and tranform into a np array
    class_1_concat = np.concatenate([np.array(i) for i in class_1_data])
    del class_1_data
    class_2_data = collect_data(cohort_classes[class_2], path, electrode)
    # concatenate all data per channel type and tranform into a np array
    class_2_concat = np.concatenate([np.array(i) for i in class_2_data])    
    del class_2_data
    # make data sklearn compatible
    X, y = make_sklearn_compatible(class_1_concat, class_2_concat)
    # reshape given that we work with 1D data
    X = X.reshape(-1, 1)
    y = y.reshape(-1, 1)
    del class_1_concat, class_2_concat

    ########################        
    # Set up the model
    ########################
    # CLASSIFICATION USING LightGBM
    clf = LGBMClassifier(
        boosting_type="gbdt", objective="binary", learning_rate=0.01,
        metric="auc")
    #######################################        
    # Hyperparam tuning using randomsearch
    #######################################        
    best_params = cross_val_hyperparam_tuning(RUN_RANDOMSEARCH, clf,
                                          X,y, path)
    
    skf = StratifiedKFold(n_splits=5, shuffle=True,random_state=c.random_state)
    model = LGBMClassifier(**best_params)
    
    # Gather the scores across the folds
    scores = cross_val_score(model, X, y=np.ravel(y,order='C'),
                              cv=skf, scoring='roc_auc', n_jobs=-1) 
    results = np.mean(scores)
    
    
    # construct the fname
    class_1 = snake_case(class_1)
    class_2 = snake_case(class_2)
    class_name = class_1+'_vs_'+class_2

    
    path2results = c.join(path.to_results(),class_name,
                          f'electrode_{electrode}')
    if not c.exists(path2results):
        c.make(path2results)
    
    fname = c.join(path2results, f'{class_name}_{electrode}.npy')
    np.save(fname, results)
    
    
    # log and print
    info = f'{class_name}_{electrode}. AUC: {results}'
    c.logging.info(info)
    print(info)
    
    # return the mean AUC across folds
    return results
# %%
# =============================================================================
# EXECUTE AND RUN FOR ALL POSSIBLE CLASSES AND GIVEN ELECTRODES
# =============================================================================
# * This step can be parallelized at a selected level (e.g classes or elecs)
# I did not do it because I ran the analysis at my laptop.

if __name__=='__main__':
    
    # set to false in order to not relaunch the RandomSearch and
    # use the best hyperparameters that already calculated
    RUN_RANDOMSEARCH = True
    # call the path constructor
    path = c.FetchPaths(c.PROJECTS_PATH, c.PROJECT_NAME)
    # get the map of patients --> condition
    cohort_classes = load_the_cohort_class_info(path)
    
    
    class_1 = 'Healthy control'
    # Loop through classes
    for class_2 in c.classes:
        if class_2==class_1:
            continue
        # Now loop through electrodes
        for electrode in c.electrodes:
            modeling(class_1, class_2, electrode, path)
            
        
    
    
    