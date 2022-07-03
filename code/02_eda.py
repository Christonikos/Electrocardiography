#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Exploratory Data Analysis based on the signal metadata. 

@author: Christos
"""

# =============================================================================
# IMPORT MODULES
# =============================================================================
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statannot import add_stat_annotation
import config as c
from utils import snake_case, load_the_cohort_class_info


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================


def collect_signal_metadata(patients, electrode, path):
    '''
    Return a concatenated dataframe  of signal metadata for a given electrode
    and a selected sub-cohort. The signal metadata are extracted
    @00_get_patient_info.py

    Parameters
    ----------
    patients : list
        A list of patients for which to pull data. It can be the full
        cohort or a sub-cohort (e.g: healthy control)
    electrode : string
        The lead for which to pull data (e.g 'i'). The full list of available
        leads can be found in the config file. 

    Returns
    -------
    The concatenated dataframe

    '''
    
    # collect the signal metadata 
    collector = []
    for idx, patient in enumerate(patients):
        # get the available records per patient
        records = [f for f in os.listdir(c.join(path.to_info(),patient)) if not f.startswith('.')]
        # for this stage of the analysis, use only the first record
        record =records[0]
    
        
        signal_metadata = pd.read_csv(c.join(path.to_info(),patient,
                                             record, 'signal_metadata',
                                             f'{patient}_{record}_signal_metadata.csv'),
                                      index_col=0).loc[electrode]      
        collector.append(signal_metadata)
    
    # construct the dataframe and index it by the patient name
    sub_cohort_dataframe = pd.concat(collector, axis=1)
    sub_cohort_dataframe.columns=patients
    
    return sub_cohort_dataframe.T    





def plot_eda(features_of_interest, class_1, class_2, cohort_classes, path):
    '''
    

    Parameters
    ----------
    features_of_interest : List
        e.g: ['channel_variance','mean_amplitude',
              'power_spectral_density_max']
    class_1 : String
        e.g: 'Healthy control'.
    class_2 : String
        e.g: Myocardial infarction.
    cohort_classes : Dict, created @collect_signal_metadata
        Contains the list of patients that correspond to each class
    

    Returns
    -------
    None.

    '''
    fig=plt.figure(dpi=100, facecolor='w', edgecolor='w')
    fig.set_size_inches(36,10)        
    counter = 0
    for idx, feature in enumerate(features_of_interest):
        for electrode in c.electrodes:
            counter=counter+1
            
            # pool data for a given electrode
            class_1_data = collect_signal_metadata(cohort_classes[class_1], electrode, path)
            class_2_data = collect_signal_metadata(cohort_classes[class_2], electrode, path)
            # aggregate data
            data = [class_1_data[feature].values, class_2_data[feature].values]
            # tranform into a dataframe
            df=pd.DataFrame(data).T
            df.columns=[class_1,class_2]
            
            # plotting
            plt.subplot(len(features_of_interest), len(c.electrodes), counter)
            # plot
            ax = sns.boxplot(data=df)
    
            add_stat_annotation(ax, data=df,box_pairs=[(class_1, class_2)],
                                               test='Mann-Whitney', text_format='star',
                                               loc='outside', verbose=0)
            if idx <2:
                plt.xticks([])
            plt.xticks(rotation = 45)
            sns.despine(trim=True)
            
            if counter in (1,16,31):
                if feature=='channel_variance':
                    label = 'mV^2'
                elif feature=='mean_amplitude':
                    label ='mV'
                elif feature=='power_spectral_density_max':
                    label='Hz'
                plt.ylabel(label, fontweight='bold', fontsize=14)
    
            if idx==0:
                plt.title(f'{electrode}', y=1.15, fontweight='bold', fontsize=14)
            
                
    plt.suptitle(f'{class_1} VS {class_2} \n Features: {features_of_interest}',
                 y=1.05, fontweight='bold', fontsize=14)
    fig.tight_layout()
    fname = c.join(path.to_images(),f'{snake_case(class_1)}_{snake_case(class_2)}.png')
    fig.savefig(fname, bbox_inches='tight',orientation='landscape')
    plt.show()


# %%        
# =============================================================================
# EXECUTE 
# =============================================================================

if __name__=='__main__':
    # call the path constructor
    path = c.FetchPaths(c.PROJECTS_PATH, c.PROJECT_NAME)
    cohort_classes = load_the_cohort_class_info(path)
    features_of_interest = ['channel_variance','mean_amplitude',
                            'power_spectral_density_max']
    
    for class_2 in c.classes:
        if class_2 =='Healthy control':
            continue
        print(f'{snake_case("Healthy control")}_{snake_case(class_2)}')
        plot_eda(features_of_interest, 'Healthy control', class_2, cohort_classes, path)



