#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot the results of the classification analysis as a heatmap. 
Results saved @the "images" dir.

@author: Christos
"""
# =============================================================================
# IMPORT MODULES
# =============================================================================

import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import config as c
from utils import snake_case



# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================
def load_results(class_name, electrode, path):
    '''
    Given the class name (e.g: healthy_control_vs_heart_failure)
    and a given electrode (e.g: 'i'), collect the results of the
    classification analysis performed @04_modelling.py

    Parameters
    ----------
    class_name : String
        e.g: healthy_control_vs_heart_failure.
    electrode : String
        e.g: "i".
    path : Class
        The path constructor.

    Returns
    -------
    Numpy Array
        The mean AUC across 5 stratified k-folds rounded at the second decimal.

    '''    
    path2results = c.join(path.to_results(),class_name,
                      f'electrode_{electrode}')

    fname = c.join(path2results, f'{class_name}_{electrode}.npy')
    results = np.load(fname)
    
    return np.round(results,2)


# %%
# =============================================================================
# EXECUTE AND PLOT THE HEATMAP WITH THE CLASSIFICATION RESULTS
# =============================================================================

if __name__=='__main__':
    # call the path constructor
    path = c.FetchPaths(c.PROJECTS_PATH, c.PROJECT_NAME)
    
    collector={}
    class_1 = 'Healthy control'
    # Loop through classes
    for class_2 in c.classes:
        # construct the fname
        class_1 = snake_case(class_1)
        class_2 = snake_case(class_2)
        class_name = class_1+'_vs_'+class_2
        if class_name=='healthy_control_vs_healthy_control':
            continue
        collector[class_name]={}
        # Now loop through electrodes
        for electrode in c.electrodes:
            collector[class_name][electrode] = load_results(class_name, electrode, path)
            
    scores = pd.DataFrame(collector).T
    
    # sort by best overall prediction
    scores=scores.reindex(scores.mean(axis=1).sort_values(ascending=False,
                                                          na_position='first').index, axis=0)
    

    fig=plt.figure(dpi=100, facecolor='w', edgecolor='w')
    fig.set_size_inches(18.5, 10.5)

    colormap =sns.color_palette("vlag", as_cmap=True)
    sns.heatmap(scores, cmap = colormap, annot=True, 
                 cbar_kws={'label': 'AUC', })

    plt.tick_params(axis='both', which='major', 
                    labelsize=10, labelbottom = False, 
                    bottom=False, top = False, labeltop=True,)
    plt.title('Inference based on time-series alone', y=1.05)
    fig.savefig(fname=os.path.join(path.to_images(),'auc_results_time_series_only.png'),
                bbox_inches='tight')
    plt.show()
    
