#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module contains frequently occuring and general-use functions.

@author: Christos
"""

# =============================================================================
# IMPORT MODULES
# =============================================================================
from re import sub
import pickle
import config as c

# =============================================================================
# FUNCTIONS
# =============================================================================
def snake_case(s):
    '''
    Given a string, transform it to a snake case equivalent
    E.g: LoVe clImbing --> love_climbing    
    '''
    return '_'.join(
        sub('([A-Z][a-z]+)', r' \1',
        sub('([A-Z]+)', r' \1',
        s.replace('-', ' '))).split()).lower()

def load_the_cohort_class_info(path):
    '''
    Load the dictionary that maps patients to condition

    Parameters
    ----------
    path : Class
        The path constructor.

    Returns
    -------
    cohort_classes : Dictionary
        Maps patients to condition.

    '''
    fname = c.join(path.to_info(),'cohort_classes.pickle') 
    with open(fname, 'rb') as handle:
        cohort_classes = pickle.load(handle)

    return cohort_classes
    