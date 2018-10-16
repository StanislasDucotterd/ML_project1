# -*- coding: utf-8 -*-
"""Function used to compute the loss."""
import numpy as np

def compute_mse(y, tx, w):
    """Calculate the loss using mse"""
    
    n_sample = len(y)
    e = y - np.dot(tx, w)
    return (0.5/n_sample)*np.dot(e.T,e)

def compute_mae(y, tx, w):
    """Calculate the loss using mae"""
    
    n_sample = len(y)
    e = y - np.dot(tx, w)
    return np.mean(np.abs(e))