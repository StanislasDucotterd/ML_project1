# -*- coding: utf-8 -*-
"""Function used to compute the loss."""
import numpy as np
from helpers import sigmoid

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

def logistic_loss(y, tx, w):
    """compute the cost by negative log likelihood."""
    
    eps = 1e-5
    n_sample = len(y)
    log_likelihood = np.dot(y.T, np.log(sigmoid(np.dot(tx, w))+eps)) + np.dot((1 - y).T, np.log(1 - sigmoid(np.dot(tx, w))+eps))
    return np.squeeze(-log_likelihood)

def reg_logistic_loss(y, tx, w, lambda_):
    """compute the cost by regularization and negative log likelihood."""
    
    return logistic_loss(y, tx, w) + np.squeeze(lambda_*np.dot(w.T, w))