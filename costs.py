# -*- coding: utf-8 -*-
"""Function used to compute the loss."""
import numpy as np
from helpers import *

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
    
    y_pred = sigmoid(np.dot(tx, w))
    log_likelihood = np.dot(y.T, np.log(y_pred)) + np.dot((1 - y).T, np.log(1 - y_pred))
    return np.squeeze(-log_likelihood)

def reg_logistic_loss(y, tx, w, lambda_):
    """compute the cost by regularization and negative log likelihood."""
    
    return logistic_loss(y, tx, w) + np.squeeze(lambda_*np.dot(w.T, w))

def prediction_accuracy(y, y_pred):
    """compute the proportion of correct prediction"""
    
    classifier = lambda t: 1.0 if (t > 0.5) else 0.0
    classifier = np.vectorize(classifier)
    y_pred = classifier(y_pred)
    error = abs(y - y_pred)
    accuracy = 1 - sum(error) / len(error)
    return accuracy