# -*- coding: utf-8 -*-
"""Implementation of the 6 functions"""
import numpy as np
from costs import compute_mse
from compute_gradient import *
from helpers import *


def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    """Gradient descent algorithm."""
    
    w = initial_w
    for n_iter in range(max_iters):

        # compute gradient and loss
        gradient = compute_gradient(y, tx, w)[0]

        # update w by gradient
        w = w - gamma*gradient
    loss = compute_mse(y, tx, w)  
    
    return w, loss

def least_squares_SGD(y, tx, initial_w, batch_size, max_iters, gamma):
    """Stochastic gradient descent."""

    w = initial_w
    
    for n_iter in range(max_iters):
        for y_batch, tx_batch in batch_iter(y, tx, batch_size=batch_size, num_batches=1):
            
            # compute gradient
            gradient = compute_gradient(y_batch, tx_batch, w)
            
            # update w through the stochastic gradient update
            w = w - gamma * gradient
            
    # calculate loss
    loss = compute_mse(y, tx, w)

    return w, loss

def least_squares(y, tx): 
    """calculate the least squares solution."""
    
    a = tx.T.dot(tx)
    b = tx.T.dot(y)
    w = np.linalg.solve(a, b)
    loss = compute_mse(y, tx, w)
    return w, loss

def ridge_regression(y, tx, lambda_):
    """implement ridge regression."""
    
    n_sample = len(y)
    I = 2*n_sample*lambda_*np.identity(np.shape(tx)[1])
    a = np.dot(tx.T, tx) + I
    b = np.dot(tx.T, y)
    w = np.linalg.solve(a, b)
    loss = compute_mse(y, tx, w)
    return w, loss

def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """Logistic regression using gradient descent or SGD"""
    
    w = initial_w
    
    for n_iter in range(max_iters):
        link = sigmoid(np.dot(tx, w))
        
        loss = y - link
        gradient = np.dot(tx.T, error)
        w += gradient * gamma
        
    return w, loss
   
        
    