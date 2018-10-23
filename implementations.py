# -*- coding: utf-8 -*-
"""Implementation of the 6 functions"""
import numpy as np
from costs import *
from compute_gradient import *
from helpers import *


def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    """Gradient descent algorithm."""
    
    w = initial_w
    losses = []
    threshold = 1e-8
    
    for n_iter in range(max_iters):

        # compute gradient and loss
        gradient = compute_gradient(y, tx, w)[0]

        # update w by gradient
        w = w - gamma*gradient
        loss = compute_mse(y, tx, w)  
        losses.append(loss)
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break
            
    loss = losses[-1]
      
    return w, loss

def least_squares_SGD(y, tx, initial_w, batch_size, max_iters, gamma):
    """Stochastic gradient descent."""

    w = initial_w
    
    for y_batch, tx_batch in batch_iter(y, tx, batch_size=batch_size, num_batches=max_iters):
            
            # compute gradient
        gradient = compute_gradient(y_batch, tx_batch, w)[0]
            
            # update w through the stochastic gradient update
        w = w - gamma * gradient
            
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

def logistic_regression2(y, tx, initial_w, max_iters, gamma):
    """Logistic regression using gradient descent or SGD"""
    
    w = initial_w
    losses = []
    threshold = 1e-10
    
    for n_iter in range(max_iters):       
        gradient = np.dot(tx.T, (sigmoid(np.dot(tx, w)) - y))
        w -= gradient * gamma
        loss = logistic_loss(y, tx, w)
        losses.append(loss)
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break
        
    loss = losses[-1]
        
    return w, loss

def logistic_regression3(y, tx, initial_w, batch_size, max_iters, gamma):
    """Logistic regression using gradient descent or SGD"""
    
    w = initial_w
    losses = []
    threshold = 1e-10
    n_iter = 0
    loss = 0
    
    for y_batch, tx_batch in batch_iter(y, tx, batch_size=batch_size, num_batches=max_iters):  
        n_iter += 1
        gradient = np.dot(tx_batch.T, (sigmoid(np.squeeze(np.dot(tx_batch, w))) - y_batch))
        w -= gradient * gamma
        loss = logistic_loss(y_batch, tx_batch, w)
        if (n_iter%10000 == 0):
            y_ = sigmoid(np.dot(tx,w))
            classifier = lambda t: 1.0 if (t > 0.5) else 0.0
            classifier = np.vectorize(classifier)
            y_ = classifier(y_)
            ratio = 1 - sum(abs(y_ - y))/len(y)
            print("Itération = {i}".format(i = n_iter) + ", ratio = {r}".format(r = ratio) + ", cost = {c}".format(c = loss))
        
    return w, loss

def reg_logistic_regression(y, tx, lambda_, initial_w, batch_size, max_iters, gamma):
    """Logistic regression with regularization using SGD"""
    
    w = initial_w
    losses = []
    threshold = 1e-10
    n_iter = 0
    
    for y_batch, tx_batch in batch_iter(y, tx, batch_size=batch_size, num_batches=max_iters):
        n_iter += 1
        gradient = 2*lambda_*w + np.dot(tx_batch.T, (sigmoid(np.squeeze(np.dot(tx_batch, w))) - y_batch))
        w -= gradient * gamma
        loss = reg_logistic_loss(y_batch, tx_batch, w, lambda_)
        if (n_iter%10000 == 0):
            y_ = sigmoid(np.dot(tx,w))
            classifier = lambda t: 1.0 if (t > 0.5) else 0.0
            classifier = np.vectorize(classifier)
            y_ = classifier(y_)
            ratio = 1 - sum(abs(y_ - y))/len(y)
            print("Itération = {i}".format(i = n_iter) + ", ratio = {r}".format(r = ratio))
        losses.append(loss)
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break
            
    loss = losses[-1]
    
    return w, loss