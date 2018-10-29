# -*- coding: utf-8 -*-
"""Implementation of the 6 functions"""
import numpy as np
from costs import *
from compute_gradient import *
from helpers import *
from build_polynomial import *

def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    """Gradient descent algorithm."""
    
    w = initial_w
    losses = []
    
    for n_iter in range(max_iters):
        gradient = compute_gradient(y, tx, w)[0]
        w = w - gamma*gradient # Update w by a fraction of the gradient
        loss = compute_mse(y, tx, w)  
        losses.append(loss)
<<<<<<< HEAD
        
        # Stop when the losses change less than the threshold
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break

    return w, losses[-1]
=======
            
    loss = losses[-1]
      
    return w, loss
>>>>>>> b1053566a80673628390500c5fade41be70dd693

def least_squares_SGD(y, tx, initial_w, batch_size, max_iters, gamma):
    """Stochastic gradient descent."""

    w = initial_w
    losses = []
    threshold = 1e-6
    n_iter = 0
    
<<<<<<< HEAD
    for y_batch, tx_batch in batch_iter(y, tx, batch_size=batch_size, num_batches=max_iters):
        gradient = compute_gradient(y_batch, tx_batch, w)[0] # Compute the gradient with only a batch of data
        w = w - gamma * gradient # Update w with the stochastic gradient
            
    loss = compute_mse(y, tx, w)

    return w, loss
=======
    for y_batch, tx_batch in batch_iter2(y, tx, batch_size=batch_size, num_batches=max_iters):
        
        n_iter += 1
            
            # compute gradient
        gradient = compute_gradient(y_batch, tx_batch, w)[0]
            
            # update w through the stochastic gradient update
        w = w - gamma * gradient
        loss = compute_mse(y, tx, w)
        losses.append(loss)
        if n_iter > 1:
            if abs(losses[-2] - losses[-1]) < threshold:
                break
    return w, losses[-1]
>>>>>>> b1053566a80673628390500c5fade41be70dd693

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
    threshold = 1e-6
    
    for n_iter in range(max_iters):       
        gradient = np.dot(tx.T, (sigmoid(np.dot(tx, w)) - y))
        w -= gradient * gamma
        loss = logistic_loss(y, tx, w)
        losses.append(loss)
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break
        
    loss = losses[-1]
        
    return w, loss

def logistic_regression3(y, tx, initial_w, max_iters, gamma):
    """Logistic regression using gradient descent or SGD"""
    
    w = initial_w
    losses = []
    threshold = 1e-6
    
    for y_batch, tx_batch in batch_iter(y, tx, batch_size = 1, num_batches=max_iters):  
        gradient = np.dot(tx_batch.T, (sigmoid(np.dot(tx_batch, w))) - y_batch)
        w -= gradient * gamma
        loss = np.squeeze(logistic_loss(y_batch, tx_batch, w))
        losses.append(loss)
        
    return w, losses

def logistic_regression4(y, tx, initial_w, max_iters, gamma, step_reduction, lambda_=0.0, batch_size=1, logs=False, shuffle=False):
    """Logistic regression using gradient descent or SGD"""
    
    w = initial_w
    n_iter = 0
    
    classifier = lambda t: 1.0 if (t > 0.5) else 0.0
    classifier = np.vectorize(classifier)
    
    batches = batch_iter2(y, tx, batch_size, max_iters, shuffle)
    
    for y_batch, tx_batch in batches:
        n_iter += 1
        
        batch_w = np.dot(tx_batch, w)
        sig = sigmoid(np.squeeze(batch_w))
        reg = 2 * lambda_ * w
        
        error = y_batch - sig
        gradient = reg - np.dot(tx_batch.T, error)/len(y_batch)
        w -= gradient * gamma / step_reduction(n_iter)
        if (logs and n_iter%10000 == 0):
            y_ = sigmoid(np.dot(tx,w))
            y_ = classifier(y_)
            ratio = 1 - sum(abs(y_ - y))/len(y)
            print("Itération = {i}".format(i = n_iter) + ", ratio = {r}".format(r = ratio), end="\r")
        
    return w

def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """Logistic regression with regularization using SGD"""
    
    w = initial_w
    losses = []
    threshold = 1e-10
    n_iter = 0
    
    for y_batch, tx_batch in batch_iter(y, tx, batch_size = 1, num_batches=max_iters):
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
            print("Itération = {i}".format(i = n_iter) + ", ratio = {r}".format(r = ratio), end="\r")
    
    return w, loss

def train_category(x, y, columns, iterations, gamma, step_reduction, lambda_, batch_size, poly_deg, logs=False, shuffle=False):
    x, mean_x, std_x = standardize(x)
    tx = np.c_[np.ones(len(x)), x]
    tx_poly = build_poly_all_features(tx, poly_deg)
    initial_w = np.zeros(poly_deg * columns + 1,)
    
    w = logistic_regression4(y, tx_poly, initial_w, iterations, gamma, step_reduction, lambda_, batch_size, logs, shuffle)
    
    return w, mean_x, std_x

def test_category(x_test, mean_x, std_x, ids, w, classifier, poly_deg):
    x_test = (x_test - mean_x) / std_x
    x_test = np.c_[np.ones(len(x_test)), x_test]
    x_test_poly = build_poly_all_features(x_test, poly_deg)
    y_odds = sigmoid(np.dot(x_test_poly, w))
    y_predic = classifier(y_odds)
    y_id = np.c_[ids, y_predic]
    
    return y_id
