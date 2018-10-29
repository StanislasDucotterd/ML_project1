# -*- coding: utf-8 -*-
"""Function used to compute the loss."""
import numpy as np
from costs import *
from compute_gradient import *
from helpers import *

def logistic_regression5(y, tx, initial_w, max_iters, gamma):
    """Logistic regression using gradient descent or SGD"""
    
    w = initial_w
    n_iter = 0
    
    for y_batch, tx_batch in batch_iter(y, tx, batch_size = 1, num_batches=max_iters):  
        n_iter += 1
        gradient = np.dot(tx_batch.T, (sigmoid(np.squeeze(np.dot(tx_batch, w))) - y_batch))
        w -= gradient * gamma / np.sqrt(n_iter)
    y_ = sigmoid(np.dot(tx,w))
    classifier = lambda t: 1.0 if (t > 0.5) else 0.0
    classifier = np.vectorize(classifier)
    y_ = classifier(y_)
    ratio = 1 - sum(abs(y_ - y))/len(y)
    loss = logistic_loss(y_batch, tx_batch, w)
        
    return w, loss, ratio
