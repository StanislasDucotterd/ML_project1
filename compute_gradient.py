# -*- coding: utf-8 -*-
"""Gradient Descent"""
import numpy as np

def compute_gradient(y, tx, w):
    """Compute the gradient."""
   
    n_sample = len(y)
    e = y - np.dot(tx,w)
    return (-1/n_sample)*np.dot(tx.T,e), e