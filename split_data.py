# -*- coding: utf-8 -*-
"""Function used to split the data between the training and the testing"""
import numpy as np
import random

def split_data(x, y, ratio, seed=1):
    """split the dataset based on the split ratio."""
    # set seed
    np.random.seed(seed)
    # generate random indices
    num_row = len(y)
    indices = np.random.permutation(num_row)
    index_split = int(np.floor(ratio * num_row))
    index_tr = indices[: index_split]
    index_te = indices[index_split:]
    # create split
    x_tr = x[index_tr]
    x_te = x[index_te]
    y_tr = y[index_tr]
    y_te = y[index_te]
    return x_tr, x_te, y_tr, y_te

def k_fold_cross_validation(x, y, k, test_pos):
    """split the data into k parts, k-1 for the training and 1 for the test"""
    
    n_sample = np.shape(x)[0]
    if (test_pos > k or test_pos < 1):
        raise ValueError('test_pos is not in {1,...,k}')
    else:
        n_subsample = n_sample // k
        
        indices_train1 = np.linspace(0, n_subsample * (test_pos-1) -1, n_subsample * (test_pos-1)).astype(int)
        indices_test = np.linspace(n_subsample * (test_pos-1), n_subsample*test_pos -1, n_subsample).astype(int)
        indices_train2 = np.linspace(n_subsample*test_pos, n_sample -1, n_subsample * (k - test_pos)).astype(int)
        indices_train = np.r_[indices_train1, indices_train2]
        
        x_train = x[indices_train,:]
        x_test = x[indices_test,:]
        y_train = y[indices_train]
        y_test = y[indices_test]        
        
        return x_train, y_train, x_test, y_test