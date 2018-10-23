# -*- coding: utf-8 -*-
"""implement a polynomial basis function."""
import numpy as np

def build_poly(x, degree):
    """polynomial basis functions for input data x, for j=0 up to j=degree."""
    n_sample = len(x)
    phi = np.ones((n_sample,degree+1))
    for i in range(1, degree+1):
        phi[:,i] = x**i
    return phi

def build_poly_feature(x, n_feature, degree):
    "Build a polynomial for a given feature"
    tx = x
    for deg in range(2,degree+1):
        tx = np.c_[tx, x[:,n_feature]**deg]
    return tx

def build_poly_all_features(x, degree):
    "build polynomial for all features"
    n_features = len(x[0,:])
    tx = x
    for feature in range(1,n_features):
        tx = build_poly_feature(tx, feature, degree)
    return tx