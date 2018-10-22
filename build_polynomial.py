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