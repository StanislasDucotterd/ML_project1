# -*- coding: utf-8 -*-
"""implement a polynomial basis function."""
import numpy as np
from logistic_regression5 import *

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

def build_poly_vector(x, poly_vect):
    "Build a different polynom for each feature"
    tx = x
    for i in range(len(poly_vect)):
        tx = build_poly_feature(tx, i+1, poly_vect[i])
    return tx

def best_polynom(y, x, n_iter, gamma):
    "Find a polynome for each feature in order to best fit the data"
    polynoms = np.zeros((np.shape(x)[1]-1,))
    phi = x
    for n_feature in range(1,31):
        accuracy = logistic_regression5(y, phi, np.zeros(np.shape(phi)[1]), n_iter, gamma)[2]
        print("Feature " + str(n_feature) + " accuracy : " + str(accuracy))
        best_degree = 1
        for degree in range(2,6):
            phi_test = build_poly_feature(phi, n_feature, degree)
            accuracy_ = logistic_regression5(y, phi_test, np.zeros(np.shape(phi_test)[1]), n_iter, gamma)[2]
            print("Degree " + str(degree) + " accuracy : " + str(accuracy_))
            if (accuracy_ > accuracy):
                accuracy = accuracy_
                best_degree = degree
        phi = build_poly_feature(phi, n_feature, best_degree)
        polynoms[n_feature - 1] = best_degree
    final_accuracy = logistic_regression5(y, phi, np.zeros(np.shape(phi)[1]), n_iter, gamma)[2]
    return polynoms, final_accuracy