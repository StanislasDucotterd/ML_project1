# -*- coding: utf-8 -*-
"""some helper functions."""
import numpy as np
import matplotlib.pyplot as plt
from implementations import *
from split_data import *

def validation_polynomials(y, tx, degree, max_iters, gamma, step_reduction, lambda_=0.0, batch_size=1, logs=False,shuffle=False):
    """Make a plot of the validation and train set accuracy for each degree"""
    
    accuracy_test = np.zeros((degree,1))
    accuracy_train = np.zeros((degree,1))
    for degree in range(1,degree + 1):
        tx_ = build_poly_all_features(tx, degree)
        for pos in range(1,11):
            x_train, y_train, x_test, y_test = k_fold_cross_validation(tx_, y, 10, pos)
            w = logistic_regression4(y_train, x_train, np.zeros((np.shape(x_train)[1],)), max_iters, gamma, step_reduction, lambda_,batch_size, logs)
            y_pred_test = sigmoid(np.dot(x_test, w[0]))
            y_pred_train = sigmoid(np.dot(x_train, w[0]))
            accuracy_test[degree - 1,0] += prediction_accuracy(y_test, y_pred_test)/10
            accuracy_train[degree - 1, 0] += prediction_accuracy(y_train, y_pred_train)/10
    return accuracy_train, accuracy_test

def plot_polynomials(accuracy_train, accuracy_test, degree):
    polynomials = np.linspace(1,degree,degree)
    plt.plot(polynomials, accuracy_test, 'r', label = 'Validation set')
    plt.plot(polynomials, accuracy_train, 'b', label = 'Training set')
    plt.ylabel('Prediction accuracy')
    plt.xlabel('Polynomial degree')
    plt.legend()
    plt.title('Prediction accuracy for the validation set and the training set')
    plt.show
