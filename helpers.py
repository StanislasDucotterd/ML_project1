# -*- coding: utf-8 -*-
"""some helper functions."""
import numpy as np
import csv
import matplotlib.pyplot as plt
from implementations import *


def load_data(filename):
    """load data."""
    x = np.loadtxt(filename, delimiter=",", skiprows=1, usecols = range(2,32), unpack=True)
    x = x.T
    classification = np.loadtxt(filename, dtype = str, delimiter=",", skiprows=1, usecols = 1, unpack=True)
    classifier = lambda t: 1.0 if (t == 's') else 0.0
    classifier = np.vectorize(classifier)
    y = classifier(classification)
    return x, y

def load_csv_data(data_path, sub_sample=False):
    """Loads data and returns y (class labels), tX (features) and ids (event ids)"""
    y = np.genfromtxt(data_path, delimiter=",", skip_header=1, dtype=str, usecols=1)
    x = np.genfromtxt(data_path, delimiter=",", skip_header=1)
    ids = x[:, 0].astype(np.int)
    input_data = x[:, 2:]

    # convert class labels from strings to binary (-1,1)
    yb = np.ones(len(y))
    yb[np.where(y=='b')] = -1
    
    # sub-sample
    if sub_sample:
        yb = yb[::50]
        input_data = input_data[::50]
        ids = ids[::50]

    return yb, input_data, ids

def standardize(x):
    """Standardize the original data set."""
    mean_x = np.mean(x, axis = 0)
    x = x - mean_x
    std_x = np.std(x, axis = 0)
    x = x / std_x
    return x, mean_x, std_x


def build_model_data(height, weight):
    """Form (y,tX) to get regression data in matrix form."""
    y = weight
    x = height
    num_samples = len(y)
    tx = np.c_[np.ones(num_samples), x]
    return y, tx


def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    """
    Generate a minibatch iterator for a dataset.
    Takes as input two iterables (here the output desired values 'y' and the input data 'tx')
    Outputs an iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`.
    Data can be randomly shuffled to avoid ordering in the original data messing with the randomness of the minibatches.
    Example of use :
    for minibatch_y, minibatch_tx in batch_iter(y, tx, 32):
        <DO-SOMETHING>
    """
    data_size = len(y)

    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_y = y[shuffle_indices]
        shuffled_tx = tx[shuffle_indices]
    else:
        shuffled_y = y
        shuffled_tx = tx
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        if start_index != end_index or 1:
            yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]
            
def batch_iter2(y, tx, batch_size, num_batches=1, shuffle=True):
    """
    Generate a minibatch iterator for a dataset.
    Takes as input two iterables (here the output desired values 'y' and the input data 'tx')
    Outputs an iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`.
    Data can be randomly shuffled to avoid ordering in the original data messing with the randomness of the minibatches.
    Example of use :
    for minibatch_y, minibatch_tx in batch_iter(y, tx, 32):
        <DO-SOMETHING>
    """
    data_size = len(y)

    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_y = y[shuffle_indices]
        shuffled_tx = tx[shuffle_indices]
    else:
        shuffled_y = y
        shuffled_tx = tx
    for batch_num in range(num_batches):
        start_index = (batch_num * batch_size) % data_size
        end_index = min(start_index + batch_size, data_size)
        yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def create_csv_submission(ids, y_pred, name):
    """
    Creates an output file in csv format for submission to kaggle
    Arguments: ids (event ids associated with each prediction)
               y_pred (predicted class labels)
               name (string name of .csv output file to be created)
    """
    with open(name, 'w') as csvfile:
        fieldnames = ['Id', 'Prediction']
        writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames)
        writer.writeheader()
        for r1, r2 in zip(ids, y_pred):
            writer.writerow({'Id':int(r1),'Prediction':int(r2)})
            
def validation_polynomials(y, tx, degree):
    """Make a plot of the validation and train set accuracy for each degree"""
    
    accuracy_test = np.zeros((degree,1))
    accuracy_train = np.zeros((degree,1))
    for degree in range(1,degree + 1):
        tx_ = build_poly_all_features(tx, degree)
        for pos in range(1,11):
            x_train, y_train, x_test, y_test = k_fold_cross_validation(tx_, y, 10, pos)
            w = logistic_regression4(y_train, x_train, np.zeros((1 + degree*30,)), 500001, 0.05)
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
    plt.savefig('polynoms with validation and training set')
    plt.show
          