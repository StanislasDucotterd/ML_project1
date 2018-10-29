# -*- coding: utf-8 -*-
"""Implementation of the 6 functions"""
import numpy as np

def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    """Gradient descent algorithm."""
    
    w = initial_w
    losses = []
    threshold = 1e-8
    
    for n_iter in range(max_iters):
        gradient = compute_gradient(y, tx, w)
        w = w - gamma * gradient  # Update w by a fraction of the gradient
        loss = compute_mse(y, tx, w)  
        losses.append(loss)
        
        # Stop when the losses change less than the threshold
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break

    return w, loss


def least_squares_SGD(y, tx, initial_w, max_iters, gamma):
    """Stochastic gradient descent."""

    w = initial_w
    
    for y_batch, tx_batch in batch_iter(y, tx, 1, max_iters):
        gradient = compute_gradient(y_batch, tx_batch, w)  # Compute the gradient with only a batch of data
        w = w - gamma * gradient  # Update w with the stochastic gradient

    loss = compute_mse(y, tx, w)

    return w, loss


def least_squares(y, tx): 
    """Calculate the least squares solution."""

    a = tx.T.dot(tx)
    b = tx.T.dot(y)
    w = np.linalg.solve(a, b)
    loss = compute_mse(y, tx, w)

    return w, loss


def ridge_regression(y, tx, lambda_):
    """Calculate the ridge regression solution"""
    
    n_sample = len(y)
    I = 2 * n_sample * lambda_ * np.identity(np.shape(tx)[1])
    a = np.dot(tx.T, tx) + I
    b = np.dot(tx.T, y)
    w = np.linalg.solve(a, b)
    loss = compute_mse(y, tx, w)

    return w, loss


def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """Logistic regression using SGD"""

    w = initial_w
    losses = []

    for y_batch, tx_batch in batch_iter(y, tx, 1, max_iters):
        gradient = np.dot(tx_batch.T, (sigmoid(np.dot(tx_batch, w))) - y_batch)
        w -= gradient * gamma
        loss = np.squeeze(logistic_loss(y_batch, tx_batch, w))
        losses.append(loss)

    return w, loss


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """Logistic regression with regularization using SGD"""
    
    w = initial_w
    loss = 0
    n_iter = 0
    
    for y_batch, tx_batch in batch_iter(y, tx, 1, max_iters):
        n_iter += 1
        gradient = 2 * lambda_ * w + np.dot(tx_batch.T, (sigmoid(np.squeeze(np.dot(tx_batch, w))) - y_batch))
        w -= gradient * gamma
        loss = reg_logistic_loss(y_batch, tx_batch, w, lambda_)

    return w, loss


# HELPER METHODS

def compute_error(y, tx, w):
    """Calculate the error vector"""
    return y - np.dot(tx, w)


def compute_mse(y, tx, w):
    """Calculate the loss using MSE"""
    e = compute_error(y, tx, w)
    return (e**2).mean()/2


def compute_gradient(y, tx, w):
    """Compute the gradient."""
    n_samples = len(y)
    error = compute_error(y, tx, w)
    return (-1 / n_samples) * tx.T.dot(error)


def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    """
    >> This is a modified version of the batch_iter method given as a solution of a lab. This one can produce any number
    >> of batches, even bigger than the dataset itself (it will however yield the same data samples multiple times in
    >> this case)
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
    """Compute a sigmoid function of parameter z"""
    return 1 / (1 + np.exp(-z))


def logistic_loss(y, tx, w):
    """Compute the cost by negative log likelihood."""
    """We create the function handle_big_values in order to avoid to divide by zero error"""
    y_pred = np.dot(tx, w)
    handle_big_values = np.vectorize(lambda t: np.log(sigmoid(t)) if (t > -709.0) else t)
    log_likelihood = np.dot(y.T, handle_big_values(y_pred)) + np.dot((1 - y).T, handle_big_values(1 - y_pred))

    return np.squeeze(-log_likelihood)


def reg_logistic_loss(y, tx, w, lambda_):
    """Compute the cost by regularization and negative log likelihood."""

    return logistic_loss(y, tx, w) + np.squeeze(lambda_ * np.dot(w.T, w))

