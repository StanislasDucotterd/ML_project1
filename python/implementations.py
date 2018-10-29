# -*- coding: utf-8 -*-
"""Implementation of the 6 functions"""
import numpy as np


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


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def logistic_loss(y, tx, w):
    """Compute the cost by negative log likelihood."""


    y_pred = sigmoid(np.dot(tx, w))

    for i in range(len(y_pred)):
        if y_pred[i] == 0.0:
            y_pred[i] = 10**(-9)
        elif y_pred[i] == 1.0:
            y_pred[i] = 1 - 10**(-9)

    diff_log = np.log(1 - y_pred)
    diff_t = (1 - y).T
    pred_log = np.log(y_pred)

    dot1 = np.dot(y.T, pred_log)
    dot2 = np.dot(diff_t, diff_log)
    log_likelihood = dot1 + dot2

    #loss = 0
    #for n in range(len(tx)):
    #    loss += y[n] * np.log(sigmoid(tx[n].T.dot(w))) + (1 - y[n]) * np.log(1 - sigmoid(tx[n].T.dot(w)))

    #return -loss

    return np.squeeze(-log_likelihood)


def reg_logistic_loss(y, tx, w, lambda_):
    """compute the cost by regularization and negative log likelihood."""

    return logistic_loss(y, tx, w) + np.squeeze(lambda_ * np.dot(w.T, w))


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
    """Logistic regression using gradient descent or SGD"""

    w = initial_w
    losses = []
    threshold = 1e-10

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
    threshold = 1e-10
    n_iter = 0
    loss = 0
    pred_accuracy = 0
    
    for y_batch, tx_batch in batch_iter(y, tx, batch_size = 1, num_batches=max_iters):  
        n_iter += 1
        gradient = np.dot(tx_batch.T, (sigmoid(np.squeeze(np.dot(tx_batch, w))) - y_batch))
        w -= gradient * gamma
        loss = logistic_loss(y_batch, tx_batch, w)
        if (n_iter%10000 == 0):
            y_ = sigmoid(np.dot(tx,w))
            classifier = lambda t: 1.0 if (t > 0.5) else 0.0
            classifier = np.vectorize(classifier)
            y_ = classifier(y_)
            ratio = 1 - sum(abs(y_ - y))/len(y)
            print("Itération = {i}".format(i = n_iter) + ", ratio = {r}".format(r = ratio) + ", cost = {c}".format(c = loss))
            if (abs(pred_accuracy - ratio) < 1e-6):
                break
            pred_accuracy = ratio
        
    return w, loss, pred_accuracy


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