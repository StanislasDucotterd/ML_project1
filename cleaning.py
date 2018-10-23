import numpy as np

def non_outlier_mean(dataset):
    mean_vector = np.zeros(len(dataset[0]))
    for i in range(0, len(dataset[0])):
        column_sum = 0
        n_samples = 0
        for j in range(0, len(dataset)):
            if dataset[j, i] != -999.0:
                column_sum += dataset[j, i]
                n_samples += 1
        mean_vector[i] = column_sum / n_samples
    return mean_vector

def mean_replaced_outliers(dataset, mean_vector):
    for i in range(0, len(dataset[0])):
        for j in range(0, len(dataset)):
            if dataset[j, i] == -999.0:
                dataset[j, i] = mean_vector[i]

def clean_data(dataset):
    mean_vector = non_outlier_mean(dataset)
    mean_replaced_outliers(dataset, mean_vector)
    dataset = standardize_columns(dataset)[0]
    return dataset

def standardize_columns(x):
    """Standardize the original data set."""
    mean_x = np.mean(x, axis=0)
    x = x - mean_x
    std_x = np.std(x, axis=0)
    x = x / std_x
    return x, mean_x, std_x