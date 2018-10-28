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
    return standardize_columns(dataset)

def standardize_columns(x):
    """Standardize the original data set."""
    mean_x = np.mean(x, axis=0)
    x = x - mean_x
    std_x = np.std(x, axis=0)
    x = x / std_x
    return x, mean_x, std_x

def split_categories_train(x, y, columns = 'all'):
    x0_count, x1_count, x2_count, x3_count = 0, 0, 0, 0

    for row in x:
        if row[22] == 0.0:
            x0_count += 1
        elif row[22] == 1.0:
            x1_count += 1
        elif row[22] == 2.0:
            x2_count += 1
        else:
            x3_count += 1

    x0 = np.ndarray(shape=(x0_count, 30), dtype=float)
    x1 = np.ndarray(shape=(x1_count, 30), dtype=float)
    x2 = np.ndarray(shape=(x2_count, 30), dtype=float)
    x3 = np.ndarray(shape=(x3_count, 30), dtype=float)
    
    y0 = np.ndarray(shape=(x0_count,), dtype=float)
    y1 = np.ndarray(shape=(x1_count,), dtype=float)
    y2 = np.ndarray(shape=(x2_count,), dtype=float)
    y3 = np.ndarray(shape=(x3_count,), dtype=float)

    i0, i1, i2, i3 = 0, 0, 0, 0

    for i in range(0, len(x)):
        row = x[i]
        if row[22] == 0.0:
            x0[i0] = row
            y0[i0] = y[i]
            i0 += 1
        elif row[22] == 1.0:
            x1[i1] = row
            y1[i1] = y[i]
            i1 += 1
        elif row[22] == 2.0:
            x2[i2] = row
            y2[i2] = y[i]
            i2 += 1
        else:
            x3[i3] = row
            y3[i3] = y[i]
            i3 += 1
    
    if columns == 'all':
        x0 = np.delete(x0, [4, 5, 6, 12, 22, 23, 24, 25, 26, 27, 28, 29], axis=1)
        x1 = np.delete(x1, [4, 5, 6, 12, 22, 26, 27, 28], axis=1)
        x2 = np.delete(x2, [22], axis=1)
        x3 = np.delete(x3, [22], axis=1)
    elif columns == 'der':
        x0 = np.delete(x0, [4, 5, 6, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29], axis=1)
        x1 = np.delete(x1, [4, 5, 6, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29], axis=1)
        x2 = np.delete(x2, [13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29], axis=1)
        x3 = np.delete(x3, [13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29], axis=1)
    elif columns == 'pri':
        x0 = np.delete(x0, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 22, 23, 24, 25, 26, 27, 28, 29], axis=1)
        x1 = np.delete(x1, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 22, 26, 27, 28], axis=1)
        x2 = np.delete(x2, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 22], axis=1)
        x3 = np.delete(x3, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 22], axis=1)
    else:
        print('columns attribute value unrecognized')

    return x0, x1, x2, x3, y0, y1, y2, y3

def split_categories_test(x, ids, columns = 'all'):
    x0_count, x1_count, x2_count, x3_count = 0, 0, 0, 0

    for row in x:
        if row[22] == 0.0:
            x0_count += 1
        elif row[22] == 1.0:
            x1_count += 1
        elif row[22] == 2.0:
            x2_count += 1
        else:
            x3_count += 1

    x0 = np.ndarray(shape=(x0_count, 30), dtype=float)
    x1 = np.ndarray(shape=(x1_count, 30), dtype=float)
    x2 = np.ndarray(shape=(x2_count, 30), dtype=float)
    x3 = np.ndarray(shape=(x3_count, 30), dtype=float)
    
    ids0 = np.ndarray(shape=(x0_count,), dtype=float)
    ids1 = np.ndarray(shape=(x1_count,), dtype=float)
    ids2 = np.ndarray(shape=(x2_count,), dtype=float)
    ids3 = np.ndarray(shape=(x3_count,), dtype=float)

    i0, i1, i2, i3 = 0, 0, 0, 0

    for i in range(0, len(x)):
        row = x[i]
        if row[22] == 0.0:
            x0[i0] = row
            ids0[i0] = ids[i]
            i0 += 1
        elif row[22] == 1.0:
            x1[i1] = row
            ids1[i1] = ids[i]
            i1 += 1
        elif row[22] == 2.0:
            x2[i2] = row
            ids2[i2] = ids[i]
            i2 += 1
        else:
            x3[i3] = row
            ids3[i3] = ids[i]
            i3 += 1
            
    if columns == 'all':
        x0 = np.delete(x0, [4, 5, 6, 12, 22, 23, 24, 25, 26, 27, 28, 29], axis=1)
        x1 = np.delete(x1, [4, 5, 6, 12, 22, 26, 27, 28], axis=1)
        x2 = np.delete(x2, [22], axis=1)
        x3 = np.delete(x3, [22], axis=1)
    elif columns == 'der':
        x0 = np.delete(x0, [4, 5, 6, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29], axis=1)
        x1 = np.delete(x1, [4, 5, 6, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29], axis=1)
        x2 = np.delete(x2, [13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29], axis=1)
        x3 = np.delete(x3, [13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29], axis=1)
    elif columns == 'pri':
        x0 = np.delete(x0, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 22, 23, 24, 25, 26, 27, 28, 29], axis=1)
        x1 = np.delete(x1, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 22, 26, 27, 28], axis=1)
        x2 = np.delete(x2, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 22], axis=1)
        x3 = np.delete(x3, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 22], axis=1)
    else:
        print('columns attribute value unrecognized')

    return x0, x1, x2, x3, ids0, ids1, ids2, ids3

def clean_first_column(x):
    for row in x:
        if row[0] == -999.0:
            row[0] = 0
    return x

        