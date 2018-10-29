
# coding: utf-8

# In[1]:


# Imports
import numpy as np
from helpers import *
from implementations import *
from cleaning import *
from build_polynomial import *
from split_data import *


# In[2]:


# Load the data from csv
y, x, ids = load_csv_data('data/train.csv', sub_sample=False)


# In[3]:


# Define all the parameters

# Classifier turning the -1s to 0s
classifier_train = lambda t: 1.0 if (t == 1.0) else 0.0
classifier_train = np.vectorize(classifier_train)

# Classifier based on the odds
# odd > 0.5 => decision(odd) = 1
# odd <= 0.5 => decision(odd) = -1
classifier_test = lambda t: 1 if (t > 0.5) else -1
classifier_test = np.vectorize(classifier_test)

iterations = 500001 # Number of iterations of the SGD (will linearly slow down the learning process)
batch_size = 10 # Number of samples used for each step of the SGD (will linearly slow down the learning process)
lambda_ = 10**(-4) # Regression parameter
gamma = 0.05 # Initial step size
poly_deg = 3 # Degree of the polynomial expansion of the data
logs = True # If set to True, prints some logs of the progress of the training (can slow down the learning process a little bit)
shuffle = False # Wether or not the batches from batch_iter are random. Has to be set to False to keep the program deterministic

# 'columns' parameters are of the form ['name', #1, #2, #3, #4] where #k is the number of columns we keep for data category k
columns_pri = ['pri', 9, 13, 16, 16] # useful when we decide to use only primitive columns
columns_der = ['der', 9, 9, 29, 29] # useful when we decide to use only derived columns
columns_all = ['all', 18, 22, 29, 29] # useful when we decide to use all columns

columns = columns_all # We use all available columns (except the ones that are undefined for the data category)


# In[4]:


# Clean the data

# Turn the -1s to 0s in y
y = classifier_train(y)

# Replace the outliers in column DER_mass_MMC by 0
x = clean_first_column(x)

# Split the data into 4 categories, depending on their value of column PRI_num_jet
x0, x1, x2, x3, y0, y1, y2, y3 = split_categories_train(x, y, columns=columns[0])


# In[5]:


w0, mean_x0, std_x0 = train_category(x0, y0, columns[1], iterations, gamma, lambda_, batch_size, poly_deg, logs, shuffle)


# In[6]:


w1, mean_x1, std_x1 = train_category(x1, y1, columns[2], iterations, gamma, lambda_, batch_size, poly_deg, logs, shuffle)


# In[7]:


w2, mean_x2, std_x2 = train_category(x2, y2, columns[3], iterations, gamma, lambda_, batch_size, poly_deg, logs, shuffle)


# In[8]:


w3, mean_x3, std_x3 = train_category(x3, y3, columns[4], iterations, gamma, lambda_, batch_size, poly_deg, logs, shuffle)


# In[9]:


y_test, x_test, ids_test = load_csv_data('data/test.csv', sub_sample=False)
x_test0, x_test1, x_test2, x_test3, ids0, ids1, ids2, ids3 = split_categories_test(x_test, ids_test, columns=columns[0])


# In[10]:


y_id_0 = test_category(x_test0, mean_x0, std_x0, ids0, w0, classifier_test, poly_deg)
y_id_1 = test_category(x_test1, mean_x1, std_x1, ids1, w1, classifier_test, poly_deg)
y_id_2 = test_category(x_test2, mean_x2, std_x2, ids2, w2, classifier_test, poly_deg)
y_id_3 = test_category(x_test3, mean_x3, std_x3, ids3, w3, classifier_test, poly_deg)


# In[11]:


# Merge the results together
y_id = np.append(y_id_0, np.append(y_id_1, np.append(y_id_2, y_id_3, axis=0), axis=0), axis=0)

# Sort the results by ID
y_id = y_id[y_id[:, 0].argsort()]

# Keep only the predictions
y_predic = y_id[:,1]


# In[12]:


# Create the csv file with the results
create_csv_submission(ids_test, y_predic, 'submission_deg3_reg.csv')

