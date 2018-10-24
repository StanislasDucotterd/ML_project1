
# coding: utf-8

# In[3]:


import numpy as np
from helpers import *
from implementations import *
from cleaning import *
from build_polynomial import *
y, x, ids = load_csv_data('train.csv', sub_sample=False)
classifier = lambda t: 1.0 if (t == 1.0) else 0.0
classifier = np.vectorize(classifier)
y = classifier(y)
tx, mean_x, std_x = clean_data(x)
tx = np.c_[np.ones(len(y)), tx]


# In[4]:


tx2 = build_poly_all_features(tx, 2)
tx5 = build_poly_all_features(tx, 5)


# In[5]:


w2 = logistic_regression3(y, tx2, np.zeros((61,)), 1, 50000, 0.005)


# In[6]:


w5 = logistic_regression3(y, tx2, np.zeros((61,)), 1, 50000, 0.005)


# In[7]:


y_test, x_test, ids = load_csv_data('test.csv', sub_sample=False)


# In[8]:


mean_nonoutlier = non_outlier_mean(x_test)
np.shape(mean_nonoutlier)


# In[9]:


mean_replaced_outliers(x_test, mean_nonoutlier)


# In[10]:


x_test = x_test - mean_x
x_test = x_test / std_x


# In[11]:


x_test = np.c_[np.ones(len(y_test)), x_test]


# In[12]:


x_test2 = build_poly_all_features(x_test, 2)


# In[14]:


y_pred = sigmoid(np.dot(x_test2,w2[0]))


# In[16]:


y_predic = np.zeros(np.shape(y_pred))
classifier = lambda t: 1 if (t > 0.5) else -1
classifier = np.vectorize(classifier)
y_predic = classifier(y_pred)


# In[17]:


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


# In[18]:


create_csv_submission(ids, y_predic, 'submission2.csv')

