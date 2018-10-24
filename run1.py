
# coding: utf-8

# In[1]:


import numpy as np
from helpers import *
from implementations import *
from cleaning import *
y, x, ids = load_csv_data('train.csv', sub_sample=False)


# In[2]:


classifier = lambda t: 1.0 if (t == 1.0) else 0.0
classifier = np.vectorize(classifier)
y = classifier(y)


# In[3]:


tx, mean_x, std_x = clean_data(x)


# In[4]:


tx = np.c_[np.ones(len(y)), tx]


# In[ ]:


w = logistic_regression3(y, tx, np.zeros((31,)), 1, 200000, 0.005)[0]


# In[7]:


y_test, x_test, ids = load_csv_data('test.csv', sub_sample=False)


# In[ ]:


mean_nonoutlier = non_outlier_mean(x_test)
np.shape(mean_nonoutlier)


# In[ ]:


mean_nonoutlier = mean_nonoutlier[:30]


# In[ ]:


np.shape(mean_nonoutlier)


# In[ ]:


mean_replaced_outliers(x_test, mean_nonoutlier)


# In[ ]:


x_test = x_test - mean_x
x_test = x_test / std_x


# In[ ]:


x_test = np.c_[np.ones(len(y_test)), x_test]


# In[ ]:


y_pred = sigmoid(np.dot(x_test,w))


# In[ ]:


y_predic = np.zeros(np.shape(y_pred))
classifier = lambda t: 1 if (t > 0.5) else -1
classifier = np.vectorize(classifier)
y_predic = classifier(y_pred)


# In[ ]:


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


# In[ ]:


create_csv_submission(ids, y_predic, 'submission.csv')

