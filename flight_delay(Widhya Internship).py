#!/usr/bin/env python
# coding: utf-8

# # Flight delay task

# In[1]:


import pandas as pd


# In[2]:


#reading the data
data = pd.read_csv(r"C:\Users\Sumana bushireddy\Downloads\archive\flights.csv",low_memory = False)
data


# In[3]:


#taking a sample of a data
data_sample = data.head(100000)
data_sample


# In[4]:


#getting the info of the dataset
data_sample.info()


# In[5]:


#to know how many flights got diverted
data_sample['DIVERTED'].value_counts()


# In[7]:


#finding the correlation matrix
cor_matrix = data_sample.corr().abs()
print(cor_matrix)


# In[8]:


import numpy as np
upper_tri = cor_matrix.where(np.triu(np.ones(cor_matrix.shape),k=1).astype(np.bool))
print(upper_tri)


# In[18]:


#printing columns to be dropped
to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > 0.8)]
print()
print(to_drop)


# In[19]:


#getting the column name with high correlation with arrival_delay
data_sample['ARRIVAL_DELAY'].corr(data_sample['DEPARTURE_DELAY'])


# In[21]:


#to know number of null values in columns
data_sample.isnull().sum()


# In[24]:


#filling null values with mean of a column
data_sample = data_sample.fillna(data_sample.mean())
data_sample


# In[25]:


#rechecking the null values
data_sample.isnull().sum()


# In[104]:


#appending a column called result
data_sample['Result'] = np.NaN


# In[105]:


data_sample


# In[106]:


#calculating the length 
l = len(data_sample)
l


# In[109]:


#making result value 1 if arrival_delay > 15 else 0
data_sample['Result'] = [1 if x > 15 else 0 for x in data_sample['ARRIVAL_DELAY']]


# In[110]:


data_sample


# In[111]:


#counting number of flights delayed
data_sample['Result'].value_counts()


# In[121]:


#making other dataset with given columns
data_sample2 = data_sample[['MONTH','DAY','SCHEDULED_DEPARTURE','DEPARTURE_DELAY','SCHEDULED_ARRIVAL','DIVERTED','CANCELLED','AIR_SYSTEM_DELAY','SECURITY_DELAY','AIRLINE_DELAY','LATE_AIRCRAFT_DELAY','WEATHER_DELAY','Result']]
data_sample2


# In[122]:


#collecting x abd y values
x = data_sample2.iloc[:,0:12]
#y = data_sample2.iloc[:,-1]
y = data_sample2.iloc[:,-1]
y


# In[123]:


#importing sklearn
from sklearn.linear_model import LinearRegression  
from sklearn.model_selection import train_test_split  


# In[124]:


#spliting the data
x_train, x_test, y_train, y_test = train_test_split(x, y,test_size=0.3, random_state=42)


# In[125]:


#importing standardscaler and fitting the model
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform (x_test)


# In[126]:


#importing decision tree classifier
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier()
classifier.fit(x_train, y_train)


# In[127]:


#predicting values
y_pred = classifier.predict(x_test)


# In[128]:


#importing roc curve
import matplotlib.pyplot as plt
from sklearn.metrics import (confusion_matrix, precision_recall_curve, auc,
                             roc_curve, recall_score, classification_report, f1_score,
                             precision_recall_fscore_support)


# In[129]:


#calculating auc value
fp_rate, tp_rate, thresholds = roc_curve(y_test, y_pred)
auc(fp_rate, tp_rate)

