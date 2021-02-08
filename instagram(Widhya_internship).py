#!/usr/bin/env python
# coding: utf-8

# In[1]:


#importing libraries
import numpy as np
import pandas as pd


# In[2]:


#reading the dataset
data = pd.read_csv(r"C:\Users\Sumana bushireddy\Downloads\instagram.csv",low_memory = False)
data


# In[3]:


#taking time column into a list
time = data['Time since posted'].tolist()
time


# In[6]:


#replace hours with empty space
res = [x.replace(' hours', '') for x in time]
res


# In[9]:


#making a integer list
final_list = []
for i in range(0, len(res)): 
    final_list.append(int(res[i])) 
print(final_list)


# In[11]:


#appending the list as a column
data['Time since posted'] = final_list
data


# In[12]:


#importing sklearn
from sklearn.linear_model import LinearRegression  
from sklearn.model_selection import train_test_split  


# In[13]:


#making a different dataset with only inputs and output variables
data_1 = data[['Followers','Time since posted','Likes']]
data_1


# In[21]:


#collecting values of x and y
x = data_1.iloc[:,0:2].values
y = data_1.iloc[:,-1].values


# In[22]:


#spliting the data
x_train, x_test, y_train, y_test = train_test_split(x, y,test_size=0.2, random_state=0)


# In[25]:


#fitting the model
x = LinearRegression()
x.fit(x_train,y_train)


# In[29]:


#predicting likes according to question
likes = np.array([300,10]).reshape(1,-1)
pred = x.predict(likes)
pred


# In[30]:


#predicting values
x.predict([[300,10]])


# In[31]:


#getting predictions for whole test dataset
predictions = x.predict(x_test)
print(predictions)


# In[32]:


#Finding actual vs predicted
data_2 = pd.DataFrame({'Actual':y_test, 'Predictions': predictions})
data_2


# In[33]:


#importing mse for calculating mse
from sklearn.metrics import mean_squared_error


# In[35]:


#calculating mse
mean_squared_error(y_test, predictions)


# In[ ]:




