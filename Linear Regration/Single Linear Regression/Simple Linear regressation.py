
# coding: utf-8

# # Simple linear regression

# In[35]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[36]:


# Importing data into data set
salary_data_set = pd.read_csv("Salary_Data.csv");
salary_data_set


# In[37]:


# input variable get seperated 
x_experience = salary_data_set.iloc[:, :-1].values
x_experience


# In[38]:


# Dependent variables
y_salary = salary_data_set.iloc[:,1].values
y_salary


# In[39]:


# Seperating the testing and training data set
from sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x_experience, y_salary, test_size=1/3, random_state=0)


# In[40]:


# Fitting training module into the linear regresser
from sklearn.linear_model import LinearRegression
regresser = LinearRegression()
regresser.fit(x_train, y_train)


# In[41]:


# Predecting of the test result
y_predict = regresser.predict(x_test)
y_predict


# In[42]:


# regresser.score(y_test, y_predict)


# In[45]:


# Plotting of the training data into the scatter graph
plt.scatter(x_train, y_train, color = 'red')
plt.plot(x_train, regresser.predict(x_train), color = 'blue')
plt.title("Salary Vs. Experience [ Training Set ]")
plt.xlabel("Years of experience")
plt.ylabel("Salary")
plt.show()


# In[48]:


# Plotting of the test data into the scatter graph
plt.scatter(x_test, y_test, color = 'red')
plt.plot(x_train, regresser.predict(x_train), color = 'blue')
plt.title("Salary Vs. Experience [ Test Set ]")
plt.xlabel("Years of experience")
plt.ylabel("Salary")
plt.show()

