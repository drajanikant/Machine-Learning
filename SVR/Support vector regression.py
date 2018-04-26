
# coding: utf-8

# # Support vector regression

# Importing the statements
import pandas as pd
import numpy as pn
import matplotlib.pyplot as plt


# Importing data into data set
data_set = pd.read_csv("Position_Salaries.csv");

# input variable get seperated 
x = data_set.iloc[:, 1:2].values
print(x)

# Dependent variables
y = data_set.iloc[:,2].values

# Feature scalling 
from sklearn.preprocessing import StandardScaler

sc_x = StandardScaler()
x = sc_x.fit_transform(x)

sc_y = StandardScaler()
y = sc_y.fit_transform(y)


# Fitting SVR into data set
from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf')
regressor.fit(x, y)

# Prediction
y_pred = regressor.predict(6.5)

# Plotting the data into the graph
plt.scatter(x, y, color='red')
plt.plot(x, regressor.predict(x), color='blue')
plt.xlabel('Position')
plt.ylabel('Salary')
plt.title('plotting the calary vs the positon')
plt.show()
