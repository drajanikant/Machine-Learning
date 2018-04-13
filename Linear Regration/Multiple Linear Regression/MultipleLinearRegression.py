#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  7 14:24:36 2018

@author: rajanikant
"""

# Import the required library for the Model Building
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data_set = pd.read_csv("50_Startups.csv");

# input variable get seperated 
x = data_set.iloc[:, :-1].values
y = data_set.iloc[:,4].values


# Encoding categorical data
# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
x[:, 3] = labelencoder_X.fit_transform(x[:, 3])

onehotencoder = OneHotEncoder(categorical_features = [3])
x = onehotencoder.fit_transform(x).toarray()


# Avoiding the dummy variable trap
x = x[:,1:]

# Deviding the data set into the training and testing datasets
from sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# Feature Scalling
"""
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)
"""

# fitting Multiple regression
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)

# Predicting the test set result
y_pred = regressor.predict(x_test)

# Checking the score
from sklearn.metrics import accuracy_score
score = accuracy_score(y_test, y_pred)

regressor.score(y_test, y_pred)  

# Building the optimal module using backword elimination
import statsmodels.formula.api as sm
x = np.append( arr = np.ones((50,1)).astype(int), values = x, axis = 1)
x_opt = x[:, [0, 1, 2, 3, 4, 5]]
regressor_ols = sm.OLS(endog=y, exog=x_opt).fit()
regressor_ols.summary()

x_opt = x[:, [0, 1, 3, 4, 5]]
regressor_ols = sm.OLS(endog=y, exog=x_opt).fit()
regressor_ols.summary()

x_opt = x[:, [0, 3, 4, 5]]
regressor_ols = sm.OLS(endog=y, exog=x_opt).fit()
regressor_ols.summary()

x_opt = x[:, [0, 3, 5]]
regressor_ols = sm.OLS(endog=y, exog=x_opt).fit()
regressor_ols.summary()

x_opt = x[:, [0, 3]]
regressor_ols = sm.OLS(endog=y, exog=x_opt).fit()
regressor_ols.summary()



# Plotting of the test data into the scatter graph
plt.scatter(x_test, y_test, color = 'red')
plt.plot(x_train, regressor.predict(x_train), color = 'blue')
plt.title("Salary Vs. Experience [ Test Set ]")
plt.xlabel("Startup Datas")
plt.ylabel("Salary")
plt.show("profit")
