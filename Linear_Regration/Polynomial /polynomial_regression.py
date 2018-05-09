#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 10 11:42:44 2018

@author: rajanikant
"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Linear_Regration/Polynomial/Position_Salaries.csv')
x = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

# Splitting the dataset into the Training set and Test set
"""from sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)"""

# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""

# building the linear regression model 
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(x,y)

# Fitting the algorithm into the polynomial regression
from sklearn.preprocessing import PolynomialFeatures
ply_reg = PolynomialFeatures( degree=4)
x_poly = ply_reg.fit_transform(x)

lin_reg2 = LinearRegression()
lin_reg2.fit(x_poly, y)

# Visualisation of Linear Regression
plt.scatter(x, y, color="red")
plt.plot(x, lin_reg.predict(x), color="blue")
plt.title("Linear regresion model")
plt.xlabel("Experience of employee")
plt.ylabel("Salary of employee")
plt.show()

# Visulisation of Polynomial Regression

x_grid = np.arange(min(x), max(x), 0.1)
x_grid = x_grid.reshape((len(x_grid),1))
plt.scatter(x, y, color="red")
plt.plot(x_grid, lin_reg2.predict(ply_reg.fit_transform(x_grid)), color="blue")
plt.title("polynomial regresion model")
plt.xlabel("Experience of employee")
plt.ylabel("Salary of employee")
plt.show()


# Pridiction for new employee -- Linear Regression
lin_reg.predict(6.5)

# Pridiction for new employee -- Ploynomial Regression
lin_reg2.predict(ply_reg.fit_transform(6.5))