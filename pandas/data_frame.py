#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 25 15:23:28 2018
Title: Data frames in the pandas
Description: Oprations on the data frames in pandas.
@author: rajanikant
"""

import pandas as pd

# Importing the data from the data sources
dataset = pd.read_csv("Iris.csv")

# Printing the rows
dataset[1:5] # printing the rows from 1 to 5

dataset.head(10) # print the top 10 rows

dataset.tail(3) # print the last 3 rows

# priniting the columns 

dataset["SepalLengthCm"].head() # print the column with top 5 results

dataset[["SepalLengthCm", "SepalWidthCm","PetalLengthCm"]].head()

dataset["SepalLengthCm"].max()

dataset["SepalLengthCm"].min()

dataset.describe()