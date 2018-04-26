import pandas as pd

# Reading the data form the csv file that have the headers at 2nd row
dataset = pd.read_csv("pandas/weather_data.csv", header= 1)

# Reading the csv file which dosent have headers
dataset2 = pd.read_csv("pandas/weather_data1.csv", header= None, names=['day','temperature','windspeed','event'])

# Reading the data with specified amount of rows
dataset3 = pd.read_csv("pandas/weather_data.csv", header= 1, nrows=3)


# Cleaning the data while reading the data
dataset4 = pd.read_csv("pandas/stock_data.csv", na_values={
        "eps":["n.a.", "not available"],
    "revenue":["n.a.", "not available",-1],
    "price":["n.a.", "not available"],
    "people":["n.a.", "not available"]
})

# Writing the files into the csv

dataset4.to_csv("pandas/new_data.csv")

dataset4.to_csv("pandas/new_data.csv", header=False)

dataset4.to_csv("pandas/new_data.csv", header=False, index=False)