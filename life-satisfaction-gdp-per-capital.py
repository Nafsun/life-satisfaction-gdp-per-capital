# -*- coding: utf-8 -*-
"""
Created on Thu Jan 28 09:36:36 2021

@author: Nafsun
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.linear_model

# Load the data
oecd_bli = pd.read_csv("oecd_bli_2015.csv", thousands=',')
gdp_per_capita = pd.read_csv('gdp_per_capita.csv', encoding='latin1', na_values='n/a')

# Filter out what we need in the life expentancy csv file
oecd_bli = oecd_bli[(oecd_bli['Inequality'] == 'Total') & (oecd_bli['Indicator'] == 'Life expectancy')]

# merge the two arrays base on Country column
marger = pd.merge(gdp_per_capita, oecd_bli, on=["Country"]) 

# Prepare the data
gdp_value = marger.loc[:, "2015"]
bli_value = marger.loc[:, "Value"]

#concatenate the two datas we need into a 2d array
country_stats = pd.concat([gdp_value, bli_value], axis=1)

#Choose each element as X, y
X = np.array(country_stats.loc[:, "2015"]) # GDP per Capital
y = np.array(country_stats.loc[:, "Value"]) # life expectancy

# Visualize the data
plt.scatter(x=X, y=y, s=40)
plt.show()

#Convert the GDP per Capital from 1D to 2D
X = X[:, np.newaxis]

# Select a linear model
model = sklearn.linear_model.LinearRegression()

# Train the model
model.fit(X, y)

print(model.get_params())

# Make a prediction for Cyprus and Nigeria
X_new = [[22587], [2229]] # Cyprus's and Nigeria GDP per capita

print(model.predict(X_new)) # outputs [[ 5.96242338]]
