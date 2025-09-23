
# Author: Swati Mishra
# Created: Sep 3, 2024
# Modified by Charles Welch, Sep 9 2025
# License: MIT License
# Purpose: This python includes OLS method for Linear Regression 

# Usage: python linear_regression.py

# Dependencies: None
# Python Version: 3.6+

# Modification History:
# - Version 1 - added linear regression implementation

# References:
# - https://www.python.org/dev/peps/pep-0008/
# - Python Documentation

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# import data
data = pd.read_csv("datasets/gdp-vs-happiness.csv")

#drop columns that will not be used
by_year = (data[data['Year']==2018]).drop(columns=["World regions according to OWID","Code"])
# remove missing values from columns 
df = by_year[(by_year['Cantril ladder score'].notna()) & (by_year['GDP per capita, PPP (constant 2021 international $)']).notna()]

#create np.array for gdp and happiness where happiness score is above 4.5
happiness=[]
gdp=[]
for row in df.iterrows():
    if row[1]['Cantril ladder score']>4.5:
        happiness.append(row[1]['Cantril ladder score'])
        gdp.append(row[1]['GDP per capita, PPP (constant 2021 international $)'])

class linear_regression():
 
    def __init__(self,x_:list,y_:list) -> None:

        self.input = np.array(x_)
        self.target = np.array(y_)

    def preprocess(self,):

        #normalize the values
        hmean = np.mean(self.input)
        hstd = np.std(self.input)
        x_train = (self.input - hmean)/hstd

        #arrange in matrix format
        X = np.column_stack((np.ones(len(x_train)),x_train))

        #normalize the values
        gmean = np.mean(self.target)
        gstd = np.std(self.target)
        y_train = (self.target - gmean)/gstd

        #arrange in matrix format
        Y = (np.column_stack(y_train)).T

        return X, Y

    def train(self, X, Y):
        #compute and return beta
        return np.linalg.inv(X.T.dot(X)).dot(X.T).dot(Y)
    
    def predict(self, X_test,beta):
        #predict using beta
        Y_hat = X_test*beta.T
        return np.sum(Y_hat,axis=1)

#instantiate the linear_regression class  
lr_ols = linear_regression(happiness,gdp)

# preprocess the inputs
X,Y = lr_ols.preprocess()

#compute beta
beta = lr_ols.train(X,Y)

# use the computed beta for prediction
Y_predict = lr_ols.predict(X,beta)

# below code displays the predicted values

# access the 1st column (the 0th column is all 1's)
X_ = X[...,1].ravel()

#set the plot and plot size
fig, ax = plt.subplots()
fig.set_size_inches((15,8))

# display the X and Y points
ax.scatter(X_,Y)

#display the line predicted by beta and X
ax.plot(X_,Y_predict,color='r')

#set the x-labels
ax.set_xlabel("Happiness")

#set the x-labels
ax.set_ylabel("GDP per capita")

#set the title
ax.set_title("Cantril Ladder Score vs GDP per capita of countries (2018)")

#show the plot
plt.show()
