import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from yellowbrick.regressor import ResidualsPlot

base = pd.read_csv('health_insurance.csv')

X = base.iloc[:, 0].values # All lines, first column only
Y = base.iloc[:, 1].values # All lines, second column only

print(X, '\n') # Age
print(Y, '\n') # Cost

# Correlation coefficient
print(np.corrcoef(X, Y), '\n') # Around 93% of 'cost' is correlated to 'age'

# If there is a strong correlation coefficient, you can use linear regression

X = X.reshape(-1, 1) # Turn into a matrix

regression = LinearRegression()
regression.fit(X, Y) # Find the coefficients

#b0
print(f'b0: {regression.intercept_}')

#b1
print(f'b1: {regression.coef_}\n')


forecast = regression.predict(X)
print(forecast, '\n')


# Making a chart to analyse the linear regression
chart = px.scatter(x=X.ravel(), y=Y) # 'ravel': Turning it into vector again
chart.add_scatter(x=X.ravel(), y=forecast, name='Regression')
chart.show()

# Test
print(regression.predict([[40]]), '\n')


# Algorithm's quality
print(regression.score(X, Y), '\n')


viewer = ResidualsPlot(regression)
viewer.fit(X, Y)
viewer.poof()