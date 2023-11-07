import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_absolute_error, mean_squared_error


base = pd.read_csv('health_insurance2.csv')

X = base.iloc[:, 0:1].values
Y = base.iloc[:, 1].values

poly = PolynomialFeatures(degree=4) # 'degree': n

X_poly = poly.fit_transform(X)

print(X_poly.shape) # 3 columns: power(Xi, 0); power(Xi, 1); power(Xi, 2)


regression = LinearRegression()
regression.fit(X_poly, Y)

print(f'b0: {regression.intercept_}')
print(f'bi: {regression.coef_}\n')

new_data = [[40]]
new_data = poly.fit_transform(new_data)
print(regression.predict(new_data))


forecast = regression.predict(X_poly)

chart = px.scatter(x= X[:,0], y=Y)
chart.add_scatter(x=X[:, 0], y=forecast, name='Regression')
chart.show()