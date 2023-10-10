import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from yellowbrick.regressor import ResidualsPlot

base = pd.read_csv('house_prices.csv')
column_to_remove = 'date'
base = base.drop(columns=[column_to_remove])

X = base.iloc[:, 2:18].values
Y = base.iloc[:, 1].values

X_trainment, X_test, Y_trainment, Y_test = train_test_split(X, Y, test_size=0.3, random_state=0)

print(X_trainment.shape)

poly = PolynomialFeatures(degree=2)
X_trainment_poly = poly.fit_transform(X_trainment)
X_test_poly = poly.transform(X_test)

regression = LinearRegression()
regression.fit(X_trainment_poly, Y_trainment)

print(f'trainment score: {regression.score(X_trainment_poly, Y_trainment)}')
print(f'test score: {regression.score(X_test_poly, Y_test)}')

forecast = regression.predict(X_trainment_poly)