import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from yellowbrick.regressor import ResidualsPlot

base = pd.read_csv('house_prices.csv')
column_to_remove = 'date'
base = base.drop(columns=[column_to_remove])

X = base.iloc[:, 2:19].values
Y = base.iloc[:, 1].values

X_trainment, X_test, Y_trainment, Y_test = train_test_split(X, Y, test_size=0.3, random_state=0)

multiple_regression = LinearRegression()
multiple_regression.fit(X_trainment, Y_trainment)

print(f'b0: {multiple_regression.intercept_}')
print(f'bi: {multiple_regression.coef_}\n')

print(f'trainment score: {multiple_regression.score(X_trainment, Y_trainment)}')
print(f'test score: {multiple_regression.score(X_test, Y_test)}\n') # Better than simple linear regression


forecast_test = multiple_regression.predict(X_test)

abs_result = abs(Y_test - forecast_test)
mean_error = print(f'mean error: {abs_result.mean()}')

print(f'mean absolute error: {mean_absolute_error(Y_test, forecast_test)}') # same as 'mean_error'
print(f'mean square error: {mean_squared_error(Y_test, forecast_test)}')
print(f'root mean square error: {np.sqrt(mean_squared_error(Y_test, forecast_test))}')