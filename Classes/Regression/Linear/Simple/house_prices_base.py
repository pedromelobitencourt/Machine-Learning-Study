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


print(base.isnull().sum(), '\n') # There is no 'null' values

# figure = plt.figure(figsize=(20, 20))
# print(base.corr(), '\n') # Verifying the correlations

# sns.heatmap(base.corr(), annot=True)
# plt.show()

X = base.iloc[:, 4:5].values
Y = base.iloc[:, 1].values

X_trainment, X_test, Y_trainment, Y_test = train_test_split(X, Y, test_size=0.3, random_state=0)

linear_regression = LinearRegression()
linear_regression.fit(X_trainment, Y_trainment)

print(f'b0: {linear_regression.intercept_}\n')
print(f'b1: {linear_regression.coef_}\n')

print(f'score: {linear_regression.score(X_trainment, Y_trainment)}')
print(f'score: {linear_regression.score(X_test, Y_test)}\n')

forecast_trainment = linear_regression.predict(X_trainment)


# chart = px.scatter(x=X_trainment.ravel(), y=forecast_trainment)
# chart.show()

# chart2 = px.line(x=X_trainment.ravel(), y=forecast_trainment)
# chart2.data[0].line.color = 'red'

# chart1 = px.scatter(x=X_trainment.ravel(), y=Y_trainment)

# chart3 = go.Figure(data=chart1.data + chart2.data)
# chart3.show()


forecast_test = linear_regression.predict(X_test)
abs_result = abs(Y_test - forecast_test)
print(abs_result, '\n')

mean_error = print(f'mean error: {abs_result.mean()}')

print(f'mean absolute error: {mean_absolute_error(Y_test, forecast_test)}') # same as 'mean_error'
print(f'mean square error: {mean_squared_error(Y_test, forecast_test)}')
print(f'root mean square error: {np.sqrt(mean_squared_error(Y_test, forecast_test))}')

# chart = px.scatter(x=X_trainment.ravel(), y=forecast_test)
# chart.show()

# chart2 = px.line(x=X_trainment.ravel(), y=forecast_test)
# chart2.data[0].line.color = 'red'

# chart1 = px.scatter(x=X_test.ravel(), y=Y_test)

# chart3 = go.Figure(data=chart1.data + chart2.data)
# chart3.show()