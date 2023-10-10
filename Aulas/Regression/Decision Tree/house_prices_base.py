import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error


base = pd.read_csv('house_prices.csv')

X = base.iloc[:, 3:18].values
Y = base.iloc[:, 2].values

X_trainment, X_test, Y_trainment, Y_test = train_test_split(X, Y, test_size=0.3, random_state=0)

regression = DecisionTreeRegressor()
regression.fit(X_trainment, Y_trainment)

forecast = regression.predict(X_test)
print(forecast)

print(f'trainment score: {regression.score(X_trainment, Y_trainment)}') # Almost 1 (a perfect score, but this is expected)
print(f'test score: {regression.score(X_test, Y_test)}\n') # around 72%

print(f'mean absolute error: {mean_absolute_error(Y_test, forecast)}')
print(f'mean squared error: {mean_squared_error(Y_test, forecast)}')
print(f'root mean squared error: {np.sqrt(mean_absolute_error(Y_test, forecast))}\n')


# Visualizing the 'splits'
# X_test_tree = np.arange(min(X), max(X), 0.1) # from min(X) until max(X), incrementing 0.1 per time
# X_test_tree = X_test_tree.reshape(-1, 1)

# chart = px.scatter(x=X_trainment.ravel(), y=Y_trainment)
# chart.show()