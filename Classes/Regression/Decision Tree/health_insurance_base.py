import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


base = pd.read_csv('health_insurance2.csv')

X = base.iloc[:, 0:1].values
Y = base.iloc[:, 1].values

regression = DecisionTreeRegressor()
regression.fit(X, Y)

forecast = regression.predict(X)
print(forecast)

print(f'score: {regression.score(X, Y)}') # 1: perfect score

# Visualizing the 'splits'
X_test_tree = np.arange(min(X), max(X), 0.1) # from min(X) until max(X), incrementing 0.1 per time
X_test_tree = X_test_tree.reshape(-1, 1)

chart = px.scatter(x=X.ravel(), y=Y)
chart.add_scatter(x=X_test_tree.ravel(), y=regression.predict(X_test_tree), name='Regression')
chart.show()

print(f'A 40-year-old person will pay {regression.predict([[40]])}')