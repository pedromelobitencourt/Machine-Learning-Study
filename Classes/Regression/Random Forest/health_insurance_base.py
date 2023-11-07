import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor


base = pd.read_csv('health_insurance2.csv')

X = base.iloc[:, 0:1].values
Y = base.iloc[:, 1].values


n_estimators = 10

regression = RandomForestRegressor(n_estimators=n_estimators)
regression.fit(X, Y)

print(f'score: {regression.score(X, Y)}\n') # 1: perfect score


forecast = regression.predict(X)
print(forecast, '\n')


# Visualizing the 'splits'
X_test_tree = np.arange(min(X), max(X), 0.1) # from min(X) until max(X), incrementing 0.1 per time
X_test_tree = X_test_tree.reshape(-1, 1)

chart = px.scatter(x=X.ravel(), y=Y)
chart.add_scatter(x=X_test_tree.ravel(), y=regression.predict(X_test_tree), name='Regression')
chart.show()

print(f'A 40-year-old person will pay: {regression.predict([[40]])}')