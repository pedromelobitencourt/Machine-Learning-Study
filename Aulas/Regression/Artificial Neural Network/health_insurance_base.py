import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler


base = pd.read_csv('health_insurance2.csv')

X = base.iloc[:, 0:1].values
Y = base.iloc[:, 1].values

scaler_x = StandardScaler()
X_scaled = scaler_x.fit_transform(X)

scaler_y = StandardScaler()
Y_scaled = scaler_y.fit_transform(Y.reshape(-1, 1))

regressor = MLPRegressor(max_iter=450)
regressor.fit(X_scaled, Y_scaled.ravel())

print(regressor.score(X_scaled, Y_scaled))

####################################################################
                        # Generating a chart
chart = px.scatter(x=X_scaled.ravel(), y=Y_scaled.ravel())
chart.add_scatter(x=X_scaled.ravel(), y=regressor.predict(X_scaled), name='Regression')
chart.show()

####################################################################


new_data = [[40]]
new_data = scaler_x.transform(new_data)

scaled_result = [regressor.predict(new_data)]

result = scaler_y.inverse_transform(scaled_result)
print(result)