import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error


base = pd.read_csv('house_prices.csv')

X = base.iloc[:, 3:19].values
Y = base.iloc[:, 2].values

X_trainment, X_test, Y_trainment, Y_test = train_test_split(X, Y, test_size=0.3, random_state=1)


###################################################################
#                   STANDARDIZATION

X_scaler = StandardScaler()
X_trainment_scaled = X_scaler.fit_transform(X_trainment)

Y_scaler = StandardScaler()
Y_trainment_scaled = Y_scaler.fit_transform(Y_trainment.reshape(-1, 1))

X_test_scaled = X_scaler.transform(X_test)
Y_test_scaled = Y_scaler.transform(Y_test.reshape(-1, 1))

# print(X_trainment_scaled.shape, '\n')
# print(X_trainment_scaled, '\n')
# print(Y_trainment_scaled, '\n')

# print(X_test.shape, '\n')

regressor = MLPRegressor(max_iter=1000, hidden_layer_sizes=(9, 9))
regressor.fit(X_trainment_scaled, Y_trainment_scaled.ravel())

# print(regressor.score(X_trainment_scaled, Y_trainment_scaled))
# print(regressor.score(X_test_scaled, Y_test_scaled))

forecast = regressor.predict(X_test_scaled)

Y_test_inverse = Y_scaler.inverse_transform(Y_test_scaled)
inverse_forecast = Y_scaler.inverse_transform([forecast])

# inverse_forecast[0], inverse_forecast[1] = [inverse_forecast[1], inverse_forecast[0]]

print('\n\n', Y_test_inverse.shape, '\n\n\n', inverse_forecast.reshape(-1, 1).shape)

print(f'mean absolute error: {mean_absolute_error(Y_test_inverse, inverse_forecast.reshape(-1, 1))}')
print(f'mean squared error: {mean_squared_error(Y_test_inverse, inverse_forecast.reshape(-1, 1))}')
print(f'root mean squared error: {np.sqrt(mean_absolute_error(Y_test_inverse, inverse_forecast.reshape(-1, 1)))}\n')