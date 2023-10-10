import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error


base = pd.read_csv('house_prices.csv')

X = base.iloc[:, 3:18].values
Y = base.iloc[:, 2].values

X_trainment, X_test, Y_trainment, Y_test = train_test_split(X, Y, test_size=0.3, random_state=1)


###################################################################
#                   STANDARDIZATION

X_scaler = StandardScaler()
X_trainment_scaled = X_scaler.fit_transform(X_trainment)

Y_scaler = StandardScaler()
Y_trainment_scaled = Y_scaler.fit_transform(Y_trainment.reshape(-1, 1))

# Always check if they have the same number of data
print(X_trainment_scaled.shape)
print(Y_trainment_scaled.shape, '\n')

X_test_scaled = X_scaler.transform(X_test)
Y_test_scaled = Y_scaler.transform(Y_test.reshape(-1, 1))
####################################################################


regression = SVR(kernel='rbf')
regression.fit(X_trainment_scaled, Y_trainment_scaled.ravel())

print(f'trainment score: {regression.score(X_trainment_scaled, Y_trainment_scaled)}')
print(f'test score: {regression.score(X_test_scaled, Y_test_scaled)}')

forecast = regression.predict(X_test_scaled)

Y_test_inverse = Y_scaler.inverse_transform(Y_test_scaled)
forecast_inverse = Y_scaler.inverse_transform(forecast)

print('forecast: ', forecast_inverse, '\n')
print(f'y_test: {Y_test_inverse}\n')

print(f'mean absolute error: {mean_absolute_error(Y_test_inverse, forecast_inverse)}')
print(f'mean squared error: {mean_squared_error(Y_test_inverse, forecast_inverse)}')
print(f'root mean squared error: {np.sqrt(mean_absolute_error(Y_test_inverse, forecast_inverse))}\n')


# Visualizing the 'splits'
# X_test_tree = np.arange(min(X), max(X), 0.1) # from min(X) until max(X), incrementing 0.1 per time
# X_test_tree = X_test_tree.reshape(-1, 1)

# chart = px.scatter(x=X_trainment.ravel(), y=Y_trainment)
# chart.show()