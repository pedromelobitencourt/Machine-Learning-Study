import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler


base = pd.read_csv('health_insurance2.csv')

X = base.iloc[:, 0:1].values
Y = base.iloc[:, 1].values


n_estimators = 10

# Kernels: linear, polynomial, rbf

# regression = SVR(kernel='linear')
# regression = SVR(kernel='poly', degree=3)
regression = SVR(kernel='rbf')


# If kernel == 'linear': It looks like 'linear regression': It makes a line
# If kernel == 'poly': It looks like 'polynomial regression': It makes a curve
# regression.fit(X, Y)

# If kernel == 'rbf': It is almost a constant line, because the data is not normalized and it needs to be because of 'rbf'


#############################################################################
#                               FOR kernel == 'rbf'
scaler_x = StandardScaler()
X_scaled = scaler_x.fit_transform(X)

scaler_y = StandardScaler()
Y_scaled = scaler_y.fit_transform(Y.reshape(-1, 1))

regression.fit(X_scaled, Y_scaled.ravel())

print(f'score: {regression.score(X_scaled, Y_scaled.ravel())}\n') 


forecast = regression.predict(X_scaled)
print(forecast, '\n')


# Visualizing the 'splits'
X_test_tree = np.arange(min(X_scaled), max(X_scaled), 0.1) # from min(X) until max(X), incrementing 0.1 per time
X_test_tree = X_test_tree.reshape(-1, 1)

chart = px.scatter(x=X_scaled.ravel(), y=Y_scaled.ravel())
chart.add_scatter(x=X_test_tree.ravel(), y=regression.predict(X_test_tree), name='Regression')
chart.show()

new_data = [[40]]
new_data = scaler_x.transform(new_data)

print(new_data)

print(f'A 40-year-old person will pay: {scaler_y.inverse_transform(regression.predict(new_data))}')
############################################################## 


# print(f'score: {regression.score(X, Y)}\n') 


# forecast = regression.predict(X)
# print(forecast, '\n')


# # Visualizing the 'splits'
# X_test_tree = np.arange(min(X), max(X), 0.1) # from min(X) until max(X), incrementing 0.1 per time
# X_test_tree = X_test_tree.reshape(-1, 1)

# chart = px.scatter(x=X.ravel(), y=Y)
# chart.add_scatter(x=X_test_tree.ravel(), y=regression.predict(X_test_tree), name='Regression')
# chart.show()

# print(f'A 40-year-old person will pay: {regression.predict([[40]])}')