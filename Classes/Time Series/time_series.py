import pandas as pd
import numpy as np
import matplotlib.pylab as plt 
from datetime import datetime
from statsmodels.tsa.seasonal import seasonal_decompose
from pmdarima.arima import auto_arima

# Read the dataset
dataset = pd.read_csv('../../Databases/AirPassengers.csv')

# Parse dates using pd.to_datetime
dataset['Month'] = pd.to_datetime(dataset['Month'], format='%Y-%m')
dataset = dataset.set_index('Month')

time_series = dataset['#Passengers']
print(time_series)
print(time_series['1949-02'])
print(time_series[datetime(1949, 2, 1)])
print(time_series['1950-01-01':'1950-07-31'])
print(time_series[:'1950-07-31'])
print(time_series['1950'])
print('\n===============================================')

print(time_series.index.max()) # last index
print(time_series.index.min()) # initial index
print('\n===============================================')

plt.plot(time_series)
time_series_year = time_series.resample('A').sum()
plt.plot(time_series_year)

time_series_month = time_series.groupby([ lambda x: x.month ]).sum()
plt.plot(time_series_month)

time_series_dates = time_series['1960-01-01':'1960-12-01']
plt.plot(time_series_dates)

## Time Series Decomposition

decomposition = seasonal_decompose(time_series)
trend = decomposition.trend
seasonal = decomposition.seasonal
random = decomposition.resid

plt.plot(trend) # becoming higher or...
plt.plot(seasonal) # seasonal trending
plt.plot(random) # some random events that happened in that time (uncontrolled events or unexpected events)


## ARIMA Predictions

model = auto_arima(time_series, order=(2, 1, 2))
predictions = model.predict(n_periods=24) # how many years forwards until 1960 (last year) + 12 months

print('Predictions:\n', predictions)

train = time_series[:130]
test = time_series[130:]

model2 = auto_arima(train, suppress_warnings=True)

prediction = pd.DataFrame(model2.predict(n_periods=14), index=test.index)
prediction.columns = ['passengers_predictions']

print(prediction)

plt.figure(figsize=(8, 5))
plt.plot(train, label='Training')
plt.plot(test, label='Test')
plt.plot(prediction, label='Predictions')
plt.legend()
plt.show()