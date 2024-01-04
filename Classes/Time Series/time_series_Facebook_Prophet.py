import pandas as pd
from prophet import Prophet

dataset = pd.read_csv('../../Databases/page_wikipedia.csv')
print(dataset.describe(), '\n=============================================')
print(dataset.hist(), '\n=============================================')

dataset = dataset[['date', 'views']].rename(columns={'date': 'ds', 'views': 'y'})
print(dataset, '\n=============================================')

dataset = dataset.sort_values(by='ds') # we must sort it by date
print(dataset, '\n=============================================')


## Creating the model and Predictions

model = Prophet()
model.fit(dataset)

future = model.make_future_dataframe(periods=90) # 90 days
forecast = model.predict(future)

print(forecast.head(), '\n=============================================')
print(len(forecast), '\n=============================================')
print(len(dataset), '\n=============================================')

model.plot(forecast, xlabel='Date', ylabel='Views')
model.plot_components(forecast)