import pandas as pd
import numpy as np
from pyod.models.knn import KNN

credit = pd.read_csv('../../Databases/credit_data.csv')
credit.dropna(inplace=True)

detector = KNN()
detector.fit(credit.iloc[:, 1:4])

forecasts = detector.labels_
print('Forecast:', forecasts) # 0: not an outlier; 1: an outlier
print(np.unique(forecasts, return_counts=True), '\n=========================================')

forecast_trust = detector.decision_scores_
print('Forecasts Trust:', forecast_trust, '\n=========================================')

outliers = []

for i in range(len(forecasts)):
    if forecasts[i] == 1:
        outliers.append(i)

outliers_list = credit.iloc[outliers, :]
print('Outliers list:', outliers_list)