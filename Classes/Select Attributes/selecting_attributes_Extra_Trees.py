import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import VarianceThreshold

census = pd.read_csv('../../Databases/census.csv')

columns = census.columns[:-1] # columns names

x_census = census.iloc[:, 0:14].values
y_census = census.iloc[:, 14].values

label_encoder_workclass = LabelEncoder()
label_encoder_education = LabelEncoder()
label_encoder_marital = LabelEncoder()
label_encoder_occupation = LabelEncoder()
label_encoder_relationship = LabelEncoder()
label_encoder_race = LabelEncoder()
label_encoder_sex = LabelEncoder()
label_encoder_country = LabelEncoder()

x_census[:, 1] = label_encoder_workclass.fit_transform(x_census[:, 1])
x_census[:, 3] = label_encoder_education.fit_transform(x_census[:, 3])
x_census[:, 5] = label_encoder_marital.fit_transform(x_census[:, 5])
x_census[:, 6] = label_encoder_occupation.fit_transform(x_census[:, 6])
x_census[:, 7] = label_encoder_relationship.fit_transform(x_census[:, 7])
x_census[:, 8] = label_encoder_race.fit_transform(x_census[:, 8])
x_census[:, 9] = label_encoder_sex.fit_transform(x_census[:, 9])
x_census[:, 13] = label_encoder_country.fit_transform(x_census[:, 13])

scaler = MinMaxScaler()
x_census_scaler = scaler.fit_transform(x_census)


## Extra Trees Technique: which attributes are the most important

selection = ExtraTreesClassifier()
selection.fit(x_census_scaler, y_census)

relevant = selection.feature_importances_
print('more relavent attributes:', relevant)

indexes = []
threshold = 0.029

for i in range(len(relevant)):
    if relevant[i] >= threshold:
        indexes.append(i)

print('indexes: ', indexes)
print(columns[indexes])

x_census_extra = x_census[:, indexes]
print('extra:', x_census_extra)

onehot_encoder = ColumnTransformer(transformers=[('OneHot', OneHotEncoder(), [1, 3, 5, 7])], remainder='passthrough')
x_census_extra = onehot_encoder.fit_transform(x_census_extra).toarray()
print('\nNew x_census:', x_census_extra.shape)

x_census_training_extra, x_census_test_extra, y_census_training_extra, y_census_test_extra = train_test_split(x_census_extra, y_census, test_size=0.15, random_state=0)

# 84.7% with the previous data
random_forest_census = RandomForestClassifier(criterion='entropy', min_samples_leaf=1, min_samples_split=5, n_estimators=100)
random_forest_census.fit(x_census_training_extra, y_census_training_extra)

forecasts = random_forest_census.predict(x_census_test_extra)
accuracy = accuracy_score(y_census_test_extra, forecasts)
print('accuracy:', accuracy) # 84.60%