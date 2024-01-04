import pandas as pd 
import numpy as np
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from imblearn.under_sampling import TomekLinks

census_base = pd.read_csv('../../Databases/census.csv')
print(np.unique(census_base['income'], return_counts=True))

# sns.countplot(x=census_base['income'])

## Preprocessing

x_census = census_base.iloc[:, 0:14].values
y_census = census_base.iloc[:, 14].values

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

## undersampling processing

tl = TomekLinks(sampling_strategy='all')
x_under, y_under = tl.fit_resample(x_census, y_census)

print(x_under.shape, y_under.shape)
print('y_census:', np.unique(y_census, return_counts=True))
print('y_under:', np.unique(y_under, return_counts=True))

onehot_encoder = ColumnTransformer(transformers=[('OneHot', OneHotEncoder(), [1, 3, 5, 6, 7, 8, 9, 13])], remainder='passthrough')
x_census = onehot_encoder.fit_transform(x_under).toarray()
print('\nNew x_census:', x_census.shape, x_under.shape)

x_census_training_under, x_census_test_under, y_census_training_under, y_census_test_under = train_test_split(x_under, y_under, test_size=0.15, random_state=0)

# 84.7% with the previous data
random_forest_census = RandomForestClassifier(criterion='entropy', min_samples_leaf=1, min_samples_split=5, n_estimators=100)
random_forest_census.fit(x_census_training_under, y_census_training_under)

forecasts = random_forest_census.predict(x_census_test_under)
accuracy = accuracy_score(y_census_test_under, forecasts)
print('accuracy:', accuracy) # 89.38%