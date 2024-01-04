import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
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


## Low Variance Technique: which attributes are the most important

for i in range(x_census.shape[1]):
    print(f'Variance of column {i}:', x_census_scaler[:, i].var())

selection = VarianceThreshold(threshold=0.05)
x_census_variance = selection.fit_transform(x_census_scaler)

print(selection.variances_)

indexes = np.where(selection.variances_ > 0.05)
print('Indexes:', indexes)
print(columns[indexes])

census_variance = census.drop(columns=['age', 'workclass', 'final-weight', 'education-num', 'race', 'capital-gain', 'capital-loos', 'hour-per-week', 'native-country'], axis=1)
print(census_variance)

x_census_variance = census_variance.iloc[:, 0:5].values
y_census_variance = census_variance.iloc[:, 5].values

x_census_variance[:, 0] = label_encoder_education.fit_transform(x_census_variance[:, 0])
x_census_variance[:, 1] = label_encoder_marital.fit_transform(x_census_variance[:, 1])
x_census_variance[:, 2] = label_encoder_occupation.fit_transform(x_census_variance[:, 2])
x_census_variance[:, 3] = label_encoder_relationship.fit_transform(x_census_variance[:, 3])
x_census_variance[:, 4] = label_encoder_sex.fit_transform(x_census_variance[:, 4])


onehot_encoder = ColumnTransformer(transformers=[('OneHot', OneHotEncoder(), [0, 1, 2, 3, 4])], remainder='passthrough')
x_census_variance = onehot_encoder.fit_transform(x_census_variance).toarray()
print('\nNew x_census:', x_census_variance.shape)

scaler = MinMaxScaler()
x_census_variance = scaler.fit_transform(x_census_variance)

print(x_census_variance)

x_census_training_over, x_census_test_over, y_census_training_over, y_census_test_over = train_test_split(x_census_variance, y_census_variance, test_size=0.15, random_state=0)

# 84.7% with the previous data
random_forest_census = RandomForestClassifier(criterion='entropy', min_samples_leaf=1, min_samples_split=5)
random_forest_census.fit(x_census_training_over, y_census_training_over)

forecasts = random_forest_census.predict(x_census_test_over)
accuracy = accuracy_score(y_census_test_over, forecasts)
print('accuracy:', accuracy) # 81.78%