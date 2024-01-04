import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

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

census_scaler = StandardScaler()
x_census = census_scaler.fit_transform(x_census)

x_training, x_test, y_training, y_test = train_test_split(x_census, y_census, test_size=0.15, random_state=0)
print(x_training.shape, x_test.shape)

## PCA

pca = PCA(n_components=8)

x_training_pca = pca.fit_transform(x_training)
x_test_pca = pca.transform(x_test)

print(x_training_pca.shape, x_test_pca.shape)

print(pca.explained_variance_ratio_) # the n_components explain n% of the variables

# 84.7% with the previous data
random_forest_census = RandomForestClassifier(criterion='entropy', min_samples_leaf=1, min_samples_split=5, n_estimators=40)
random_forest_census.fit(x_training_pca, y_training)

forecasts = random_forest_census.predict(x_test_pca)
accuracy = accuracy_score(y_test, forecasts)
print('accuracy:', accuracy) # 83.44%