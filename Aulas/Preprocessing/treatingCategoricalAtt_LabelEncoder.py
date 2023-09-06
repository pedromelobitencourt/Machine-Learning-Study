import pandas as pd # Load csv files
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
import numpy as np
import seaborn as sns # View graphs
import matplotlib.pyplot as plt # View graphs
import plotly.express as px # Create a dynamic graph

base_census = pd.read_csv('census.csv')

# We will turn the categorical variables into numeric variables

X_census = base_census.iloc[:, 0:14].values
Y_census = base_census.iloc[:, 14].values

label_encoder_test = LabelEncoder()
test = label_encoder_test.fit_transform(X_census[:, 1]) # Each unique value of the column will be assigned to a number
print(test)

label_encoder_workclass = LabelEncoder()
label_encoder_education = LabelEncoder()
label_encoder_marital = LabelEncoder()
label_encoder_occupation = LabelEncoder()
label_encoder_relationship = LabelEncoder()
label_encoder_race = LabelEncoder()
label_encoder_sex = LabelEncoder()
label_encoder_country = LabelEncoder()

X_census[:, 1] = label_encoder_workclass.fit_transform(X_census[:, 1])
X_census[:, 3] = label_encoder_education.fit_transform(X_census[:, 3])
X_census[:, 5] = label_encoder_marital.fit_transform(X_census[:, 5])
X_census[:, 6] = label_encoder_occupation.fit_transform(X_census[:, 6])
X_census[:, 7] = label_encoder_relationship.fit_transform(X_census[:, 7])
X_census[:, 8] = label_encoder_race.fit_transform(X_census[:, 8])
X_census[:, 9] = label_encoder_sex.fit_transform(X_census[:, 9])
X_census[:, 13] = label_encoder_country.fit_transform(X_census[:, 13])

print(X_census[3])

# One problem using only Label Encoder is that it creates a lot of categorie. Moreover, it gives numbers to attributes, but
# machine learning algorithms may treat a value as more important than other because of it

# To fix this problem, we will use One Hot Encoder; It creates some columns

# Gol Palio Uno
#  1    2    3
# Gol: 1 0 0
# Palio: 0 1 0
# Uno: 0 0 1

# remainer='passthrough': It won't erase the attributes that are not in the index list
onehotencoder_census = ColumnTransformer(transformers=[('OneHot', OneHotEncoder(), [1, 3, 5, 6, 7, 8, 9, 13])], remainder='passthrough')

X_census = onehotencoder_census.fit_transform(X_census).toarray()
print(X_census[1]) # Now there will be more columns, because each attribute value (of categorical attributes) will be a column
print(X_census.shape) # (rows, columns)


# ======================================================================================================================================

                # Scheduling values (standardization)

# ======================================================================================================================================

scaler_census = StandardScaler()
X_census = scaler_census.fit_transform(X_census) # Putting all values in the same scale