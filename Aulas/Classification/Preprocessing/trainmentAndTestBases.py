import pandas as pd # Load csv files
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler # OneHotEncoder: To create columns for each attribute value
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split # Split the dataset into training and test sets
import pickle # Save the trainment and test sets

# ======================================================================================================================================
                                        # CREDIT DATA BASE
base_credit = pd.read_csv("credit_data.csv")

mean = base_credit['age'][base_credit['age'] > 0].mean()
base_credit.loc[base_credit['age'] < 0, 'age'] = mean

base_credit['age'].fillna(base_credit['age'].mean(), inplace=True)

X_credit = base_credit.iloc[:, 1:4].values

Y_credit = base_credit.iloc[:, 4].values

scaler_credit = StandardScaler()
X_credit = scaler_credit.fit_transform(X_credit)
# ======================================================================================================================================


# ======================================================================================================================================
                                            # CENSUS DATA BASE
base_census = pd.read_csv('census.csv')
X_census = base_census.iloc[:, 0:14].values
Y_census = base_census.iloc[:, 14].values

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

onehotencoder_census = ColumnTransformer(transformers=[('OneHot', OneHotEncoder(), [1, 3, 5, 6, 7, 8, 9, 13])], remainder='passthrough')

X_census = onehotencoder_census.fit_transform(X_census).toarray()
scaler_census = StandardScaler()
X_census = scaler_census.fit_transform(X_census)


# ======================================================================================================================================

# You have to put 'random_state', because if you don't, the algorithm will split the dataset in a different way every time you run the code
# test_size: 25% of the database
X_credit_trainment, X_credit_test, Y_credit_trainment, Y_credit_test = train_test_split(X_credit, Y_credit, test_size=0.25, random_state=0)

print(X_credit_trainment.shape) # (rows, columns); columns: age, income, loan
print(Y_credit_trainment.shape, '\n') # (rows,); columns: default (it will pay or not)

X_census_trainment, X_census_test, Y_census_trainment, Y_census_test = train_test_split(X_census, Y_census, test_size=0.15, random_state=0)

print(X_census_trainment.shape, X_census_test.shape)
print(Y_census_trainment.shape, Y_census_test.shape)


# ======================================================================================================================================
                                            # Saving the datebases in the current dir
with open('credit.pkl', mode='wb') as f:
    pickle.dump([X_credit_trainment, Y_credit_trainment, X_credit_test, Y_credit_test], f)

with open('census.pkl', mode='wb') as f:
    pickle.dump([X_census_trainment, Y_census_trainment, X_census_test, Y_census_test], f)