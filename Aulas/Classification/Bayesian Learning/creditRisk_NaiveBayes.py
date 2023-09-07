import pandas as pd # Load csv files
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split # Split the dataset into training and test sets
from sklearn.naive_bayes import GaussianNB # Naive Bayes
import pickle # Save the trainment and test sets


# ==================================================================================================
                                   # Loading the datebase and preprocessing

base_credit_risk = pd.read_csv('risco_credito.csv')

X_credit_risk = base_credit_risk.iloc[:, 0:4].values
Y_credit_risk = base_credit_risk.iloc[:, 4].values

print(X_credit_risk.shape) # (14, 4)
print(Y_credit_risk.shape) # (14,)

label_encoder_history = LabelEncoder()
label_encoder_debt = LabelEncoder()
label_encoder_warranty = LabelEncoder()
label_encoder_income = LabelEncoder()

X_credit_risk[:, 0] = label_encoder_history.fit_transform(X_credit_risk[:, 0])
X_credit_risk[:, 1] = label_encoder_debt.fit_transform(X_credit_risk[:, 1])
X_credit_risk[:, 2] = label_encoder_warranty.fit_transform(X_credit_risk[:, 2])
X_credit_risk[:, 3] = label_encoder_income.fit_transform(X_credit_risk[:, 3])


with open('credit_risk.pkl', 'wb') as f:
    pickle.dump([X_credit_risk, Y_credit_risk], f)

# ==================================================================================================
                                      # Naive Bayes
naive_credit_risk = GaussianNB()
naive_credit_risk.fit(X_credit_risk, Y_credit_risk) # Predictors and classes

# historia boa (0), dívida alta (0), garantias nenhuma (1), renda > 35 (2) 
# historia ruim (2), dívida alta (0), garantias adequada (0), renda < 15 (0)

forecast = naive_credit_risk.predict([[0, 0, 1, 2], [2, 0, 0, 0]]) # Laplacian Correction is implicit done
print(forecast) # ['baixo', 'moderado']
print(naive_credit_risk.classes_) # ['alto', 'baixo', 'moderado']
print(naive_credit_risk.class_count_) # [6. 5. 3.]
print(naive_credit_risk.class_prior_) # [0.428 0.357 0.214]; Probabilities