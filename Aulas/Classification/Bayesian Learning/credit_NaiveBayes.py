import pandas as pd # Load csv files
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split # Split the dataset into training and test sets
from sklearn.naive_bayes import GaussianNB # Naive Bayes
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from yellowbrick.classifier import ConfusionMatrix
import pickle # Save the trainment and test sets

with open('credit.pkl', 'rb') as f:
    X_credit_trainment, Y_credit_trainment, X_credit_test, Y_credit_test = pickle.load(f)

naive_credit_data = GaussianNB()
naive_credit_data.fit(X_credit_trainment, Y_credit_trainment) # Generate the probability table

# Testing
forecast = naive_credit_data.predict(X_credit_test)

# Compare the results (the real values and the predictors values)
print(accuracy_score(Y_credit_test, forecast)) # 0.938 = 93.8% of accuracy
print(confusion_matrix(Y_credit_test, forecast)) # [[423, 8], [23, 41]]; 423 pay the loan and was correctly chosen

# Does the same thing as the 'accuracy_score', but graph
cm = ConfusionMatrix(naive_credit_data)
cm.fit(X_credit_trainment, Y_credit_trainment)
cm.score(X_credit_test, Y_credit_test)
cm.show()

print(classification_report(Y_credit_test, forecast))