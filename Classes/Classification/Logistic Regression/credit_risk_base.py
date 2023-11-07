from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from yellowbrick.classifier import ConfusionMatrix
import numpy as np
import pickle

with open('credit_risk.pkl', 'rb') as f:
    X_credit_risk, Y_credit_risk = pickle.load(f) # We have already encoded it

# Removing 'moderado' value, because it's easier to use logistic regression if there is only 2 classes
X_credit_risk = np.delete(X_credit_risk, [2, 7, 11], axis=0) # Removing a line (axis=0)
Y_credit_risk = np.delete(Y_credit_risk, [2, 7, 11], axis=0)

logistic_credit_risk = LogisticRegression(random_state=1) # Always the same result
logistic_credit_risk.fit(X_credit_risk, Y_credit_risk) # trainment: finding the parameters

print(logistic_credit_risk.intercept_) # parameter B0
print(logistic_credit_risk.coef_) # [[B1 B2 B3 B4]]

# good credit report, high debt, no guarantee, income > 35000
# bad credit report, high debt, adequate guarantee, income < 15000
forecast = logistic_credit_risk.predict([[0, 0, 1, 2], [2, 0, 0, 0]])
print(forecast) # ['low', 'high']