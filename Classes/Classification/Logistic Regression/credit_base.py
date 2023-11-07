from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from yellowbrick.classifier import ConfusionMatrix
import numpy as np
import pickle

with open('credit.pkl', 'rb') as f:
    X_trainment, Y_trainment, X_test, Y_test = pickle.load(f)

logistic_credit = LogisticRegression(random_state=1)
logistic_credit.fit(X_trainment, Y_trainment)

print(logistic_credit.intercept_) #B0
print(logistic_credit.coef_)

forecast = logistic_credit.predict(X_test)

accuracy = accuracy_score(Y_test, forecast)
print(accuracy)

cm = ConfusionMatrix(logistic_credit)
cm.fit(X_trainment, Y_trainment)
cm.score(X_test, Y_test)
cm.show()

print(classification_report(Y_test, forecast))