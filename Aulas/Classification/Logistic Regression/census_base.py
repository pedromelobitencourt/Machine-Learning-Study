from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from yellowbrick.classifier import ConfusionMatrix
import numpy as np
import pickle

with open('census.pkl', 'rb') as f:
    X_trainment, Y_trainment, X_test, Y_test = pickle.load(f)

logistic_census = LogisticRegression(random_state=1)
logistic_census.fit(X_trainment, Y_trainment)

print(logistic_census.intercept_)
print(logistic_census.coef_)

forecast = logistic_census.predict(X_test)

accuracy = accuracy_score(Y_test, forecast)
print(accuracy)

cm = ConfusionMatrix(logistic_census)
cm.fit(X_trainment, Y_trainment)
cm.score(X_test, Y_test)
cm.show()

print(classification_report(Y_test, forecast))