from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
from yellowbrick.classifier import ConfusionMatrix
import matplotlib.pyplot as plt
import pickle

with open('credit.pkl', 'rb') as f:
    X_credit_trainment, Y_credit_trainment, X_credit_test, Y_credit_test = pickle.load(f)

trees_qnt = 40
credit_random_forest = RandomForestClassifier(n_estimators=trees_qnt, criterion='entropy', random_state=0)
credit_random_forest.fit(X_credit_trainment, Y_credit_trainment)

forecast = credit_random_forest.predict(X_credit_test)

accuracy = accuracy_score(Y_credit_test, forecast)
print(accuracy) # 98.4%: Is it worth it?

cm = ConfusionMatrix(credit_random_forest)
cm.fit(X_credit_trainment, Y_credit_trainment)
cm.score(X_credit_test, Y_credit_test)
cm.show()

print(classification_report(Y_credit_test, forecast))