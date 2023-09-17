from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
from yellowbrick.classifier import ConfusionMatrix
import matplotlib.pyplot as plt
import pickle

with open('census.pkl', 'rb') as f:
    X_census_trainment, Y_census_trainment, X_census_test, Y_census_test = pickle.load(f)

trees_qnt = 100
census_random_forest = RandomForestClassifier(n_estimators=trees_qnt, criterion='entropy', random_state=0)
census_random_forest.fit(X_census_trainment, Y_census_trainment)

forecast = census_random_forest.predict(X_census_test)
accuracy = accuracy_score(Y_census_test, forecast)

print(accuracy)

cm = ConfusionMatrix(census_random_forest)
cm.fit(X_census_trainment, Y_census_trainment)
cm.score(X_census_test, Y_census_test)
cm.show()

print(classification_report(Y_census_test, forecast))