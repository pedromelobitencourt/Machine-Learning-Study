from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
from yellowbrick.classifier import ConfusionMatrix
import matplotlib.pyplot as plt
import pickle

with open('census.pkl', 'rb') as f:
    X_census_trainment, Y_census_trainment, X_census_test, Y_census_test = pickle.load(f)

tree_census = DecisionTreeClassifier(criterion='entropy', random_state=0)
tree_census.fit(X_census_trainment, Y_census_trainment)

forecast = tree_census.predict(X_census_test)

accuracy = accuracy_score(Y_census_test, forecast)
print(accuracy)

cm = ConfusionMatrix(tree_census)
cm.fit(X_census_trainment, Y_census_trainment)
cm.score(X_census_test, Y_census_test)
cm.show()

print(classification_report(Y_census_test, forecast))