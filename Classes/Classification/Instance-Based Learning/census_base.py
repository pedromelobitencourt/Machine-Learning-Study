from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
from yellowbrick.classifier import ConfusionMatrix
import pickle

with open('census.pkl', 'rb') as f:
    X_census_trainment, Y_census_trainment, X_census_test, Y_census_test = pickle.load(f)

knn_census = KNeighborsClassifier(n_neighbors=10, metric='minkowski', p=2)
knn_census.fit(X_census_trainment, Y_census_trainment)

forecast = knn_census.predict(X_census_test)

accuracy = accuracy_score(Y_census_test, forecast)
print(accuracy, '\n')

cm = ConfusionMatrix(knn_census)
cm.fit(X_census_trainment, Y_census_trainment)
cm.score(X_census_test, Y_census_test)
cm.show()

print(classification_report(Y_census_test, forecast))