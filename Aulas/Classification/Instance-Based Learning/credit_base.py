from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
from yellowbrick.classifier import ConfusionMatrix
import pickle

with open('credit.pkl', 'rb') as f: # There is Standardization
    X_credit_trainment, Y_credit_trainment, X_credit_test, Y_credit_test = pickle.load(f)

knn_credit = KNeighborsClassifier(n_neighbors=10, metric='minkowski', p=2) # Distance metric
knn_credit.fit(X_credit_trainment, Y_credit_trainment) # Just storing the data

forecast = knn_credit.predict(X_credit_test)
print(forecast, '\n')

accuracy = accuracy_score(Y_credit_test, forecast)
print(accuracy, '\n')

cm = ConfusionMatrix(knn_credit)
cm.fit(X_credit_trainment, Y_credit_trainment)
cm.score(X_credit_test, Y_credit_test)
cm.show()

print(classification_report(Y_credit_test, forecast))