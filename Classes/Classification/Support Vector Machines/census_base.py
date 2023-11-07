from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from yellowbrick.classifier import ConfusionMatrix
import pickle

with open('census.pkl', 'rb') as f:
    X_trainment, Y_trainment, X_test, Y_test = pickle.load(f)

svm_census = SVC(kernel='linear', random_state=1, C=1.2)
svm_census.fit(X_trainment, Y_trainment)

forecast = svm_census.predict(X_test)

accuracy = accuracy_score(Y_test, forecast)
print(accuracy)

print(classification_report(Y_test, forecast))

cm = ConfusionMatrix(svm_census)
cm.fit(X_trainment, Y_trainment)
cm.score(X_test, Y_test)
cm.show()