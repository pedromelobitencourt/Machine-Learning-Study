from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from yellowbrick.classifier import ConfusionMatrix
import pickle

with open('credit.pkl', 'rb') as f:
    X_trainment, Y_trainment, X_test, Y_test = pickle.load(f)

# High C can lead to overfitting the training data base, unlike the Low C, but this one may have some training errors
svm_credit = SVC(kernel='rbf', random_state=1, C=1.6)
svm_credit.fit(X_trainment, Y_trainment)

forecast = svm_credit.predict(X_test)

accuracy = accuracy_score(Y_test, forecast)
print(accuracy)

print(classification_report(Y_test, forecast))

cm = ConfusionMatrix(svm_credit)
cm.fit(X_trainment, Y_trainment)
cm.score(X_test, Y_test)
cm.show()