from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report
from yellowbrick.classifier import ConfusionMatrix
from math import ceil
import pickle

with open('census.pkl', 'rb') as f:
    X_trainment, Y_trainment, X_test, Y_test = pickle.load(f)

entries = X_trainment.shape[1]
out = (Y_trainment.shape[1]) if len(Y_trainment.shape) > 1 else 1

hidden_sizes = ceil((entries + out) / 2)

neural_network = MLPClassifier(max_iter=500, tol=0.0000100,
                               solver='adam', activation='relu', hidden_layer_sizes=(hidden_sizes, hidden_sizes), verbose=True)
neural_network.fit(X_trainment, Y_trainment)

forecast = neural_network.predict(X_test)

accuracy = accuracy_score(Y_test, forecast)
print(accuracy)

print(classification_report(Y_test, forecast))

cm = ConfusionMatrix(neural_network)
cm.fit(X_trainment, Y_trainment)
cm.score(X_test, Y_test)
cm.show()