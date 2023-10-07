from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report
from yellowbrick.classifier import ConfusionMatrix
import pickle

with open('credit.pkl', 'rb') as f:
    X_trainment, Y_trainment, X_test, Y_test = pickle.load(f)

# Convergent Warning: It indicates that the error can be lowered yet (you can increase 'max_iter')
# 'verbose' attribute = True: It show the error in each iteration
# iteration = ephocs

# If the training ends before the iteration you determined, it's because of 'tol' and 'num_iter_no_change'
# 'tol': How much the error should decrease
# 'num_iter_no_change': How many iterations are allowed to not satisfy the 'tol' condition consecutively
# 'solver': The function that will adjust the weights
# 'activation': Activation functions
# 'hidden_layer_sizes': Self explanatory -> (100, 100): Two hidden layers with 100 neurons each one

neural_network = MLPClassifier(max_iter=1500, solver='adam', activation='relu', tol=0.0000100,
                               hidden_layer_sizes=(2, 2), verbose=True)
neural_network.fit(X_trainment, Y_trainment)

forecast = neural_network.predict(X_test)

accuracy = accuracy_score(Y_test, forecast)
print(accuracy)

print(classification_report(Y_test, forecast))

cm = ConfusionMatrix(neural_network)
cm.fit(X_trainment, Y_trainment)
cm.score(X_test, Y_test)
cm.show()
