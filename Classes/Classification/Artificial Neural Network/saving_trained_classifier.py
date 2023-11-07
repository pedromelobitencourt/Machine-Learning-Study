from sklearn.neural_network import MLPClassifier
import numpy as np
import pickle

with open('credit.pkl', 'rb') as f:
    X_trainment, Y_trainment, X_test, Y_test = pickle.load(f)

X = np.concatenate((X_trainment, X_test))
Y = np.concatenate((Y_trainment, Y_test))

# After all analysis

neural_network_classifier = MLPClassifier(activation='relu', batch_size=56, solver='adam')

# Final model: We can train all data base
neural_network_classifier.fit(X, Y)
pickle.dump(neural_network_classifier, open('final_trained_neural_network_classifier.sav', 'wb')) # Saving the classifier