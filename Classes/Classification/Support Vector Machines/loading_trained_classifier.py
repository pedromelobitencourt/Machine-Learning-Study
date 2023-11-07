import numpy as np
import pickle

with open('credit.pkl', 'rb') as f:
    X_trainment, Y_trainment, X_test, Y_test = pickle.load(f)

X = np.concatenate((X_trainment, X_test))
Y = np.concatenate((Y_trainment, Y_test))

new_data = X[0]

svm = pickle.load(open('final_trained_svm_classifier.sav', 'rb'))

new_data = new_data.reshape(1, -1)

forecast = svm.predict(new_data)
print(forecast)